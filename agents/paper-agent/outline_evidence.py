"""Build verifiable measurements for an outline from the prepared run facts."""
import hashlib
import json
import math
from decimal import Decimal


def reference(fact):
    source = fact["source"]
    return source["file"] + "#" + source["json_pointer"]


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_text(metric, value):
    if metric == "rel_l2":
        return f"relative L2 error of {value:.3%}"
    return f"{metric} of {value:.5g}"


def method_label(label):
    # Keep hyperparameters in the cited arm; these labels are for a reader.
    family = label.split("[", 1)[0]
    names = {"physics-MLE": "physics-based maximum-likelihood estimator",
             "physics-MMSE": "physics-based posterior mean", "hybrid": "hybrid estimator"}
    return names.get(family, label)


def held_out_target(data, metric, value):
    """Use the run's explicit test objective for both the split and the comparison direction."""
    criterion = data.get("success_criterion")
    objectives = data.get("selection_objectives")
    if not criterion or not objectives:
        return None
    rule = criterion["value"]
    threshold = rule.get("threshold")
    if rule.get("metric") != metric or not numeric(threshold):
        return None

    # Spec.objectives records test/metric/operator/threshold together. Do not
    # infer a direction from the name of the metric or a validation score.
    prefix = f"Success criterion: test {metric} "
    operators = set()
    for objective in objectives["value"]:
        if not isinstance(objective, str) or not objective.startswith(prefix):
            continue
        fields = objective[len(prefix):].split()
        if len(fields) != 2 or fields[0] not in ("<=", ">="):
            return None
        try:
            bound = float(fields[1])
        except ValueError:
            return None
        if bound != threshold:
            return None
        operators.add(fields[0])
    if len(operators) != 1:
        return None
    upper_bound = operators == {"<="}
    passed = value <= threshold if upper_bound else value >= threshold
    sources = [criterion, objectives]
    audit = data.get("audit_outcome")
    if audit and audit["value"] is not None:
        if audit["value"] is not passed:
            return None
        sources.append(audit)

    limit = str(threshold)
    if metric == "rel_l2":
        limit = format((Decimal(str(threshold)) * 100).normalize(), "f") + "%"
    comparison = "at most" if upper_bound else "at least"
    outcome = "meeting" if passed else "missing"
    return f"{outcome} the prespecified target of {comparison} {limit}", sources


def measurements(data, *, for_paper=False):
    entries = {}

    def add(text, *facts, table_field=None, **metadata):
        refs = list(dict.fromkeys(map(reference, facts)))
        identity = json.dumps([data["run_id"]["value"], refs, text]).encode()
        key = "E" + hashlib.sha256(identity).hexdigest()[:12]
        if for_paper and table_field is not None:
            metadata["table_field"] = table_field
        entries[key] = {"text": text, "sources": refs, **metadata}

    # Include implementation details only when the revision step supplies source excerpts.
    implementation = data.get("implementation", {})
    if "generate" in implementation and "sample_parameters" in implementation:
        simulator = data["simulator"]
        params = simulator["value"]["params"]
        count = params["walks_per_pair"]
        if type(count) is int and count > 0:
            add(f"{count:,} walks per ordered pair", simulator, implementation["generate"])
        limits = params["weights"]
        if numeric(limits["low"]) and numeric(limits["high"]):
            add(f'{limits["low"]} to {limits["high"]}', simulator, implementation["sample_parameters"])

    split = data.get("data_configuration")
    if split:
        for key, label in (("n_train", "training"), ("n_val", "validation"), ("n_test", "held-out")):
            n = split["value"].get(key)
            if numeric(n):
                add(f"{n:,} {label} cases", split)

    for study in data["studies"]:
        arms = {}
        for fact in study["validation_arms"]:
            arm = fact["value"]
            arm_id = fact["source"]["json_pointer"].split("/")[-1].replace("~1", "/").replace("~0", "~")
            arms[arm_id] = fact
            if arm.get("status") != "ok":
                continue
            name = method_label(arm.get("label") or arm_id)
            seed = f" (seed {arm['seed']})" if arm.get("seed") is not None else ""
            for metric, value in (arm.get("val") or {}).items():
                if numeric(value):
                    add(f"{name} had validation {metric_text(metric, value)}{seed}", fact,
                        table_field=metric)
            if numeric(arm.get("forward_evals")):
                add(f"{name} used {arm['forward_evals']:,} forward evaluations{seed}", fact,
                    table_field="forward_evals")

        if for_paper:
            # Only combine verdicts from the same study with the same recorded basis.
            groups = {}
            wording = {"inconclusive": "the prespecified comparison remains unresolved",
                       "contradicted": "the result contradicted the prespecified claim",
                       "supported": "the result supported the prespecified claim",
                       "untestable": "the available evidence could not test the claim"}
            for verdict_fact in study["recorded_verdicts"]:
                v = verdict_fact["value"]
                outcome = v.get("outcome")
                if outcome in wording:
                    add(wording[outcome], verdict_fact, reader_verdict=True)
                    if outcome == "inconclusive" and v.get("reason"):
                        groups.setdefault(v["reason"], []).append(verdict_fact)
            for group in groups.values():
                if len(group) > 1:
                    requirements = [set(f["value"].get("unmet_requirements") or []) for f in group]
                    if not any(requirements) or set.intersection(*requirements):
                        add("the required margins remain unconfirmed" if all(
                            numeric((f["value"].get("predicate") or {}).get("margin")) for f in group)
                            else "the prespecified comparisons remain unresolved", *group, reader_verdict=True)

        relative_error_groups = {}
        for fact in study["recorded_verdicts"]:
            verdict = fact["value"]
            pred = verdict.get("predicate") or {}
            outcome = verdict.get("outcome")
            if outcome:
                add(f"the recorded verdict for {verdict['claim_id']} was {outcome}", fact)
            treatment = arms.get(pred.get("treatment"))
            control = arms.get(pred.get("control"))
            if not treatment or not control:
                continue
            a, b = treatment["value"], control["value"]
            if any(x.get("status") != "ok" for x in (a, b)):
                continue
            first = method_label(a.get("label") or pred["treatment"])
            second = method_label(b.get("label") or pred["control"])
            if first == second:
                first, second = pred["treatment"], pred["control"]
            pair = f"{first} versus {second}"
            # Effects in the registered relative comparison are positive for improvement.
            op = pred.get("operator")
            if op not in ("at_least_better", "no_more_than_worse"):
                continue
            margin = pred.get("margin")
            if numeric(margin):
                rule = "at least" if op == "at_least_better" else "at most"
                direction = "improvement" if op == "at_least_better" else "worsening"
                add(f"{pair}: the registered {rule} {margin:.2%} {direction} criterion "
                    f"for validation {pred['metric']} was recorded as {verdict['outcome']}",
                    fact, treatment, control)
            if a.get("seed") is None or a["seed"] != b.get("seed") or a.get("fidelity") != b.get("fidelity"):
                continue
            context = f"{pair}, validation seed {a['seed']}"
            if numeric(verdict.get("effect")):
                add(f"{context}: observed relative improvement in {pred['metric']} "
                    f"was {verdict['effect']:.2%}", fact, treatment, control)
            bounds = verdict.get("interval")
            if isinstance(bounds, list) and len(bounds) == 2 and all(map(numeric, bounds)):
                add(f"{context}: the recorded interval for relative improvement in {pred['metric']} "
                    f"was {bounds[0]:.2%} to {bounds[1]:.2%}", fact, treatment, control)
            # Group comparisons only within one study and against the *same fitted*
            # treatment. A shared method label alone does not identify a fit.
            metric = pred.get("metric")
            if metric == "rel_l2" and numeric(verdict.get("effect")) and first != second:
                names = {"extra_trees/raw+logy": "Extra Trees", "torch_mlp/raw+logy": "the neural regressor",
                         "ridge/raw+logy": "ridge regression"}
                baseline = names.get(second, second)
                key = (pred["treatment"], a["seed"])
                relative_error_groups.setdefault(key, []).append((baseline, verdict["effect"], fact, control))
                if for_paper and isinstance(bounds, list) and len(bounds) == 2 and all(map(numeric, bounds)):
                    add(f"on validation seed {a['seed']}, the paired interval for error reduction with "
                        f"{first} relative to {baseline} was {bounds[0]:.2%} to {bounds[1]:.2%}",
                        fact, treatment, control, compact_comparison=True)
            x, y = a.get("forward_evals"), b.get("forward_evals")
            if numeric(x) and numeric(y) and y > 0 and x != y:
                direction = "fewer" if x < y else "more"
                if for_paper:
                    add(f"the {first} used {abs(x - y) / y:.2%} {direction} forward evaluations "
                        f"than the {second} on validation seed {a['seed']}", treatment, control)
                add(f"the {first} used {abs(x - y) / y:.2%} {direction} forward evaluations "
                    f"than the {second} on validation seed {a['seed']} ({x:,} versus {y:,})",
                    treatment, control, table_field="forward_evals")

        for (treatment_id, seed), comparisons in relative_error_groups.items():
            if len(comparisons) < 2 or len({row[0] for row in comparisons}) != len(comparisons):
                continue
            treatment = arms[treatment_id]
            name = method_label(treatment["value"].get("label") or treatment_id)
            clauses, facts = [], [treatment]
            for baseline, effect, verdict_fact, control in comparisons:
                direction = "lower" if effect > 0 else "higher"
                clauses.append(f"{abs(effect):.2%} {direction} than {baseline}" if effect
                               else f"the same as {baseline}")
                facts.extend([verdict_fact, control])
            contrasts = ", ".join(clauses[:-1]) + " and " + clauses[-1]
            add(f"on validation seed {seed}, relative L2 error for the {name} was {contrasts}",
                *facts, compact_comparison=True)

    outcome_counts = (data.get("study_summary") or {}).get("claim_outcomes")
    if for_paper and outcome_counts and isinstance(outcome_counts["value"], dict):
        for outcome in ("supported", "contradicted", "inconclusive", "untestable"):
            count = outcome_counts["value"].get(outcome)
            if type(count) is not int or count < 0:
                continue
            if count == 0:
                # Zero across all claims also means zero among the comparisons.
                for subject in ("claim", "comparison"):
                    add(f"No registered {subject} was recorded as {outcome}", outcome_counts,
                        summary_sentence=True)
            else:
                noun, verb = ("claim", "was") if count == 1 else ("claims", "were")
                add(f"{count} registered {noun} {verb} recorded as {outcome}", outcome_counts,
                    summary_sentence=True)

    final = data.get("final_model") or {}
    labels, test = final.get("methods"), final.get("held_out_metrics")
    if test and labels:
        name = ", ".join(map(method_label, labels["value"])) or "selected estimator"
        for metric, value in (test["value"] or {}).items():
            if numeric(value):
                result = f"{name} had held-out {metric_text(metric, value)}"
                add(result, labels, test)
                target = held_out_target(data, metric, value)
                if target:
                    conclusion, sources = target
                    add(f"{result}, {conclusion}", labels, test, *sources)
    return entries
