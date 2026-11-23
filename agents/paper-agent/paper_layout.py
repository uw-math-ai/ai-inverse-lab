"""Render evidence-backed comparison tables and review the reader-facing Results."""
import re

from outline import TOKEN, expand_measurements
from outline_evidence import method_label, numeric, reference


STYLE = """In Results let prose answer the scientific question. If exact arm values are useful,
attach an optional "table" to a paragraph: its value is a list of at least two exact arm
source references, also included in the paragraph's sources. Choose successful arms from
the same study, validation seed and fidelity. For a V1 run with baseline_validation and
a final validation_selection_score, those two exact sources may form a validation comparison. The renderer supplies names, numeric cells,
units and caption. Do not type a Markdown table. When using a table, use the shorter
relative comparison token in prose; leave the underlying counts and errors in the table.
For a meaningful verdict, name the comparison in ordinary language and use a reader_verdict
token plus a short explanation of what evidence is missing or why the decision rule did
not settle it. Combine comparisons only if the same limitation applies. Cite the linked
objection or diagnostic for that reason when available. Check later observations before
describing an earlier reviewer concern. Keep internal study/claim IDs in sources, and omit
routine run-status summaries. If a table or verdict adds nothing to a finding, omit it."""


class PaperLayout:
    def __init__(self, evidence):
        self.metric = evidence["primary_metric"]["value"]
        self.arms = {}
        self.reasons = {}
        self.ids = set()
        self.cells = {}
        for study in evidence["studies"]:
            self.ids.add(study["id"])
            for arm in study["validation_arms"]:
                self.arms[reference(arm)] = (study["id"], arm["value"])
            reasons = {o["value"]["id"]: reference(o) for o in study.get("review_objections", [])}
            for diagnostic in study.get("diagnostic_checks", []):
                ref = reference(diagnostic)
                reasons["diagnostic:" + ref.rsplit("/", 1)[-1]] = ref
            for verdict in study["recorded_verdicts"]:
                v = verdict["value"]
                self.ids.add(v["claim_id"])
                self.reasons[reference(verdict)] = {reasons[k] for k in v.get("unmet_requirements") or []
                                                     if k in reasons}

        # A V1 run stores its baseline and final selection score as run-level
        # validation facts. The final test measurements belong to a different split.
        baseline = evidence.get("baseline_validation")
        selected = evidence.get("final_model", {}).get("validation_selection_score")
        if not self.arms and baseline and selected:
            for fact, label, scores in (
                (baseline, "Baseline", baseline["value"]),
                (selected, "Selected predictor (selection estimate)", {self.metric: selected["value"]}),
            ):
                self.arms[reference(fact)] = ("run validation", {"label": label, "val": scores, "status": "ok"})

    def table(self, rows, sources):
        if (not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(r, str) for r in rows)
                or len(set(rows)) != len(rows) or any(r not in sources or r not in self.arms for r in rows)):
            raise ValueError("A comparison table needs distinct validation arms cited in this paragraph.")
        arms = [self.arms[r][1] for r in rows]
        settings = {(self.arms[r][0], a.get("seed"), a.get("fidelity")) for r, a in zip(rows, arms)}
        if len(settings) != 1 or any(a.get("status") != "ok" or not numeric((a.get("val") or {}).get(self.metric))
                                     for a in arms):
            raise ValueError("Table rows must share a study, validation seed, fidelity and measured metric.")
        names = [method_label(a.get("label") or r.rsplit("/", 1)[-1]) for r, a in zip(rows, arms)]
        if len(set(names)) != len(names):
            names = [a.get("label") or name for a, name in zip(arms, names)]
        if len(set(names)) != len(names):
            raise ValueError("These arms have indistinguishable method labels; describe the comparison in prose.")
        percent = self.metric == "rel_l2"
        header = "Relative L2 error (%)" if percent else self.metric
        has_cost = any(numeric(a.get("forward_evals")) for a in arms)
        headers = ["Estimator", header] + (["Forward evaluations"] if has_cost else [])
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for ref, arm, name in zip(rows, arms, names):
            value = arm["val"][self.metric]
            score = f"{value * 100:.3f}" if percent else f"{value:.5g}"
            cells = [name.replace("|", r"\|").replace("\n", " "), score]
            fields = {self.metric}
            if has_cost:
                cost = arm.get("forward_evals")
                cells.append(f"{cost:,}" if numeric(cost) else "—")
                if numeric(cost):
                    fields.add("forward_evals")
            self.cells.setdefault(ref, set()).update(fields)
            lines.append("| " + " | ".join(cells) + " |")
        seed = arms[0].get("seed")
        caption = "Validation set" + (f", seed {seed}" if seed is not None else "")
        return "\n".join(lines + ["", "*" + caption + ".*"])

    def checks(self, plan, catalog):
        clean_ids = clean_tables = no_repeats = linked_reasons = True
        for section in plan["sections"]:
            for point in section["points"]:
                refs = set(point["sources"])
                visible = expand_measurements(point["text"], refs, catalog)
                if any(re.search(r"(?<![\w-])" + re.escape(i) + r"(?![\w-])", visible) for i in self.ids):
                    clean_ids = False
                if re.search(r"^\s*\|?\s*:?-{3,}:?\s*\|", visible, re.M):
                    clean_tables = False
                for token in TOKEN.findall(point["text"]):
                    entry = catalog[token]
                    if any(entry.get("table_field") in self.cells.get(ref, ()) for ref in entry["sources"]):
                        no_repeats = False
                    if entry.get("reader_verdict"):
                        relevant = [self.reasons[ref] for ref in entry["sources"] if self.reasons.get(ref)]
                        if relevant and not (set.intersection(*relevant) & refs):
                            linked_reasons = False
        return {"internal_ids_kept_out_of_prose": clean_ids, "tables_use_numeric_cells": clean_tables,
                "table_measurements_not_repeated_in_prose": no_repeats,
                "reader_verdicts_cite_available_reasons": linked_reasons}
