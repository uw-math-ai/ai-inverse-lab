# Autodata (LLM challenger): inverse-problem data generator

This is the same harness as [`autodata-inverse`](../autodata-inverse/README.md), but a **language model** proposes
the simulation settings, as in Meta FAIR's Autodata ([arXiv:2606.25996](https://arxiv.org/abs/2606.25996)).
Each round the model gets:

- the inverse problem (recover μ on X∖B from first-passage times on B, Gao, Li & Yang, arXiv:2603.20710);
- the spec schema, the acceptance gates and the harness limits;
- the strategy guidance;
- the last eight rounds with their numbers, and the categorized feedback: too easy, strong failed, failed
  verifier.

It replies with one JSON spec. The harness then verifies the instance, runs the weak and strong solvers and
accepts or refines. Two cases are handled specially:

- **Invalid spec:** it is recorded and returned to the model as verifier feedback the next round.
- **Model call fails, or the reply has no JSON spec:** the rule-based challenger proposes that round
  instead. The log marks such rounds `[llm failed ...; heuristic]`.

Seeds, solvers, gates and limits are set by the harness and cannot be changed by the model.

## Inputs

These are the same as [`autodata-inverse`](../autodata-inverse/README.md#inputs-all-optional), except that
`parallel` is not available and there is one extra input:

| field | default | meaning |
|---|---|---|
| `llm_meta` | `false` | mode `meta`: the model also writes the root-cause analysis and rewrites the strategy JSON, whose values are clipped to safe ranges. Off: rule-based mutation |

Sample input:

```json
{"scenario_ids": ["paper9_grid", "tree_leaves"], "max_rounds": 6}
```

## Outputs

These are the same files as `autodata-inverse`: `report.md`, `result.json`, `dataset.jsonl` and `rounds.jsonl`,
plus `meta_history.jsonl` and `best_strategy.json` in meta mode. Each round's `challenger_note` in
`rounds.jsonl` holds the model's rationale. The log ends with the number of model calls and the tokens
reported.

## Platform settings and cost

- Connection: **required**, on the `provider` network. The providers are `claude` (Claude Code CLI),
  `codex` (Codex CLI), `openrouter` and `openai` (Responses API), called as in section 5 of
  `agents/instructions.md`. The CLIs run with no tools. The model comes from the connection
  (`AGENT_MODEL`), otherwise the provider's default.
- Tier: `sol`.
- **Charges: yes.** This agent incurs model charges or uses subscription limits. It makes about one model
  call per round, at most `max_rounds` × number of scenarios. A default run is up to 144 calls; meta mode
  with `llm_meta` uses more. Token usage is reported to the platform through `platform_usage`.
- Handoff: when the plan reaches its threshold, the run stops before the next round and writes the dataset so
  far. `HANDOFF.md` lists the finished scenarios; pass them back as `skip_scenarios` to continue.
- Datasets: none. Swarm: none. Recommended server: an AMATH worker. Resources: 1 CPU, 512 MiB, 4 h
  timeout.
- Dependencies: Python 3.12 standard library, the image's `claude` and `codex` CLIs, and
  `agents/shared/platform_usage.py`.

## Local test

Without a provider the agent exits with code 2 and makes no call. A local test with no charges can point
`AGENT_PROVIDER=openrouter` and `OPENROUTER_BASE_URL` at a local fake `/chat/completions` server. Make paid
test runs only when they are needed.
