# Test research agent

A deterministic smoke test for the Research Agent Workplace website. It waits a
few seconds while reporting progress, then writes a report and a JSON result. It
is useful for checking that catalog discovery, logs, progress and artifact
downloads work end to end.

## Inputs

| Field     | Type    | Required | Default | Notes                                      |
| --------- | ------- | -------- | ------- | ------------------------------------------ |
| `topic`   | string  | yes      | —       | 1–1000 characters                          |
| `seconds` | integer | no       | `3`     | 1–60, the number of seconds of simulated work |
| `fail`    | boolean | no       | `false` | If `true`, the agent exits with code 1 and writes no artifacts |

Sample input:

```json
{"topic": "Inverse problems for the heat equation", "seconds": 3}
```

## Outputs (in `AGENT_OUTPUT_DIR`)

- `report.md`: a short Markdown report naming the topic
- `result.json`: `{"topic": ..., "seconds": ..., "completed": true}`

Logs include JSON progress lines such as `{"progress": 0.5}`.

## Dependencies and cost

- Runtime image: `research-agent-runtime:v1` (Python 3.12 standard library only)
- Network: none
- API charges: none

## Local test

Run these commands from this folder:

```bash
mkdir -p /tmp/test-agent-out
echo '{"topic": "demo", "seconds": 2}' > /tmp/test-agent-in.json
AGENT_INPUT_PATH=/tmp/test-agent-in.json AGENT_OUTPUT_DIR=/tmp/test-agent-out python run.py
```
