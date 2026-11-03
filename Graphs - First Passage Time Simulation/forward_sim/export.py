"""Portable exports: every finite bin, plus killed and censored outcomes."""
import csv
import io
import json
import numpy as np
from .engine import wilson


def csv_text(result: dict) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["run_id", "graph", "analysis_type", "case_label", "B", "B_size",
                     "T", "N", "seed", "dt", "q", "source", "target",
                     "outcome", "step", "physical_time", "count", "r_exact", "r_empirical",
                     "ci95_lower", "ci95_upper"])
    cfg = result["config"]
    base = [result.get("id", ""), cfg["graph"], cfg.get("analysis_type", "baseline"),
            cfg.get("case_label", "Baseline"), json.dumps(result["observed"]),
            len(result["observed"]), cfg["T"], cfg["N"], cfg["seed"], cfg["dt"], json.dumps(cfg["q"])]
    for i, source in enumerate(result["observed"]):
        for j, target in enumerate(result["observed"]):
            for t in range(cfg["T"]):
                writer.writerow(base + [source, target, "hit", t+1, result["physical_times"][t],
                                        result["counts"][t][i][j], result["r_exact"][t][i][j],
                                        result["r_empirical"][t][i][j], result["ci95_lower"][t][i][j],
                                        result["ci95_upper"][t][i][j]])
            pair = next(p for p in result["pairs"] if p["source"] == source and p["target"] == target)
            for outcome in ("killed", "censored"):
                count = pair[outcome+"_count"]
                low, high = wilson(np.asarray(count), cfg["N"])
                writer.writerow(base + [source, target, outcome, "", "", count,
                                        pair[outcome+"_exact"], count/cfg["N"], float(low), float(high)])
    return stream.getvalue()
