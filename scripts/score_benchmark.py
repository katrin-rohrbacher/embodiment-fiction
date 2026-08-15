#!/usr/bin/env python
"""Score the LLM benchmark runs and write Tables 4 and 5.

Reads the per-(model, condition) CSVs from cfg.BENCHMARK_DIR; makes no API
calls, so the tables can be rebuilt at any time from the stored predictions.
Unparseable predictions count as wrong.

    python scripts/score_benchmark.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embodiment import config as cfg  # noqa: E402

COLUMNS = [("gpt-5.1", "fewshot_context"), ("gpt-5-mini", "fewshot_context"),
           ("llama3.3-70b-local", "fewshot_context"),
           ("qwen3-32b-local", "fewshot_context"), ("roberta", "cv")]


def load(path):
    df = pd.read_csv(path)
    df["pred"] = df["pred"].fillna("__invalid__")
    return df


def main():
    runs = {p.stem: load(p) for p in sorted(cfg.BENCHMARK_DIR.glob("*__*.csv"))
            if not p.name.startswith("_")}
    if not runs:
        raise SystemExit(f"no runs in {cfg.BENCHMARK_DIR}")

    # The no-context runs cover more sentences than the context runs, which
    # need a recoverable surrounding passage; score everything on the overlap.
    common = set.intersection(*(set(zip(d["filename"], d["sentence_id"]))
                                for d in runs.values()))

    def restrict(df):
        return df[[(f, s) in common
                   for f, s in zip(df["filename"], df["sentence_id"])]]

    rows = []
    for name, df in runs.items():
        df = restrict(df)
        model, condition = name.split("__")
        rows.append({
            "model": model, "condition": condition, "n": len(df),
            "accuracy": accuracy_score(df["gold"], df["pred"]),
            "macro_f1": f1_score(df["gold"], df["pred"], labels=cfg.LABELS,
                                 average="macro", zero_division=0),
            "kappa": cohen_kappa_score(df["gold"], df["pred"]),
            "invalid": (df["pred"] == "__invalid__").mean(),
        })
    table4 = pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)

    available = [(m, c) for m, c in COLUMNS
                 if (cfg.BENCHMARK_DIR / f"{m}__{c}.csv").exists()]
    table5 = pd.DataFrame({"label": cfg.LABELS})
    for model, condition in available:
        df = restrict(load(cfg.BENCHMARK_DIR / f"{model}__{condition}.csv"))
        table5[model] = [f1_score(df["gold"], df["pred"], labels=[label],
                                  average="macro", zero_division=0)
                         for label in cfg.LABELS]
    table5 = table5.sort_values("label").reset_index(drop=True)

    cfg.TABLES.mkdir(exist_ok=True)
    table4.to_csv(cfg.TABLES / "table4_accuracy.csv", index=False)
    table5.to_csv(cfg.TABLES / "table5_per_label.csv", index=False)

    print(f"{len(runs)} runs, {len(common)} sentences in common\n")
    print(table4.round(3).to_string(index=False))
    print()
    print(table5.round(2).to_string(index=False))
    print(f"\nwrote tables/table4_accuracy.csv and tables/table5_per_label.csv")


if __name__ == "__main__":
    main()
