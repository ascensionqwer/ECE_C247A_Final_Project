"""Aggregate CER metrics from Hydra run directories.

Example:
python scripts/summarize_runs.py --logs-dir logs --output-csv logs/summary.csv --plot logs/summary.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def find_result_files(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.rglob("results.json"))


def load_overrides(run_dir: Path) -> str:
    overrides_path = run_dir.joinpath(".hydra", "overrides.yaml")
    if not overrides_path.exists():
        return ""
    with open(overrides_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or []
    return "; ".join(overrides)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output-csv", type=Path, default=Path("logs/summary.csv"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []
    for result_path in find_result_files(args.logs_dir):
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        run_dir = result_path.parent
        val = results.get("val_metrics", [{}])[0]
        test = results.get("test_metrics", [{}])[0]

        rows.append(
            {
                "run_dir": str(run_dir),
                "overrides": load_overrides(run_dir),
                "val_CER": float(val.get("val/CER", float("nan"))),
                "test_CER": float(test.get("test/CER", float("nan"))),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_dir", "overrides", "val_CER", "test_CER"])
        writer.writeheader()
        writer.writerows(rows)

    if args.plot and rows:
        sorted_rows = sorted(rows, key=lambda r: float(r["test_CER"]))
        labels = [Path(str(r["run_dir"])).name for r in sorted_rows]
        values = [float(r["test_CER"]) for r in sorted_rows]

        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Test CER")
        plt.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot, dpi=150)

    print(f"Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
