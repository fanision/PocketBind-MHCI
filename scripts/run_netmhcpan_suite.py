#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NetMHCpan-4.2 baseline suite.")
    parser.add_argument("--out-dir", default=Path("artifacts/netmhcpan42"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-el",
        action="store_true",
        help="Also run EL folds. Many EL rows are multi-allelic cell-line samples and are skipped.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def metric_task(task: str) -> str:
    return {"ba": "ba", "iedb": "iedb", "cedar": "cedar", "el": "el"}[task]


def main() -> None:
    args = parse_args()
    py = sys.executable
    specs: list[tuple[str, str, Path]] = []
    for fold in range(5):
        tag = f"c{fold:03d}"
        specs.append((tag, "ba", Path(f"Trainning dataset/NetMHCpan_train/{tag}_ba")))
        specs.append((tag, "iedb", Path(f"Trainning dataset/NetMHCpan_eval/{tag}_iedb")))
        specs.append((tag, "cedar", Path(f"Trainning dataset/NetMHCpan_eval/{tag}_cedar")))
        if args.include_el:
            specs.append((tag, "el", Path(f"Trainning dataset/NetMHCpan_train/{tag}_el")))

    specs.extend(
        [
            ("external", "iedb", Path("Trainning dataset/NetMHCpan_eval/iedb_test")),
            ("external", "cedar", Path("Trainning dataset/NetMHCpan_eval/cedar_test")),
        ]
    )

    metric_rows = []
    for fold, task, input_path in specs:
        cmd = [
            py,
            "scripts/run_netmhcpan_baseline.py",
            "--input",
            str(input_path),
            "--task",
            task,
            "--out-dir",
            str(args.out_dir),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        run(cmd)

        run_name = f"{input_path.name}.{task}" + (f".n{args.limit}" if args.limit else "")
        run_dir = args.out_dir / run_name
        merged = run_dir / "merged_predictions.tsv"
        metrics_path = run_dir / "metrics.tsv"
        run(
            [
                py,
                "scripts/evaluate_predictions.py",
                "--merged",
                str(merged),
                "--task",
                metric_task(task),
                "--out",
                str(metrics_path),
            ]
        )
        metrics = pd.read_csv(metrics_path, sep="\t")
        summary = pd.read_csv(run_dir / "summary.tsv", sep="\t")
        row = {
            "fold": fold,
            "dataset": input_path.name,
            "task": task,
            **summary.iloc[0].to_dict(),
            **metrics.iloc[0].to_dict(),
        }
        metric_rows.append(row)

    out = pd.DataFrame(metric_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "suite_metrics.tsv", sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"Wrote {args.out_dir / 'suite_metrics.tsv'}")


if __name__ == "__main__":
    main()

