#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PocketBind checkpoint on standard sets.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sets",
        default="c000_ba,iedb_test,cedar_test",
        help="Comma-separated set names: c000_ba, iedb_test, cedar_test.",
    )
    return parser.parse_args()


def set_specs(names: list[str]) -> list[tuple[str, str, Path]]:
    lookup = {
        "c000_ba": ("c000_ba", "ba", Path("Trainning dataset/NetMHCpan_train/c000_ba")),
        "iedb_test": ("iedb_test", "iedb", Path("Trainning dataset/NetMHCpan_eval/iedb_test")),
        "cedar_test": ("cedar_test", "cedar", Path("Trainning dataset/NetMHCpan_eval/cedar_test")),
    }
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(f"Unknown evaluation sets: {', '.join(missing)}")
    return [lookup[name] for name in names]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, task, input_path in set_specs([item.strip() for item in args.sets.split(",") if item.strip()]):
        run_dir = args.out_dir / f"{name}.{task}"
        cmd = [
            sys.executable,
            "scripts/evaluate_pocketbind_checkpoint.py",
            "--checkpoint",
            str(args.checkpoint),
            "--input",
            str(input_path),
            "--task",
            task,
            "--batch-size",
            str(args.batch_size),
            "--out-dir",
            str(run_dir),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        comparison = pd.read_csv(run_dir / "comparison.tsv", sep="\t")
        rows.append(comparison)

    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary_path = args.out_dir / "comparison_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

