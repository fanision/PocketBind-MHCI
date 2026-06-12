#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pocketbind.data import read_pocketbind_table, write_netmhcpan_input
from pocketbind.netmhcpan import ensure_netmhcpan_install, run_netmhcpan


TASK_DEFAULTS = {
    "ba": {"context": False, "ba": True, "pathogen": False, "neo": False},
    "el": {"context": True, "ba": False, "pathogen": False, "neo": False},
    "iedb": {"context": True, "ba": False, "pathogen": True, "neo": False},
    "cedar": {"context": True, "ba": False, "pathogen": False, "neo": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NetMHCpan-4.2 baseline on a local dataset.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=TASK_DEFAULTS)
    parser.add_argument("--out-dir", default=Path("artifacts/netmhcpan42"), type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--netmhcpan-dir", default=Path("Tool/netMHCpan-4.2"), type=Path)
    parser.add_argument(
        "--netmhcpan-archive",
        default=Path("Tool/netMHCpan-4.2c.Darwin_arm64.tar.gz"),
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = TASK_DEFAULTS[args.task]
    dataset_name = args.input.name
    run_name = f"{dataset_name}.{args.task}" + (f".n{args.limit}" if args.limit else "")
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ensure_netmhcpan_install(args.netmhcpan_dir, archive_path=args.netmhcpan_archive)
    df = read_pocketbind_table(args.input, has_context=defaults["context"])
    total_rows = len(df)
    skipped = df.loc[~df["has_valid_allele"]].copy()
    df = df.loc[df["has_valid_allele"]].copy()
    valid_rows_before_limit = int(df.shape[0])
    if args.limit is not None:
        df = df.head(args.limit).copy()

    prepared_path = run_dir / "netmhcpan_input.tsv"
    write_netmhcpan_input(df, prepared_path, include_context=defaults["context"])

    predictions_path = run_dir / "netmhcpan_predictions.tsv"
    pred = run_netmhcpan(
        netmhcpan_script=args.netmhcpan_dir / "netMHCpan",
        input_path=prepared_path,
        output_path=predictions_path,
        include_context=defaults["context"],
        include_ba=defaults["ba"],
        pathogen=defaults["pathogen"],
        neo=defaults["neo"],
    )

    labels = df.reset_index(drop=True)
    merged = pd.concat([labels, pred.reset_index(drop=True).add_prefix("netmhcpan_")], axis=1)
    merged.to_csv(run_dir / "merged_predictions.tsv", sep="\t", index=False)
    skipped.to_csv(run_dir / "skipped_invalid_alleles.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {
                "input": str(args.input),
                "task": args.task,
                "total_rows": total_rows,
                "valid_rows_before_limit": valid_rows_before_limit,
                "skipped_invalid_alleles": int(skipped.shape[0]),
                "predicted_rows": int(pred.shape[0]),
                "limit": args.limit,
            }
        ]
    )
    summary.to_csv(run_dir / "summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
