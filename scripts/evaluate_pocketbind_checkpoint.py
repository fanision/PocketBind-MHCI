#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pocketbind.dataset import EncodedPocketBindDataset, PocketBindFrameBuilder
from pocketbind.encoding import Vocab
from pocketbind.metrics import evaluate
from pocketbind.model import PocketBindModel

CONTEXT_TASKS = {"el", "iedb", "cedar"}


def collate(batch: list[dict]):
    import torch

    tensor_keys = [
        "peptide_ids",
        "peptide_mask",
        "hla_ids",
        "hla_mask",
        "context_ids",
        "context_mask",
        "task_id",
    ]
    out = {}
    for key in tensor_keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values).long()
        else:
            out[key] = torch.tensor(values, dtype=torch.long)
    labels = [item["label"] for item in batch]
    if torch.is_tensor(labels[0]):
        out["label"] = torch.stack(labels).float()
    else:
        out["label"] = torch.tensor(labels, dtype=torch.float32)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PocketBind checkpoint on one dataset.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["ba", "el", "iedb", "cedar"])
    parser.add_argument(
        "--pseudoseqs",
        default=Path("Trainning dataset/NetMHCpan_train/pseudoseqs"),
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--netmhcpan-suite",
        default=Path("artifacts/netmhcpan42_suite"),
        type=Path,
        help="Directory containing NetMHCpan suite outputs.",
    )
    return parser.parse_args()


def predict(args: argparse.Namespace) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    vocab = Vocab.amino_acid()
    builder = PocketBindFrameBuilder(args.pseudoseqs)
    frame = builder.load(
        args.input,
        task=args.task,
        has_context=args.task in CONTEXT_TASKS,
        nrows=args.limit,
    )
    dataset = EncodedPocketBindDataset(frame, vocab=vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = checkpoint.get("model_config", {})
    model = PocketBindModel(
        vocab_size=int(checkpoint.get("vocab_size", vocab.size)),
        hidden_dim=int(config.get("hidden_dim", args.hidden_dim)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", args.num_layers)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scores = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("label")
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(**batch)[args.task].detach().cpu()
            if args.task != "ba":
                pred = torch.sigmoid(pred)
            scores.extend(pred.tolist())

    out = frame.loc[:, ["peptide", "allele", "context", "label", "task"]].copy()
    out["pocketbind_score"] = scores
    return out


def load_netmhcpan_metrics(args: argparse.Namespace) -> pd.DataFrame | None:
    run_name = f"{args.input.name}.{args.task}"
    path = args.netmhcpan_suite / run_name / "metrics.tsv"
    if not path.exists():
        return None
    metrics = pd.read_csv(path, sep="\t")
    metrics.insert(0, "model", "NetMHCpan-4.2")
    metrics.insert(1, "dataset", args.input.name)
    metrics.insert(2, "task", args.task)
    return metrics


def make_comparison(pocketbind_metrics: pd.DataFrame, netmhcpan_metrics: pd.DataFrame | None) -> pd.DataFrame:
    rows = [pocketbind_metrics]
    if netmhcpan_metrics is not None:
        rows.append(netmhcpan_metrics)
    comparison = pd.concat(rows, ignore_index=True, sort=False)
    if netmhcpan_metrics is not None:
        metric_cols = [
            col
            for col in ["roc_auc", "auc0.1", "auprc", "ppv_at_n", "pearson", "spearman", "mse"]
            if col in comparison.columns
        ]
        baseline = comparison.loc[comparison["model"].eq("NetMHCpan-4.2")].iloc[0]
        comparison["same_n_as_netmhcpan"] = comparison["n"].eq(baseline["n"])
        for col in metric_cols:
            comparison[f"delta_vs_netmhcpan.{col}"] = comparison[col] - baseline[col]
    return comparison


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    predictions = predict(args)
    predictions_path = args.out_dir / "predictions.tsv"
    predictions.to_csv(predictions_path, sep="\t", index=False)

    metrics = evaluate(predictions, label_col="label", score_col="pocketbind_score", task=args.task)
    pocketbind_metrics = pd.DataFrame(
        [
            {
                "model": "PocketBind-MHCI",
                "dataset": args.input.name,
                "task": args.task,
                **metrics,
            }
        ]
    )
    metrics_path = args.out_dir / "metrics.tsv"
    pocketbind_metrics.to_csv(metrics_path, sep="\t", index=False)

    netmhcpan_metrics = load_netmhcpan_metrics(args)
    comparison = make_comparison(pocketbind_metrics, netmhcpan_metrics)
    comparison_path = args.out_dir / "comparison.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False)

    print(comparison.to_string(index=False))
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
