#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pocketbind.dataset import EncodedPocketBindDataset, PocketBindFrameBuilder
from pocketbind.encoding import Vocab
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
        out[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    out["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a PocketBind-MHCI checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["ba", "el", "iedb", "cedar"])
    parser.add_argument(
        "--pseudoseqs",
        default=Path("Trainning dataset/NetMHCpan_train/pseudoseqs"),
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
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
    model = PocketBindModel(
        vocab_size=int(checkpoint.get("vocab_size", vocab.size)),
        hidden_dim=args.hidden_dim,
        num_heads=4,
        num_layers=args.num_layers,
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out} rows={len(out)}")


if __name__ == "__main__":
    main()
