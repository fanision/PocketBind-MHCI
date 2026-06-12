#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pocketbind.dataset import EncodedPocketBindDataset, PocketBindFrameBuilder
from pocketbind.encoding import Vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PocketBind-MHCI v1.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["ba", "el", "iedb", "cedar"])
    parser.add_argument(
        "--pseudoseqs",
        default=Path("Trainning dataset/NetMHCpan_train/pseudoseqs"),
        type=Path,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pocketbind/checkpoint.pt"))
    return parser.parse_args()


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
        dtype = torch.long
        out[key] = torch.tensor([item[key] for item in batch], dtype=dtype)
    out["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return out


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc

    from pocketbind.model import PocketBindModel

    args = parse_args()
    has_context = args.task != "ba"
    builder = PocketBindFrameBuilder(args.pseudoseqs)
    frame = builder.load(args.train, task=args.task, has_context=has_context)
    if args.limit is not None:
        frame = frame.head(args.limit).copy()
    if frame.empty:
        raise SystemExit("No valid rows after allele/pseudosequence filtering.")

    vocab = Vocab.amino_acid()
    dataset = EncodedPocketBindDataset(frame, vocab=vocab)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(13),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PocketBindModel(
        vocab_size=vocab.size,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss() if args.task == "ba" else torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            labels = batch.pop("label").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(**batch)[args.task]
            loss = loss_fn(pred, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("label").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                pred = model(**batch)[args.task]
                val_losses.append(float(loss_fn(pred, labels).detach().cpu()))
        print(
            f"epoch={epoch + 1} train_loss={np.mean(losses):.6f} "
            f"val_loss={np.mean(val_losses):.6f} rows={len(dataset)} device={device}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": vocab.size, "task": args.task}, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

