#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from pocketbind.dataset import EncodedPocketBindDataset, PocketBindFrameBuilder
from pocketbind.encoding import Vocab

TASKS = ["ba", "el", "iedb", "cedar"]
TASK_TO_ID = {task: idx for idx, task in enumerate(TASKS)}
CONTEXT_TASKS = {"el", "iedb", "cedar"}


class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        return [line]


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        description="Train PocketBind-MHCI v1.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--train", type=Path, help="Single-task training file.")
    parser.add_argument("--task", choices=TASKS, help="Task for --train.")
    parser.add_argument(
        "--train-spec",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Multi-task input. Repeat, e.g. --train-spec ba=... --train-spec el=...",
    )
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
    parser.add_argument(
        "--task-weights",
        default="ba=1.0,el=0.3,iedb=0.5,cedar=0.5",
        help="Comma-separated loss weights.",
    )
    parser.add_argument(
        "--auto-pos-weight",
        action="store_true",
        help="Use negative/positive label ratios as BCE pos_weight for classification tasks.",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pocketbind/checkpoint.pt"))
    return parser.parse_args()


def parse_task_weights(value: str) -> dict[str, float]:
    weights = {task: 1.0 for task in TASKS}
    if not value:
        return weights
    for item in value.split(","):
        task, weight = item.split("=", 1)
        task = task.strip()
        if task not in TASK_TO_ID:
            raise SystemExit(f"Unknown task in --task-weights: {task}")
        weights[task] = float(weight)
    return weights


def parse_train_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs = []
    if args.train or args.task:
        if not args.train or not args.task:
            raise SystemExit("--train and --task must be provided together.")
        specs.append((args.task, args.train))
    for spec in args.train_spec:
        if "=" not in spec:
            raise SystemExit(f"Invalid --train-spec {spec!r}; expected TASK=PATH")
        task, path = spec.split("=", 1)
        task = task.strip()
        if task not in TASK_TO_ID:
            raise SystemExit(f"Unknown task in --train-spec: {task}")
        specs.append((task, Path(path)))
    if not specs:
        raise SystemExit("Provide --train/--task or at least one --train-spec TASK=PATH.")
    return specs


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


def compute_multitask_loss(outputs, labels, task_ids, *, task_weights, loss_fns):
    import torch

    losses = []
    parts = {}
    for task, task_id in TASK_TO_ID.items():
        mask = task_ids == task_id
        if not torch.any(mask):
            continue
        task_loss = loss_fns[task](outputs[task][mask], labels[mask])
        weighted = task_weights[task] * task_loss
        losses.append(weighted)
        parts[task] = float(task_loss.detach().cpu())
    if not losses:
        raise RuntimeError("Batch did not contain any known task ids.")
    return torch.stack(losses).sum(), parts


def build_loss_fns(frame: pd.DataFrame, *, auto_pos_weight: bool, device):
    import torch

    losses = {"ba": torch.nn.HuberLoss()}
    for task in ["el", "iedb", "cedar"]:
        pos_weight = None
        if auto_pos_weight:
            labels = frame.loc[frame["task"] == task, "label"]
            positives = int((labels > 0).sum())
            negatives = int((labels <= 0).sum())
            if positives > 0 and negatives > 0:
                pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        losses[task] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return losses


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc

    from pocketbind.model import PocketBindModel

    args = parse_args()
    specs = parse_train_specs(args)
    task_weights = parse_task_weights(args.task_weights)
    builder = PocketBindFrameBuilder(args.pseudoseqs)
    frames = []
    for task, path in specs:
        frame_part = builder.load(
            path,
            task=task,
            has_context=task in CONTEXT_TASKS,
            nrows=args.limit,
        )
        frames.append(frame_part)
        print(f"loaded task={task} rows={len(frame_part)} path={path}")
    frame = pd.concat(frames, ignore_index=True)
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
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"loaded init checkpoint={args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fns = build_loss_fns(frame, auto_pos_weight=args.auto_pos_weight, device=device)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        task_losses: dict[str, list[float]] = {task: [] for task in TASKS}
        for batch in train_loader:
            labels = batch.pop("label").to(device)
            task_ids = batch["task_id"].to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, parts = compute_multitask_loss(
                outputs,
                labels,
                task_ids,
                task_weights=task_weights,
                loss_fns=loss_fns,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for task, value in parts.items():
                task_losses[task].append(value)

        model.eval()
        val_losses = []
        val_task_losses: dict[str, list[float]] = {task: [] for task in TASKS}
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("label").to(device)
                task_ids = batch["task_id"].to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss, parts = compute_multitask_loss(
                    outputs,
                    labels,
                    task_ids,
                    task_weights=task_weights,
                    loss_fns=loss_fns,
                )
                val_losses.append(float(loss.detach().cpu()))
                for task, value in parts.items():
                    val_task_losses[task].append(value)
        train_parts = " ".join(
            f"{task}_loss={np.mean(values):.6f}" for task, values in task_losses.items() if values
        )
        val_parts = " ".join(
            f"val_{task}_loss={np.mean(values):.6f}"
            for task, values in val_task_losses.items()
            if values
        )
        print(
            f"epoch={epoch + 1} train_loss={np.mean(losses):.6f} "
            f"val_loss={np.mean(val_losses):.6f} rows={len(dataset)} device={device} "
            f"{train_parts} {val_parts}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": vocab.size,
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "num_heads": 4,
                "num_layers": args.num_layers,
            },
            "tasks": [task for task, _ in specs],
            "task_weights": task_weights,
            "auto_pos_weight": args.auto_pos_weight,
        },
        args.out,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
