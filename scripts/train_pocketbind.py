#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pocketbind.dataset import EncodedPocketBindDataset, PocketBindFrameBuilder
from pocketbind.encoding import Vocab
from pocketbind.metrics import evaluate

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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--balanced-task-sampling",
        action="store_true",
        help="Use inverse task-frequency weights for training batches.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-rows",
        action="store_true",
        help="Use random reservoir sampling for --limit instead of taking file head.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--task-weights",
        default="ba=1.0,el=0.3,iedb=0.5,cedar=0.5",
        help="Comma-separated loss weights.",
    )
    parser.add_argument("--ba-ranking-weight", type=float, default=0.2)
    parser.add_argument("--ba-ranking-min-delta", type=float, default=0.05)
    parser.add_argument(
        "--auto-pos-weight",
        action="store_true",
        help="Use negative/positive label ratios as BCE pos_weight for classification tasks.",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pocketbind/checkpoint.pt"))
    parser.add_argument("--log-file", type=Path, default=None)
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


def ba_pairwise_ranking_loss(scores, labels, *, min_delta: float):
    import torch

    if scores.numel() < 2:
        return scores.new_tensor(0.0)
    label_delta = labels[:, None] - labels[None, :]
    score_delta = scores[:, None] - scores[None, :]
    mask = torch.abs(label_delta) >= min_delta
    if not torch.any(mask):
        return scores.new_tensor(0.0)
    direction = torch.sign(label_delta[mask])
    return torch.nn.functional.softplus(-direction * score_delta[mask]).mean()


def compute_multitask_loss(
    outputs,
    labels,
    task_ids,
    *,
    task_weights,
    loss_fns,
    ba_ranking_weight,
    ba_ranking_min_delta,
):
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
        if task == "ba" and ba_ranking_weight > 0:
            rank_loss = ba_pairwise_ranking_loss(
                outputs[task][mask],
                labels[mask],
                min_delta=ba_ranking_min_delta,
            )
            losses.append(ba_ranking_weight * rank_loss)
            parts["ba_rank"] = float(rank_loss.detach().cpu())
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


def split_train_val_indices(
    frame: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    split_frame = frame.reset_index(drop=True).copy()
    split_frame["binary_label"] = np.where(
        split_frame["task"].eq("ba"),
        split_frame["label"] >= 0.426,
        split_frame["label"] > 0,
    )

    for _, group in split_frame.groupby(["task", "binary_label"], sort=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices.tolist())
            continue
        n_val = int(round(len(indices) * val_fraction))
        n_val = min(max(1, n_val), len(indices) - 1)
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not val_indices:
        raise SystemExit("Validation split is empty; increase dataset size or val_fraction.")
    return train_indices, val_indices


def make_task_balanced_sampler(frame: pd.DataFrame, train_indices: list[int]):
    import torch
    from torch.utils.data import WeightedRandomSampler

    tasks = frame.reset_index(drop=True).loc[train_indices, "task"]
    counts = tasks.value_counts().to_dict()
    weights = [1.0 / counts[task] for task in tasks]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
    )


def checkpoint_payload(model, *, vocab_size, args, specs, task_weights):
    return {
        "model": model.state_dict(),
        "vocab_size": vocab_size,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "num_heads": 4,
            "num_layers": args.num_layers,
        },
        "tasks": [task for task, _ in specs],
        "task_weights": task_weights,
        "auto_pos_weight": args.auto_pos_weight,
        "ba_ranking_weight": args.ba_ranking_weight,
        "ba_ranking_min_delta": args.ba_ranking_min_delta,
    }


def evaluate_loader(model, loader, *, device, task_weights, loss_fns):
    import torch

    model.eval()
    losses = []
    task_losses: dict[str, list[float]] = {task: [] for task in TASKS}
    by_task = {task: {"label": [], "score": []} for task in TASKS}
    with torch.no_grad():
        for batch in loader:
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
                ba_ranking_weight=0.0,
                ba_ranking_min_delta=0.0,
            )
            losses.append(float(loss.detach().cpu()))
            for task, value in parts.items():
                task_losses.setdefault(task, []).append(value)
            for task, task_id in TASK_TO_ID.items():
                mask = task_ids == task_id
                if not torch.any(mask):
                    continue
                score = outputs[task][mask].detach().cpu()
                if task != "ba":
                    score = torch.sigmoid(score)
                by_task[task]["score"].extend(score.tolist())
                by_task[task]["label"].extend(labels[mask].detach().cpu().tolist())

    metric_rows = {}
    for task, values in by_task.items():
        if not values["label"]:
            continue
        df = pd.DataFrame({"label": values["label"], "score": values["score"]})
        metric_rows[task] = evaluate(df, label_col="label", score_col="score", task=task)
    return {
        "loss": float(np.mean(losses)),
        "task_losses": {
            task: float(np.mean(values)) for task, values in task_losses.items() if values
        },
        "metrics": metric_rows,
    }


def flatten_log_row(row: dict) -> dict[str, float | int]:
    flat = {
        "epoch": row["epoch"],
        "train_loss": row["train_loss"],
        "val_loss": row["val_loss"],
    }
    for scope in ["train_task_losses", "val_task_losses"]:
        for task, value in row[scope].items():
            flat[f"{scope}.{task}"] = value
    for task, metrics in row["val_metrics"].items():
        for metric, value in metrics.items():
            flat[f"val_metrics.{task}.{metric}"] = value
    return flat


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
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
            sample_rows=args.sample_rows,
            seed=args.seed,
        )
        frames.append(frame_part)
        print(f"loaded task={task} rows={len(frame_part)} path={path}")
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise SystemExit("No valid rows after allele/pseudosequence filtering.")

    vocab = Vocab.amino_acid()
    dataset = EncodedPocketBindDataset(frame, vocab=vocab)
    train_indices, val_indices = split_train_val_indices(
        frame,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    sampler = (
        make_task_balanced_sampler(frame, train_indices)
        if args.balanced_task_sampling
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(
        f"split train_rows={len(train_indices)} val_rows={len(val_indices)} "
        f"balanced_task_sampling={args.balanced_task_sampling}"
    )

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

    log_rows = []
    best_val_loss = float("inf")
    best_path = args.out.with_name(args.out.stem + ".best" + args.out.suffix)

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
                ba_ranking_weight=args.ba_ranking_weight,
                ba_ranking_min_delta=args.ba_ranking_min_delta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for task, value in parts.items():
                task_losses.setdefault(task, []).append(value)

        val_result = evaluate_loader(
            model,
            val_loader,
            device=device,
            task_weights=task_weights,
            loss_fns=loss_fns,
        )
        train_task_losses = {
            task: float(np.mean(values)) for task, values in task_losses.items() if values
        }
        train_parts = " ".join(
            f"{task}_loss={value:.6f}" for task, value in train_task_losses.items()
        )
        val_parts = " ".join(
            f"val_{task}_loss={value:.6f}"
            for task, value in val_result["task_losses"].items()
        )
        metric_parts = []
        for task, metrics in val_result["metrics"].items():
            for metric in ["roc_auc", "auc0.1", "auprc", "pearson", "spearman"]:
                if metric in metrics:
                    metric_parts.append(f"val_{task}_{metric}={metrics[metric]:.6f}")
        train_loss = float(np.mean(losses))
        val_loss = float(val_result["loss"])
        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_task_losses": train_task_losses,
            "val_task_losses": val_result["task_losses"],
            "val_metrics": val_result["metrics"],
        }
        log_rows.append(log_row)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} rows={len(dataset)} device={device} "
            f"{train_parts} {val_parts} {' '.join(metric_parts)}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint_payload(
                    model,
                    vocab_size=vocab.size,
                    args=args,
                    specs=specs,
                    task_weights=task_weights,
                ),
                best_path,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model,
            vocab_size=vocab.size,
            args=args,
            specs=specs,
            task_weights=task_weights,
        ),
        args.out,
    )
    log_file = args.log_file or args.out.with_suffix(".log.tsv")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([flatten_log_row(row) for row in log_rows]).to_csv(log_file, sep="\t", index=False)
    print(f"Wrote {args.out}")
    print(f"Wrote {best_path}")
    print(f"Wrote {log_file}")


if __name__ == "__main__":
    main()
