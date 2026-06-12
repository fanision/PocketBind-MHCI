# PocketBind-MHCI v1 Implementation Notes

Generated on 2026-06-12.

## Implemented Components

- `src/pocketbind/encoding.py`: amino-acid vocabulary, fixed-length peptide/HLA/context encoding.
- `src/pocketbind/dataset.py`: HLA pseudosequence join, allele normalization, torch-compatible encoded dataset.
- `src/pocketbind/model.py`: first PocketBind-MHCI neural architecture.
- `scripts/train_pocketbind.py`: single-task and multi-task training entrypoint.
- `scripts/predict_pocketbind.py`: checkpoint prediction entrypoint.

## Model v1

The current model is a compact implementation of the planned architecture:

- Peptide Transformer encoder
- HLA pseudosequence Transformer encoder
- Context Transformer encoder
- Bidirectional peptide-HLA cross-attention
- Task embedding
- Separate heads for BA, EL, IEDB/pathogen, and CEDAR/neoepitope

This is intentionally a v1 training scaffold. The next modeling improvements should add:

- Pocket/contact bias for anchor residues
- Multi-task batch sampling
- Percentile-rank calibration
- Pairwise ranking loss for BA
- EL pretraining followed by BA fine-tuning

## Smoke Tests

PyTorch installed locally:

```text
torch 2.12.0
```

BA smoke train:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  --train "Trainning dataset/NetMHCpan_train/c000_ba" \
  --task ba \
  --limit 128 \
  --epochs 1 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 1 \
  --out artifacts/pocketbind/smoke_ba.pt
```

Result:

```text
epoch=1 train_loss=0.031137 val_loss=0.015462 rows=128 device=cpu
```

CEDAR smoke train:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  --train "Trainning dataset/NetMHCpan_train/c000_cedar" \
  --task cedar \
  --limit 128 \
  --epochs 1 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 1 \
  --out artifacts/pocketbind/smoke_cedar.pt
```

Result:

```text
epoch=1 train_loss=0.711157 val_loss=0.682929 rows=128 device=cpu
```

Four-task smoke train:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  --train-spec "ba=Trainning dataset/NetMHCpan_train/c000_ba" \
  --train-spec "el=Trainning dataset/NetMHCpan_train/c000_el" \
  --train-spec "iedb=Trainning dataset/NetMHCpan_train/c000_iedb" \
  --train-spec "cedar=Trainning dataset/NetMHCpan_train/c000_cedar" \
  --limit 128 \
  --epochs 1 \
  --batch-size 32 \
  --hidden-dim 64 \
  --num-layers 1 \
  --out artifacts/pocketbind/smoke_multitask_el.pt
```

Result:

```text
loaded task=ba rows=106
loaded task=el rows=128
loaded task=iedb rows=107
loaded task=cedar rows=128
epoch=1 train_loss=0.926232 val_loss=0.839380 rows=469 device=cpu
```

EL pretrain config smoke test:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  @configs/el_pretrain_c000_smoke.args
```

Result:

```text
loaded task=el rows=1643
epoch=1 train_loss=0.419577 val_loss=0.167470 rows=1643 device=cpu
```

Four-task config smoke test:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  @configs/multitask_c000_smoke.args
```

Result:

```text
loaded task=ba rows=396
loaded task=el rows=512
loaded task=iedb rows=414
loaded task=cedar rows=512
epoch=1 train_loss=0.819751 val_loss=0.659208 rows=1834 device=cpu
```

BA fine-tune from EL checkpoint smoke test:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  @configs/ba_finetune_from_el_c000_smoke.args
```

Result:

```text
loaded task=ba rows=214
loaded init checkpoint=artifacts/pocketbind/el_pretrain_c000_smoke.pt
epoch=1 train_loss=0.057812 val_loss=0.025515 rows=214 device=cpu
```

Sampled EL pretraining test:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  @configs/el_pretrain_c000_sampled.args
```

Result:

```text
loaded task=el rows=2767
epoch=1 val_el_roc_auc=0.685288 val_el_auprc=0.080440
epoch=2 val_el_roc_auc=0.701082 val_el_auprc=0.091568
```

Sampled BA fine-tune from EL test:

```bash
PYTHONPATH=src python3 scripts/train_pocketbind.py \
  @configs/ba_finetune_from_el_c000_sampled.args
```

Result:

```text
loaded task=ba rows=4031
loaded init checkpoint=artifacts/pocketbind/el_pretrain_c000_sampled.pt
epoch=1 val_ba_roc_auc=0.638653 val_ba_pearson=0.282183
epoch=2 val_ba_roc_auc=0.642276 val_ba_pearson=0.292168
```

These sampled results verify the training route, not final performance. The model remains far below the NetMHCpan-4.2 target line at this stage.

Prediction smoke test:

```bash
PYTHONPATH=src python3 scripts/predict_pocketbind.py \
  --checkpoint artifacts/pocketbind/smoke_multitask_el.pt \
  --input "Trainning dataset/NetMHCpan_train/c000_cedar" \
  --task cedar \
  --limit 32 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 1 \
  --out artifacts/pocketbind/smoke_cedar_predictions.tsv
```

## Valid Rows After HLA/Pseudosequence Filtering

For fold `c000`:

| Dataset | Task | Valid rows | Label distribution |
|---|---:|---:|---|
| `c000_el` | EL | 963,175 | `{0: 916545, 1: 46630}` |
| `c000_iedb` | IEDB | 6,757 | `{0: 5269, 1: 1488}` |
| `c000_cedar` | CEDAR | 997 | `{0: 759, 1: 238}` |

## Current Caveats

- Training ran on CPU in this environment. MPS/GPU support should be checked on the target training machine.
- `EncodedPocketBindDataset` currently returns Python lists and uses a simple collate function; performance should be improved before full EL pretraining.
- Multi-task training currently concatenates per-task frames, then uses shuffled mini-batches and masked per-task losses.
- Dataset rows are now pre-encoded and cached; if PyTorch is available, numeric fields are cached as tensors.
- `--sample-rows` should be preferred for smoke/medium experiments because many local files are label-sorted near the top.
- Full EL training may still need streaming or sharded caching on memory-constrained machines.
