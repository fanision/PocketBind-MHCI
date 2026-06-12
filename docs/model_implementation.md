# PocketBind-MHCI v1 Implementation Notes

Generated on 2026-06-12.

## Implemented Components

- `src/pocketbind/encoding.py`: amino-acid vocabulary, fixed-length peptide/HLA/context encoding.
- `src/pocketbind/dataset.py`: HLA pseudosequence join, allele normalization, torch-compatible encoded dataset.
- `src/pocketbind/model.py`: first PocketBind-MHCI neural architecture.
- `scripts/train_pocketbind.py`: first training entrypoint.

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
- The first training script supports one task at a time; multi-task sampling is the next step.

