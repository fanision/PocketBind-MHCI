# PocketBind-MHCI

PocketBind-MHCI is a planned pan-allele, multi-task model for predicting MHC class I peptide binding and presentation.

The immediate development target is to beat **NetMHCpan-4.2** under the same local train/evaluation splits and metric scripts.

## Current Status

This repository currently contains the project notes and development plan from the initial design discussion. Code has not been implemented yet.

Local artifacts are committed through Git LFS:

- `Paper/`: local PDFs and literature notes
- `Tool/`: local NetMHCpan-4.2 package
- `Trainning dataset/`: local NetMHCpan-style train/evaluation files

After cloning on a new machine, install Git LFS and fetch the large files:

```bash
git lfs install
git lfs pull
```

## First Baseline

The first comparison target is **NetMHCpan-4.2c**.

The local workspace has used:

```text
Tool/netMHCpan-4.2c.Darwin_arm64.tar.gz
```

This package matches macOS arm64 and contains the `netMHCpan-4.2/netMHCpan` entry script.

## Planned Model

PocketBind-MHCI will use:

- Peptide sequence encoding for 8-15mer peptides
- HLA 34-aa pseudosequence encoding
- Context/flanking sequence encoding when available
- Length/core/bulge positional encoding
- Dual Transformer encoders for peptide and HLA
- Bidirectional peptide-HLA cross-attention
- Pocket/contact bias for anchor-position interactions
- Multi-task heads for BA, EL, IEDB/CEDAR epitope scoring, and percentile rank

See [`docs/development_plan.md`](docs/development_plan.md) for details.

## Documentation

- [`docs/conversation_summary.md`](docs/conversation_summary.md): project conversation and decisions so far
- [`docs/development_plan.md`](docs/development_plan.md): proposed model, data, and training plan
- [`docs/netmhcpan42_baseline.md`](docs/netmhcpan42_baseline.md): first baseline plan
- [`docs/baseline_results.md`](docs/baseline_results.md): first NetMHCpan-4.2 baseline results

## Baseline Smoke Test

```bash
PYTHONPATH=src python3 scripts/run_netmhcpan_baseline.py \
  --input "Trainning dataset/NetMHCpan_eval/cedar_test" \
  --task cedar \
  --limit 20
```
