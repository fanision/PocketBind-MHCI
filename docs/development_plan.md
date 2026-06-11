# PocketBind-MHCI Development Plan

## Summary

PocketBind-MHCI is planned as a pan-allele, multi-task model for MHC class I peptide binding and presentation.

The first success criterion is to outperform NetMHCpan-4.2 on the same local splits and metrics.

## Training Data

Use the local NetMHCpan-style data as the v1 source of truth:

- `Trainning dataset/NetMHCpan_train/c000-c004_ba`: binding affinity supervision
- `Trainning dataset/NetMHCpan_train/c000-c004_el`: eluted ligand / antigen presentation supervision
- `Trainning dataset/NetMHCpan_train/pseudoseqs`: HLA 34-aa pseudosequences
- `Trainning dataset/NetMHCpan_train/c000-c004_iedb`: IEDB epitope transfer-learning data
- `Trainning dataset/NetMHCpan_train/c000-c004_cedar`: CEDAR neoepitope transfer-learning data
- `Trainning dataset/NetMHCpan_eval/iedb_test`: final IEDB external test
- `Trainning dataset/NetMHCpan_eval/cedar_test`: final CEDAR external test

External data can be added later after the NetMHCpan-4.2 baseline is reproducible:

- Latest IEDB MHCI binding data
- MHC Ligand Atlas
- SysteMHC Atlas
- MHCflurry 2.0 public training data
- Additional CEDAR epitope/neoepitope data

## Model Architecture

Inputs:

- Peptide sequence, usually 8-15 amino acids
- HLA allele 34-aa pseudosequence
- Peptide length
- Context/flanking sequence when available
- Task type: BA, EL, IEDB, or CEDAR

Encoding:

- Amino-acid token embeddings
- Peptide positional embeddings
- Length/core/bulge embeddings
- HLA pseudosequence positional embeddings
- Context/flank embeddings with a missing-context mask for `XXXXXXXXXXXX`

Backbone:

- Peptide Transformer encoder
- HLA pseudosequence Transformer encoder
- Optional context encoder
- Bidirectional peptide-HLA cross-attention interaction blocks
- Pocket/contact bias to encourage anchor-position interactions, especially P2 and C-terminal anchor contacts

Heads:

- `BA_head`: normalized binding affinity score
- `EL_head`: eluted ligand / presentation probability
- `Epitope_head`: IEDB/CEDAR transfer-learning score
- `Rank_head`: allele-specific percentile rank
- `Composite_score`: final screening score derived from calibrated task outputs

## Training Plan

Stage 1: EL pretraining

- Data: `c000-c004_el`
- Task: binary presentation classification
- Goal: learn allele-specific motifs, anchor preferences, peptide length effects, and processing/context signal
- Loss: binary cross entropy or focal loss

Stage 2: BA fine-tuning

- Data: `c000-c004_ba`
- Task: normalized affinity regression and binder ranking
- Loss: Huber or MSE plus pairwise ranking loss within allele

Stage 3: BA + EL joint training

- Data: mixed BA and EL batches
- Initial loss weighting:

```text
total_loss = 1.0 * BA_loss + 0.3 * EL_loss + 0.2 * ranking_loss
```

- Goal: preserve presentation signal while improving binding affinity calibration

Stage 4: IEDB/CEDAR transfer learning

- Data: `c000-c004_iedb` and `c000-c004_cedar`
- Strategy: freeze part of the backbone, fine-tune the task head and final cross-attention layers
- Outputs:
  - `PocketBind-MHCI-BA`
  - `PocketBind-MHCI-EL`
  - `PocketBind-MHCI-IEDB`
  - `PocketBind-MHCI-CEDAR`
  - 5-fold ensemble

## Evaluation Plan

Primary comparison target:

- NetMHCpan-4.2

Metrics:

- BA: ROC-AUC, PR-AUC, Spearman correlation, Pearson correlation, MSE, AUC0.1, PPV@top-k
- EL/presentation: ROC-AUC, AUPRC, AUC0.1, PPV@N, Recall@top-k
- Epitope/neoepitope: ROC-AUC, AUPRC, AUC0.1, PPV@N

Split strategy:

- Use the provided five folds for train/validation.
- Keep `iedb_test` and `cedar_test` untouched until final evaluation.
- Report stratified results by allele, locus, peptide length, and task.

Acceptance criteria:

- PocketBind-MHCI must beat NetMHCpan-4.2 on at least one final external task without material degradation on BA/EL.
- A result is only considered valid when both models are evaluated on the same files with the same metric code.

