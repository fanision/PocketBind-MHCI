# NetMHCpan-4.2 Baseline Plan

## Summary

The first baseline for PocketBind-MHCI is **NetMHCpan-4.2c**.

The local package currently expected by the project is:

```text
Tool/netMHCpan-4.2c.Darwin_arm64.tar.gz
```

This is a macOS arm64 package and should run on the current Apple Silicon machine.

## Setup Plan

1. Extract the package under `Tool/`.
2. Verify that the entry script exists:

```text
Tool/netMHCpan-4.2/netMHCpan
```

3. Follow `netMHCpan-4.2/netMHCpan-4.2.readme` to configure paths if needed.
4. Run the package's bundled test files and compare output format with the provided `.out` files.

## Evaluation Inputs

Use these local files:

- BA: `Trainning dataset/NetMHCpan_train/c000-c004_ba`
- EL: `Trainning dataset/NetMHCpan_train/c000-c004_el`
- IEDB validation/test: `Trainning dataset/NetMHCpan_train/c000-c004_iedb` and `Trainning dataset/NetMHCpan_eval/iedb_test`
- CEDAR validation/test: `Trainning dataset/NetMHCpan_train/c000-c004_cedar` and `Trainning dataset/NetMHCpan_eval/cedar_test`

## Output Contract

For each prediction row, store:

- peptide
- allele
- context
- label
- NetMHCpan raw score
- NetMHCpan rank
- task/mode
- fold

## Metrics

Compute the same metrics later used for PocketBind-MHCI:

- ROC-AUC
- PR-AUC / AUPRC
- AUC0.1
- PPV@N
- PPV@top-k
- Spearman correlation for BA
- Pearson correlation for BA
- MSE for BA

## Success Criteria

Baseline reproduction is complete when:

- NetMHCpan-4.2 runs locally on a bundled example.
- All target evaluation files can be converted into NetMHCpan input format.
- Prediction outputs and metric reports are generated for BA, EL, IEDB, and CEDAR tasks.
- The same metric script can later evaluate PocketBind-MHCI predictions.

