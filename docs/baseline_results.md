# NetMHCpan-4.2 Baseline Results

Generated on 2026-06-12 with the local NetMHCpan-4.2c Darwin arm64 package.

## Reproduction Commands

Full suite:

```bash
PYTHONPATH=src python3 scripts/run_netmhcpan_suite.py \
  --out-dir artifacts/netmhcpan42_suite
```

Single datasets:

```bash
PYTHONPATH=src python3 scripts/run_netmhcpan_baseline.py \
  --input "Trainning dataset/NetMHCpan_train/c000_ba" \
  --task ba

PYTHONPATH=src python3 scripts/run_netmhcpan_baseline.py \
  --input "Trainning dataset/NetMHCpan_eval/iedb_test" \
  --task iedb

PYTHONPATH=src python3 scripts/run_netmhcpan_baseline.py \
  --input "Trainning dataset/NetMHCpan_eval/cedar_test" \
  --task cedar
```

Metrics:

```bash
PYTHONPATH=src python3 scripts/evaluate_predictions.py \
  --merged artifacts/netmhcpan42/c000_ba.ba/merged_predictions.tsv \
  --task ba \
  --out artifacts/netmhcpan42/c000_ba.ba/metrics.tsv

PYTHONPATH=src python3 scripts/evaluate_predictions.py \
  --merged artifacts/netmhcpan42/iedb_test.iedb/merged_predictions.tsv \
  --task iedb \
  --out artifacts/netmhcpan42/iedb_test.iedb/metrics.tsv

PYTHONPATH=src python3 scripts/evaluate_predictions.py \
  --merged artifacts/netmhcpan42/cedar_test.cedar/merged_predictions.tsv \
  --task cedar \
  --out artifacts/netmhcpan42/cedar_test.cedar/metrics.tsv
```

## Five-Fold Suite Results

| Task | Metric | Mean | SD |
|---|---:|---:|---:|
| BA | Pearson | 0.888063 | 0.001547 |
| BA | Spearman | 0.782867 | 0.007379 |
| BA | MSE | 0.016548 | 0.000211 |
| BA | ROC-AUC | 0.965193 | 0.001422 |
| BA | AUC0.1 | 0.877649 | 0.005253 |
| BA | AUPRC | 0.909565 | 0.004276 |
| BA | PPV@N | 0.888083 | 0.003063 |
| IEDB/pathogen | ROC-AUC | 0.879929 | 0.005449 |
| IEDB/pathogen | AUC0.1 | 0.693551 | 0.013094 |
| IEDB/pathogen | AUPRC | 0.755915 | 0.011896 |
| IEDB/pathogen | PPV@N | 0.680012 | 0.012194 |
| CEDAR/neoepitope | ROC-AUC | 0.897727 | 0.023036 |
| CEDAR/neoepitope | AUC0.1 | 0.755612 | 0.013454 |
| CEDAR/neoepitope | AUPRC | 0.763154 | 0.021995 |
| CEDAR/neoepitope | PPV@N | 0.744417 | 0.013659 |

## Key Baseline Result Files

The external test rows are `iedb_test` and `cedar_test`; `c000_ba` is kept here as the first full BA fold result.

| Dataset | Task | Rows predicted | Positives | Primary score | ROC-AUC | AUC0.1 | AUPRC | PPV@N | Pearson | Spearman | MSE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `c000_ba` | BA | 33,731 | 8,299 | `BA_score` | 0.963604 | 0.870948 | 0.904324 | 0.885928 | 0.886560 | 0.791647 | 0.016686 |
| `iedb_test` | IEDB/pathogen | 7,420 | 1,634 | `Pathogen_score` | 0.913949 | 0.736035 | 0.743497 | 0.694002 | n/a | n/a | n/a |
| `cedar_test` | CEDAR/neoepitope | 1,486 | 315 | `Neo_score` | 0.899134 | 0.783096 | 0.765643 | 0.742857 | n/a | n/a | n/a |

## Notes

- NetMHCpan cannot directly evaluate all alleles in the local files. For this first pass, only rows with unambiguous HLA-A/B/C alleles were included.
- `iedb_test` had 10,621 total rows, 7,420 valid NetMHCpan rows, and 3,201 skipped rows.
- `cedar_test` had 1,486 total rows and all rows were valid.
- `c000_ba` had 41,707 total rows, 33,731 valid NetMHCpan rows, and 7,976 skipped rows.
- The runner creates `/tmp/PocketBind_netMHCpan-4.2` as a symlink because NetMHCpan's internal shell calls fail when the installation path contains spaces.
