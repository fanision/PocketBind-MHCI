# NetMHCpan-4.2 Baseline Results

Generated on 2026-06-12 with the local NetMHCpan-4.2c Darwin arm64 package.

## Reproduction Commands

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

## Results

| Dataset | Task | Rows predicted | Positives | Primary score | ROC-AUC | AUC0.1 | AUPRC | PPV@N | Pearson | Spearman | MSE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `c000_ba` | BA | 33,731 | 29,192 | `BA_score` | 0.963604 | 0.870948 | 0.904324 | 0.885928 | 0.886560 | 0.791647 | 0.016686 |
| `iedb_test` | IEDB/pathogen | 7,420 | 1,634 | `Pathogen_score` | 0.913949 | 0.736035 | 0.743497 | 0.694002 | n/a | n/a | n/a |
| `cedar_test` | CEDAR/neoepitope | 1,486 | 315 | `Neo_score` | 0.899134 | 0.783096 | 0.765643 | 0.742857 | n/a | n/a | n/a |

## Notes

- NetMHCpan cannot directly evaluate all alleles in the local files. For this first pass, only rows with unambiguous HLA-A/B/C alleles were included.
- `iedb_test` had 10,621 total rows, 7,420 valid NetMHCpan rows, and 3,201 skipped rows.
- `cedar_test` had 1,486 total rows and all rows were valid.
- `c000_ba` had 41,707 total rows, 33,731 valid NetMHCpan rows, and 7,976 skipped rows.
- The runner creates `/tmp/PocketBind_netMHCpan-4.2` as a symlink because NetMHCpan's internal shell calls fail when the installation path contains spaces.

