# Conversation Summary

Date context: initial planning happened around 2026-06-09 to 2026-06-11.

## User Goal

Develop a model to predict the binding strength between MHC class I molecules and peptides. The model should aim to be best-in-field, not limited to the papers already present in the local `Paper/` folder.

## Local Workspace Observations

The workspace contains:

```text
Paper/
Tool/
Trainning dataset/
```

The local training data appears to follow the NetMHCpan train/evaluation style:

```text
Trainning dataset/NetMHCpan_train/
Trainning dataset/NetMHCpan_eval/
```

Important files:

- `c000-c004_ba`: binding affinity data
- `c000-c004_el`: eluted ligand / presentation data
- `c000-c004_iedb`: IEDB epitope-related data
- `c000-c004_cedar`: CEDAR neoepitope-related data
- `pseudoseqs`: HLA allele to 34-aa pseudosequence mapping
- `iedb_test`: final external IEDB test
- `cedar_test`: final external CEDAR test

Observed scale:

- EL training data: about 3.47M rows per fold
- BA training data: about 41k-44k rows per fold
- Pseudosequences: 329 rows
- Allele list: 450 rows

Example BA format:

```text
AAAANTTAL 0.5639007975354975 HLA-B07:02 XXXXXXXXXXXX
```

Example EL format:

```text
AEQNRKDAEAW 1 Sarkizova_2020__A0202 EQLAEQEAWFNE
```

Example pseudosequence format:

```text
HLA-A02:01	YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY
```

## Literature And Method Notes

The project should learn from, and eventually compare against:

- NetMHCpan-4.2
- NetMHCpan-4.1 / 4.0
- MHCflurry 2.0
- TransPHLA / TransMut
- BigMHC
- CapsNet-MHC
- UniPMT
- DeepAttentionPan and related peptide-MHC interaction models

The first baseline is now explicitly **NetMHCpan-4.2**.

## Proposed Model

Working name: **PocketBind-MHCI**.

Core idea: combine the strengths of NetMHCpan-style BA/EL transfer learning, MHCflurry-style presentation modeling, Transformer peptide-HLA interaction modeling, and structure-inspired pocket/contact bias.

The model should not be only an IC50 regression model. It should be a pan-allele, multi-task binding/presentation model.

## Key Decisions

- First comparison target: NetMHCpan-4.2.
- NetMHCpan-4.2 tool package was initially Linux arm64, then replaced by a macOS arm64 package:

```text
Tool/netMHCpan-4.2c.Darwin_arm64.tar.gz
```

- First training implementation should use the local NetMHCpan-style data before adding external datasets.
- `iedb_test` and `cedar_test` should be reserved as final external tests.
- Large local data, PDFs, and third-party tool packages should not be committed to GitHub.

