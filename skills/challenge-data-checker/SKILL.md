---
name: challenge-data-checker
description: Check chemistry datasets for train/test leakage and data-quality problems with challenge-data-checker, and turn its report into a verdict. Use when asked to audit an ML benchmark/challenge split, check whether a data upload or public release leaks a blinded test set, or vet a single raw dataset (unparseable SMILES, duplicates, identifier conflicts, salts, stereo) before splitting or publishing it.
---

# challenge-data-checker

`challenge-data-checker` compares chemical structures (SMILES) and compound
identifiers between datasets. It has two modes:

- **audit**: a train/test split. Finds leakage between the pools plus
  problems within each.
- **check**: one dataset on its own. Finds the problems that don't need a
  split.

Your job is usually not just to run it. It's to decide what the findings
*mean*, for example "does this public release leak the blinded test set?",
and to report that clearly.

## 1. Run it

Install from GitHub (or `pip install -e ".[parquet,remote]"` in a checkout):

```bash
pip install "challenge-data-checker[parquet,remote] @ git+https://github.com/OpenADMET/challenge-data-checker"
```

### Audit a split (or a release against a test set)

Write a TOML config. Every path may be a local file or a URL:

```toml
[paths]
train_files = ["https://huggingface.co/datasets/org/release/resolve/main/inhibition.tsv"]
test_files = ["data/blinded_test.csv"]
report_output = "reports/audit.json"

[columns]
smiles_column = "SMILES"
identifier_columns = ["compound_id"]
```

```bash
challenge-data-checker audit.toml          # add --strict to also fail on WARN
```

Or from Python, which also accepts DataFrames, useful after pre-processing:

```python
from challenge_data_checker import audit
report = audit(train=release_df, test=test_df, identifier_columns=["compound_id"],
               print_report=False)
```

When checking **a release against a blinded test set**, put the release
files in `train` and the test set in `test`. Check **every file** in the
release, and in every related release, not just the obviously relevant one.
Ancillary files (well-level data, QC, mass-spec) can still contain test
compounds. Run one audit per release file, or pool them all in
`train_files`; the report's `source_file` says which file each row came from.

### Check a single dataset

```bash
challenge-data-checker check --files raw_a.csv raw_b.tsv --identifier-columns compound_id \
    --report-output reports/check.json
```

```python
from challenge_data_checker import check
report = check(["raw_a.csv", "raw_b.tsv"], identifier_columns=["compound_id"])
```

### Remote and private data

HuggingFace dataset URLs (`.../resolve/<rev>/<path>` or `.../blob/...`) are
downloaded with `huggingface_hub`. For a private or gated dataset, set
`HF_TOKEN` in the environment. Never paste a token into a config or report.
Other HTTP(S) URLs are fetched directly.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | `PASS`, or `WARN` without `--strict` |
| `1` | Couldn't run: bad config, missing file, unresolvable column, usage error |
| `2` | `FAIL`: blocking findings, or `WARN` with `--strict` |

## 2. Pre-process only what the checker can't handle

The checker handles these itself, so don't convert them:
- `.csv`, `.tsv`, `.csv.gz`, `.tsv.gz` and `.parquet`
- common SMILES column names: `smiles`, `canonical_smiles`,
  `standardized_smiles`, `std_smiles`, `isomeric_smiles`, `cxsmiles`,
  `smi`, `structure`, …
- whitespace around identifier values
- salt forms, through the `strip_salts` setting, which is on by default

You still need to handle:

- **Other formats** (`.json`, `.jsonl`, `.xlsx`, SDF, …): load them into a
  pandas DataFrame and pass it to `audit()` / `check()`.
- **An unrecognised SMILES column name**: set `smiles_column` explicitly.
- **Identifiers in different forms.** A release often uses batch or sample
  IDs while the test set uses parent IDs. For example,
  `OCNT-0493952-AA-001` vs `OCNT-0493952`. Identifier overlap only matches
  identical values (after stripping whitespace), so derive a parent-ID
  column first:

  ```python
  release_df["parent_id"] = release_df["batch_id"].str.extract(r"^(OCNT-\d+)")
  ```

  Then use a column name present in both datasets (rename one if needed) in
  `identifier_columns`. If IDs differ only in case or `-`/`_`, the checker
  flags them as near-duplicates (WARN). Set
  `identifier_case_insensitive = true` if case is meaningless in the
  scheme.
- **`NA` / `N/A` cells**: pandas reads them as missing values, so such
  rows show up as `empty` unparseable SMILES. That's expected.

## 3. Read the report

Start at `report["summary"]`:

- `status`: `PASS` / `WARN` / `FAIL`.
- `blocking` / `flagged`: `findings` (distinct problems), `rows` (distinct
  affected rows) and `by_type`, the counts per finding type. This tells you
  which sections to open.

Each finding lists its rows as `{pool, source_file, row_index, raw_smiles}`.
The sections:

| Report key | What it holds |
|---|---|
| `train_test_leakage.exact` | Shared values per representation. `raw_smiles`, `canonical_smiles` and `inchikey` are exact. `parent_inchikey` is the same compound in a different salt form. `inchikey_connectivity` is the same skeleton with different stereo. |
| `train_test_leakage.identifiers` | Identifier values present in both pools |
| `train_test_leakage.tanimoto_similarity` | Pairs at or above `max_train_test_similarity` |
| `internal_duplicates.<pool>` | Repeats within one pool, per representation and identifier column |
| `identifier_namespace_issues.<col>` | An ID with several structures (each with a `conflict_type`), a compound with several IDs, near-duplicate or whitespace-padded IDs, IDs whose SMILES is broken or empty in some rows |
| `quality_flags.<pool>` | Salts/solvates, metals, multi-component entries, homodimers, net charge, undefined stereo, suspicious fragments, RDKit warnings |
| `inconsistent_smiles` | Structures written as more than one SMILES string, with the files using each form |
| `unparseable_smiles` | Rows that didn't parse, with RDKit's message, a `category`, and `resolvable_via_identifier` |

`format_report_as_text` (or `report_format = "txt"`) gives the same content
as readable text, with each finding tagged `[FAIL]` or `[WARN]`.

## 4. Triage: what does each finding mean?

The tiers are a starting point, not the verdict. For leakage, what matters
is **what the leaked rows expose**:

- **Blocking leak (ground truth exposed).** A test compound appears in a
  file that also has its measured value for the challenge endpoint (an
  activity, IC50, % inhibition, readout column, …). Exact, salt-form or
  identifier overlap counts. Anyone with the release can read off the
  answer. Report it as a leak and list the IDs and values.
- **Test-set membership disclosure.** A test compound appears in a file
  *without* the endpoint value, e.g. mass-spec ionisation areas, QC or
  inventory lists. The score isn't exposed, but the release reveals which
  compounds are in the blinded test set, which can still bias
  participants. Report it as disclosure, not as a ground-truth leak.
- **Probably benign.** Stereo-only overlap (`inchikey_connectivity`), where
  a racemate in one set and an enantiomer in the other can be a deliberate
  split. Tanimoto pairs at the default threshold of `1.0` are fingerprint
  collisions between different structures. Mention them, and say whether
  they look deliberate.

Within a dataset:

- **Expected noise in a pooled training set:**
  - duplicates within train (replicates, or combined sources)
  - salts and solvates
  - undefined stereo in racemic compounds
  - one compound with several IDs (vendor vs internal)
- **Real problems:**
  - unparseable SMILES (FAIL)
  - one ID with genuinely different compounds (`conflict_type:
    different_compound`, FAIL)
  - homodimers, i.e. a 2:1 salt with its acid stripped (FAIL)
  - structures written as different SMILES in different files
    (`inconsistent_smiles`): the files probably went through different
    processing pipelines, so check that nothing else differs between them
  - multi-component entries (two drugs concatenated)
  - duplicates within the *test* set (FAIL: they double-weight scoring)
- **Fixable:** empty SMILES with `resolvable_via_identifier` set can be
  filled in from the other row with that ID. An ID with both a broken and a
  valid SMILES shows which structure was meant.

Check the column contents before calling something a leak. Open the
source file and confirm what the overlapping rows contain.

## 5. Report back

Give a short verdict first, then the evidence:

1. **Verdict.** Leak / disclosure only / clean, or for a check: ready /
   needs fixes.
2. **Affected compounds**: IDs and SMILES, grouped by file, with what each
   file exposes (which value columns).
3. **Counts**: blocking and flagged findings and rows from the summary.
4. **Anything you deliberately set aside** as expected noise, and why.

Keep tokens, private URLs and unpublished test-set contents out of anything
that will be shared publicly.
