# challenge-data-checker

A command-line tool that audits machine learning benchmarking datasets
(train/test splits) for chemical data leakage, duplicates, and data quality
issues. Built for sequential processing of datasets up to ~10,000 molecules.

## Features

- Loads train/test data from `.csv` or `.parquet` files, with fuzzy SMILES
  column matching (falls back to case-insensitive and common alias matching,
  e.g. `structure`, `smi`, `mol`).
- Parses every SMILES with RDKit, with optional tautomer standardisation, and
  computes canonical SMILES, InChIKey, and a stereo-blind Morgan fingerprint.
- **Train-test leakage**: exact matches on raw SMILES, canonical SMILES,
  InChIKey, or any configured identifier column.
- **Tanimoto similarity fallback**: bulk pairwise Tanimoto comparison between
  every train/test fingerprint pair (via RDKit's `BulkTanimotoSimilarity`) to
  catch near-duplicates and stereo-mismatches that slip past InChIKey.
- **Internal duplicates**: repeated values within a single split (train-only
  or test-only), per representation and per identifier column.
- **Identifier namespace checks**: flags an identifier that maps to more than
  one structure, or a structure that's given more than one identifier, within
  the same identifier column.
- **Quality filters**: mixtures/disconnected fragments, salts and metal
  complexes, and suspicious tiny fragments that look like stripped
  degradation artefacts (e.g. a bare cyanide left over from a failed
  preprocessing pipeline).
- Prints a scannable summary dashboard to stdout and writes a detailed report
  (JSON or plain text) with exact row indices, file paths, and pairwise
  similarity scores for every flagged issue.

## Installation

Requires Python 3.12. Create and activate the conda environment, then install
the package in editable mode:

```bash
conda env create -f environment.yml
conda activate challenge-data-checker
pip install -e ".[parquet,test]"
```

The `parquet` extra pulls in `pyarrow` (only needed if your data files are
`.parquet`); the `test` extra pulls in `pytest`/`pytest-cov` for development.

## Usage

```bash
challenge-data-checker path/to/config.toml
```

Add `-v`/`--verbose` for debug-level logging (e.g. to see every identifier
column warning).

### Configuration file

```toml
[paths]
train_files = ["data/train_hub.csv"]
test_files = ["data/test_blind.csv"]
report_output = "reports/leakage_audit_report.json"

[columns]
smiles_column = "SMILES"
identifier_columns = ["compound_id", "external_reg_no"]

[settings]
tautomer_standardisation = false     # default false
max_train_test_similarity = 1.0      # default 1.0 (Tanimoto threshold, inclusive)
fp_radius = 2                        # default 2 (Morgan fingerprint radius)
fp_n_bits = 2048                     # default 2048 (Morgan fingerprint bit length)
report_format = "json"               # default "json"; also accepts "txt"
```

- `paths.train_files` / `paths.test_files`: one or more `.csv`/`.parquet`
  files, pooled together into a single training/test set respectively.
- `columns.smiles_column`: resolved per file — exact match first, then
  case-insensitive, then a unique alias match (`smiles`, `smi`, `structure`,
  `canonical_smiles`, `molecule`, `mol`).
- `columns.identifier_columns`: optional. A column missing from one file logs
  a warning and is simply skipped for that file's rows, without crashing.
- `settings.max_train_test_similarity`: pairs with Tanimoto similarity **>=**
  this threshold are flagged (excluding pairs already caught by exact
  InChIKey matches, which are reported separately).
- `settings.report_format`: `"json"` (default) for machine-readable output,
  or `"txt"` for a human-readable plain-text rendering of the same findings.
  `report_output` can be given any filename/extension you like — the format
  written is controlled entirely by this setting, not by the file extension.

## Output

**stdout**: a summary dashboard with row counts, unparseable SMILES, exact
vs. Tanimoto leakage counts, internal duplicate counts per representation and
identifier column, namespace issue counts, and quality-filter counts.

**Report file** (written to `paths.report_output`, in the format chosen by
`settings.report_format`): the full detail behind every summary count,
including exact row indices, source file paths, and similarity scores for
every flagged finding.

- `"json"` — a single machine-readable JSON document, suitable for
  programmatic parsing or diffing between runs.
- `"txt"` — the same information as readable, indented plain text (config,
  summary dashboard, then one section per check with each finding rendered
  as `[pool] file (row N): 'raw_smiles'` lines), suitable for a human to
  scan or paste into an issue/PR.

## Development

```bash
pytest                       # run the test suite
black src tests              # format
ruff check src tests         # lint
mypy src                     # type check
pydoclint src                # docstring style/consistency check
```
