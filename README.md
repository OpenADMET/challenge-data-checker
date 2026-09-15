# challenge-data-checker

[![Tests](https://github.com/OpenADMET/challenge-data-checker/actions/workflows/tests.yml/badge.svg)](https://github.com/OpenADMET/challenge-data-checker/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22780114.svg)](https://doi.org/10.5281/zenodo.22780114)

A command-line tool that audits machine learning benchmarking datasets
(train/test splits) for chemical data leakage, duplicates, and data quality
issues. Built for sequential processing of datasets up to ~10,000 molecules.

## Features

- Loads train/test data from `.csv` or `.parquet` files, local or hosted
  remotely (HuggingFace Hub or any other URL), with fuzzy SMILES column
  matching (falls back to case-insensitive and common alias matching, e.g.
  `structure`, `smi`, `mol`).
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
pip install -e ".[parquet,test,remote]"
```

The `parquet` extra pulls in `pyarrow` (only needed if your data files are
`.parquet`); the `test` extra pulls in `pytest`/`pytest-cov` for development;
the `remote` extra pulls in `huggingface_hub`/`requests` (only needed if a
`train_files`/`test_files` entry is a URL — see below).

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
  files (local paths and/or URLs, freely mixed), pooled together into a
  single training/test set respectively.
- `columns.smiles_column`: resolved per file — exact match first, then
  case-insensitive, then a unique alias match (`smiles`, `smi`, `structure`,
  `canonical_smiles`, `molecule`, `mol`). Falling back to an alias match logs
  a warning and is recorded per file under `config.resolved_smiles_columns`
  in the report, since it means a different column than configured is being
  audited.
- `columns.identifier_columns`: optional. A column missing from one file logs
  a warning and is simply skipped for that file's rows, without crashing.
- `settings.max_train_test_similarity`: pairs with Tanimoto similarity **>=**
  this threshold are flagged (excluding pairs already caught by exact
  InChIKey matches, which are reported separately).
- `settings.report_format`: `"json"` for machine-readable output, or `"txt"`
  for a human-readable plain-text rendering of the same findings. If omitted,
  it's inferred from `report_output`'s extension — `.txt` implies `"txt"`,
  anything else implies `"json"` — so you rarely need to set it explicitly.
  Set it to override that inference (e.g. write JSON to a `.txt` path).

### Remote data sources

Any `train_files`/`test_files` entry can be a URL instead of a local path —
useful for a blinded test set hosted on HuggingFace Hub or in a private
GitHub repo:

```toml
[paths]
train_files = ["data/train_hub.csv"]
test_files = [
    "https://huggingface.co/datasets/<org>/<dataset>/resolve/main/test_blinded.csv",
    "https://raw.githubusercontent.com/<org>/<repo>/<ref>/test_blinded.csv?token=<TOKEN>",
]
```

- A `huggingface.co/datasets/.../resolve/.../...` or `.../blob/.../...` URL
  (as copied from the Hub's "Files" tab, or a Hub download link) is fetched
  via `huggingface_hub`, which also caches it locally by repo/revision.
  Requires the `remote` extra. For a private/gated dataset, set the `HF_TOKEN`
  environment variable or run `huggingface-cli login` beforehand — there is
  no token field in the config.
- Any other `http(s)://` URL (e.g. a GitHub raw-content link) is downloaded
  directly. Requires the `remote` extra. For a private repo, include a valid
  access token in the URL itself, the same way you would in a browser.
- A failed download (bad URL, missing/expired token, network error) raises a
  clear error before any audit checks run.

### Python API

For interactive use (e.g. a Jupyter notebook), call `audit()` directly instead
of writing a TOML config — every setting is a keyword argument, and
`train`/`test` each accept a single `.csv`/`.parquet` path or URL (see
[Remote data sources](#remote-data-sources)), a single DataFrame, or a list
mixing any of these:

```python
import pandas as pd
from challenge_data_checker import audit

train_df = pd.read_csv("data/train_hub.csv")
test_df = pd.read_csv("data/test_blind.csv")

report = audit(
    train=train_df,
    test=test_df,
    identifier_columns=["compound_id", "external_reg_no"],
    max_train_test_similarity=0.95,
)
```

`report` is the same dict written to `report_output` by the CLI. All
`[settings]` keys are available as keyword arguments (`smiles_column`,
`identifier_columns`, `tautomer_standardisation`, `max_train_test_similarity`,
`fp_radius`, `fp_n_bits`). `report_output`/`report_format` work the same way
as in the config file, except both are optional: omit `report_output` to skip
writing a file entirely and only get the dict back. Pass `print_report=False`
to suppress the stdout dashboard (e.g. in a notebook loop over many configs).

Data sources can be mixed freely, e.g. `train=[train_df, "data/extra.csv"]`.
In the report, an in-memory DataFrame is labelled
`<in-memory dataframe #N (R rows)>` (by its position in the list) wherever a
file path would otherwise appear.

## Output

**stdout**: a colour-coded (via [rich](https://github.com/Textualize/rich))
summary dashboard with row counts, unparseable SMILES, and one table per
check category (exact/Tanimoto leakage, internal duplicates, identifier
namespace issues, quality filters) — green checkmarks for clean results, red
crosses with counts for flagged findings — followed by an overall PASS/FAIL
banner.

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

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution workflow, including
how we use the [Developer Certificate of Origin](https://developercertificate.org/)
and our policy on AI-assisted contributions.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). See [`CHANGELOG.md`](CHANGELOG.md)
for release history, and [`CITATION.cff`](CITATION.cff) if you use this tool in
your work.

## Citing

If this template supports work you publish, please cite it — see [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" button.

## Acknowledgements

We would like to thank our funders for their support of OpenADMET, in particular ARPAH, Radial (part of the Astera Institute (https://ror.org/00ydx1s47)), Schrödinger Inc, and the Gates Foundation.  We would also like to thank our partners Enamine, HuggingFace, OpenEye, CDD Vault, Discovery Life Sciences, and the beamline staff at NSLS-II for their support. 

This work is supported by the Advanced Research Projects Agency for Health (ARPA-H) under AVOID-OME, and Award Number 1AY1AX000035. The contents are those of the authors. They may not reflect the policies of the Department of Health and Human Services or the U.S. government. The content is solely the responsibility of the authors and does not necessarily represent the official views of the Advanced Research Projects Agency for Health.
