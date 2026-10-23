# challenge-data-checker

[![Tests](https://github.com/OpenADMET/challenge-data-checker/actions/workflows/tests.yml/badge.svg)](https://github.com/OpenADMET/challenge-data-checker/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22780114.svg)](https://doi.org/10.5281/zenodo.22780114)

A command-line tool that audits machine learning benchmarking datasets
(train/test splits) for chemical data leakage, duplicates, and data quality
issues. Built for sequential processing of datasets up to ~10,000 molecules.

## Features

- Loads train/test data from `.csv`, `.tsv` (optionally gzipped, e.g.
  `.tsv.gz`) or `.parquet` files, local or hosted remotely (HuggingFace Hub
  or any other URL), with fuzzy SMILES column matching (falls back to
  case-insensitive and common alias matching, e.g. `standardized_smiles`,
  `structure`, `smi`).
- Parses every SMILES with RDKit, with optional tautomer standardisation, and
  computes canonical SMILES, InChIKey, and a stereo-blind Morgan fingerprint.
  A SMILES that fails to parse is reported with RDKit's own message and a
  likely root cause: `empty`, `control_characters`, `truncated`,
  `placeholder`, `markush`, `stoichiometry_text`, `valence`,
  `kekulisation`, `ring_closure`, `processing` or `other`. A value that only
  parses once surrounding whitespace is stripped (e.g. a CXSMILES with a
  leading space) is kept and flagged. Warnings RDKit logs while parsing a
  valid SMILES (e.g. conflicting double-bond directions, so that bond's
  stereo is dropped) are flagged too. A SMILES with a control character
  inside it, such as an embedded newline, is treated as a failure: RDKit
  would otherwise silently read only the part before it.
- **Train-test leakage**: exact matches on raw SMILES, canonical SMILES,
  InChIKey, or any configured identifier column. Stereoisomers and
  racemate/enantiomer pairs (same InChIKey connectivity block, different
  stereo) are reported separately as stereo-only overlap.
- **Tanimoto similarity fallback**: bulk pairwise Tanimoto comparison between
  every train/test fingerprint pair (via RDKit's `BulkTanimotoSimilarity`) to
  catch near-duplicates. Pairs already reported as an exact or stereo-only
  match are skipped.
- **Internal duplicates**: repeated values within a single split (train-only
  or test-only), per representation (including stereoisomers and
  racemate/enantiomer pairs, via the InChIKey connectivity block) and per
  identifier column.
- **Identifier namespace checks**: flags an identifier that maps to more than
  one structure, or a compound that's given more than one identifier, within
  the same identifier column. Each identifier conflict is classified as
  `salt_form`, `stereo`, `tautomer` or `different_compound` (with the
  highest Tanimoto similarity between the structures), and only
  `different_compound` blocks. A compound with several identifiers is
  found even when its salt form or stereo annotation differs between rows. Identifier values are compared with
  surrounding whitespace stripped. Values that differ only in case or
  `-`/`_` are flagged as near-duplicates. Rows whose SMILES didn't parse are
  related back to the identifier as well: an identifier with a valid
  structure in some rows and a broken SMILES in others is flagged, and an
  empty SMILES whose identifier has exactly one structure elsewhere is
  reported as resolvable (with the candidate structure).
- **Quality filters**: multi-fragment entries are split by what they are:
  - **salt or solvate:** a parent compound plus known counterions or
    solvents, drawn either ionic or neutral (e.g. `.[Cl-]` or `.Cl`, TFA,
    mesylate, maleate, citrate, hydrates, common crystallisation solvents),
    and possibly further copies of the parent, as in a 2:1 salt
  - **multi-component:** two or more distinct unrecognised compounds of at
    least `multi_component_min_heavy_atoms` heavy atoms, e.g. two drugs
    concatenated by a spreadsheet error
  - **homodimer without counterion:** `X.X`, a 2:1 salt with its acid
    stripped
  - **other mixtures:** anything else with several fragments

  The parent is the largest fragment that isn't a known counterion or
  solvent, neutralised, so a large counterion such as pamoate is never
  mistaken for the compound. Separately flagged:
  - entries whose formal charges don't sum to zero, such as a carboxylate
    whose cation was stripped (zwitterions and nitro groups are neutral and
    aren't flagged)
  - entries containing a metal
  - entries with undefined stereochemistry: unspecified stereocentres
    (including ring cis/trans and sulfoxides) or stereo double bonds, which
    leave it ambiguous whether the compound is a racemate, unknown, or had
    its stereo lost in a pipeline
  - suspicious tiny fragments that look like stripped degradation
    artefacts (e.g. a bare cyanide left over from a failed preprocessing
    pipeline); common small molecules such as methanol, ethanol,
    acetonitrile and dichloromethane are allowlisted and not flagged
- Prints a scannable summary dashboard to stdout and writes a detailed report
  (JSON or plain text) with exact row indices, file paths, and pairwise
  similarity scores for every flagged issue.

## Installation

Requires Python 3.12 or newer. Supported versions and dependency ranges are at
least as wide as [SPEC 0](https://scientific-python.org/specs/spec-0000/)
recommends. CI tests Python
3.12–3.15 with the newest dependencies, and Python 3.12 with the lowest
versions `pyproject.toml` allows (pandas 2.2, numpy 1.26, RDKit 2024.03).

Install into any environment with pip:

```bash
pip install -e ".[parquet,test,remote]"
```

For development, the conda environment pins exact versions of every
dependency and tool (it's what CI's lint and type-check job uses):

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

Exit codes, so CI can gate on the result:

| Code | Meaning |
|---|---|
| `0` | `PASS`, or `WARN` (flagged findings only) |
| `1` | The audit couldn't run: configuration or input error |
| `2` | `FAIL` (blocking findings), or `WARN` when run with `--strict` |

Pass `--strict` to also fail on flagged (`WARN`) findings.

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
strip_salts = true                   # default true (also compare salt-stripped parents)
identifier_case_insensitive = false  # default false (case-only id differences only WARN)
multi_component_min_heavy_atoms = 6  # default 6 (size of a fragment counted as a compound)
max_train_test_similarity = 1.0      # default 1.0 (Tanimoto threshold, inclusive)
fp_radius = 2                        # default 2 (Morgan fingerprint radius)
fp_n_bits = 2048                     # default 2048 (Morgan fingerprint bit length)
report_format = "json"               # default "json"; also accepts "txt"
```

- `paths.train_files` / `paths.test_files`: one or more `.csv`, `.tsv`,
  `.csv.gz`, `.tsv.gz` or `.parquet` files (local paths and/or URLs, freely
  mixed), pooled together into a single training/test set respectively.
- `columns.smiles_column`: resolved per file — exact match first, then
  case-insensitive, then an alias match. Aliases in order of preference:
  `smiles`, `canonical_smiles`, `standardized_smiles`, `standardised_smiles`,
  `std_smiles`, `isomeric_smiles`, `cxsmiles`, `smi`, `structure`,
  `molecule`, `mol`. If a file has several, the earliest one is used.
  Falling back to an alias match logs a warning (naming any alias columns
  passed over) and is recorded per file under
  `config.resolved_smiles_columns` in the report, since it means a
  different column than configured is being audited.
- `columns.identifier_columns`: optional. A column missing from one file logs
  a warning and is simply skipped for that file's rows, without crashing.
- `settings.strip_salts`: also compare each molecule's parent compound, with
  known counterions and solvents stripped and charges neutralised, so a salt
  in one pool matches its free base or another salt form in the other (e.g.
  `CCN(CC)CC.Cl` in train and `CCN(CC)CC` in test). Matches are reported as
  salt-form-only overlap and as `parent_inchikey` duplicates within a pool.
  On by default; set `false` to compare only the structures as drawn.
- `settings.identifier_case_insensitive`: identifier values always have
  surrounding whitespace stripped before comparison (` CPD-1` and `CPD-1`
  are the same identifier, and the padded value is flagged). With this set
  to `true`, values that differ only in letter case are also the same
  identifier. Otherwise they, and values differing only in `-`/`_` (e.g.
  `Molport-1` vs `MolPort_1`), are reported as near-duplicate identifiers
  (WARN), since they may be genuinely different IDs in an inconsistent
  scheme.
- `settings.max_train_test_similarity`: pairs with Tanimoto similarity **>=**
  this threshold are flagged (excluding pairs already reported as an exact,
  salt-form-only or stereo-only match).
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
`train`/`test` each accept a single data file path or URL (see
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
`identifier_columns`, `tautomer_standardisation`, `strip_salts`,
`identifier_case_insensitive`, `multi_component_min_heavy_atoms`,
`max_train_test_similarity`,
`fp_radius`, `fp_n_bits`). `report_output`/`report_format` work the same way
as in the config file, except both are optional: omit `report_output` to skip
writing a file entirely and only get the dict back. Pass `print_report=False`
to suppress the stdout dashboard (e.g. in a notebook loop over many configs).

Data sources can be mixed freely, e.g. `train=[train_df, "data/extra.csv"]`.
In the report, an in-memory DataFrame is labelled
`<in-memory dataframe #N (R rows)>` (by its position in the list) wherever a
file path would otherwise appear.

## Output

### Severity tiers

Every finding is either **blocking** (`FAIL`: the split shouldn't be released
as is) or **flagged** (`WARN`: a human should check it). The overall status
is `FAIL` if anything is blocking, `WARN` if anything is flagged, and `PASS`
otherwise. Four rules decide the tier:

1. Chemistry errors always FAIL, in both pools.
2. Data-curation issues depend on the pool: WARN in train (a training set
   may combine sources or hold replicates the user has to resolve), FAIL in
   test (the test set must be scoreable ground truth).
3. Exact leakage is FAIL; non-exact leakage is WARN.
4. Heuristic quality flags are WARN.

| Finding | Train | Test | Train/test |
|---|---|---|---|
| Unparseable SMILES | FAIL | FAIL | |
| Exact overlap (raw / canonical SMILES, InChIKey) | | | FAIL |
| Same parent compound, different salt form (`strip_salts`) | | | FAIL |
| Identifier overlap | | | FAIL |
| Same connectivity, different stereochemistry | | | WARN |
| Tanimoto similarity ≥ threshold | | | WARN at the default `1.0`; FAIL if you lower it |
| Duplicate structure, salt form or identifier | WARN | FAIL | |
| Duplicate stereoisomer (e.g. racemate and enantiomer) | WARN | WARN | |
| One identifier → several different compounds | | | FAIL |
| One identifier → salt-form, stereo or tautomer variants of one compound | | | WARN |
| One compound (any salt form or stereo) → several identifiers | | | WARN |
| Identifier with surrounding whitespace; near-duplicate identifiers (case or `-`/`_` only) | | | WARN |
| Identifier with valid and unparseable SMILES; empty SMILES resolvable via its identifier | | | WARN (the unparseable rows themselves still FAIL) |
| Homodimer without counterion; net formal charge ≠ 0 | FAIL | FAIL | |
| Salt or solvate, contains metal, multi-component, other mixtures, suspicious fragments, undefined stereochemistry | WARN | WARN | |
| SMILES that only parses after stripping whitespace; RDKit parse warnings | WARN | WARN | |

Counts are of distinct problems and rows, not of checks that fired: a
duplicate seen through raw SMILES, canonical SMILES and InChIKey is one
finding. The report's `summary.status`, `summary.blocking` and
`summary.flagged` give the status and the per-tier counts (`findings`,
`rows`, and a breakdown `by_type`).

**stdout**: a colour-coded (via [rich](https://github.com/Textualize/rich))
summary dashboard with row counts, unparseable SMILES, a findings-by-severity
table, and one table per check category (leakage, internal duplicates,
identifier namespace issues, quality filters): green checkmarks for clean
results, red crosses for blocking findings, yellow warning signs for flagged
ones. It ends with an overall PASS / WARN / FAIL banner.

**Report file** (written to `paths.report_output`, in the format chosen by
`settings.report_format`): the full detail behind every summary count,
including exact row indices, source file paths, and similarity scores for
every flagged finding.

- `"json"` — a single machine-readable JSON document, suitable for
  programmatic parsing or diffing between runs.
- `"txt"` — the same information as readable, indented plain text (config,
  summary dashboard, then one section per check with each finding tagged
  `[FAIL]` or `[WARN]` and its rows rendered as
  `[pool] file (row N): 'raw_smiles'` lines), suitable for a human to scan or
  paste into an issue/PR.

## Development

```bash
pytest                       # run the test suite
black src tests              # format
ruff check src tests         # lint
mypy src                     # type check
pydoclint src                # docstring style/consistency check
```

The chemistry helpers (parsing, standardisation, counterion and solvent
recognition) live in this package rather than coming from
`openadmet.toolkit`. That's deliberate: the checker's salt and parent
handling is tuned for leakage detection and differs from the toolkit's
general-purpose standardisation, and keeping it here keeps the
dependencies light for a standalone tool.

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
