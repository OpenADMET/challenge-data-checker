# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); tagged releases use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- SMILES parse failures now keep RDKit's own message (e.g. "unclosed ring",
  "Explicit valence for atom # 1 N, 5, is greater than permitted") instead
  of a generic "RDKit failed to parse SMILES". Each failure also gets a
  `category` for its likely root cause: `empty`, `control_characters`,
  `truncated`, `placeholder`, `markush`, `stoichiometry_text`, `valence`,
  `kekulisation`, `ring_closure`, `processing` or `other` (#23). The
  summary gains `total_unparseable_by_category`, and the dashboard and text
  report show the category with each failure.
- A SMILES that fails to parse but parses once surrounding whitespace is
  stripped (e.g. a CXSMILES with a leading space) is now kept, and flagged
  as `smiles_whitespace` (WARN). Warnings RDKit logs while parsing a valid
  SMILES (e.g. conflicting double-bond directions, so that bond's stereo is
  dropped) are flagged as `rdkit_warnings` (WARN), with the message (#23).
- Tab-separated input: `.tsv` files are now accepted alongside `.csv` and
  `.parquet`, as are gzip-compressed `.csv.gz` and `.tsv.gz`, locally or
  from a URL (#14).
- Salt-form leakage: a new `strip_salts` setting (TOML, `audit()`; **on by
  default**) compares each molecule's salt- and solvent-stripped,
  neutralised parent compound, so a salt in one pool matches its free base
  or another salt form in the other, e.g. `CCN(CC)CC.Cl` in train and
  `CCN(CC)CC` in test (#5). Matches are reported under the new
  `parent_inchikey` representation, across pools and within a pool. They
  are only reported when the rows don't all share one full InChIKey, so
  exact matches aren't repeated. Adds the summary counts
  `total_salt_form_leaked_train_molecules` /
  `total_salt_form_leaked_test_molecules`, `config.strip_salts` in the
  report, and a "Same Parent Compound, Different Salt Form" text report
  section. The Tanimoto check skips pairs that share a parent. Because the
  setting is on by default, existing audits can report new leakage.
- Stereoisomer and racemate/enantiomer matching via the InChIKey
  connectivity block (`inchikey_connectivity`), within a pool and across
  pools (#10). A connectivity match is only reported when the rows don't
  all share one full InChIKey, so it never repeats an exact match. The
  report gains `internal_duplicates[pool]["inchikey_connectivity"]`,
  `train_test_leakage["exact"]["inchikey_connectivity"]`, the summary counts
  `total_stereo_leaked_train_molecules` / `total_stereo_leaked_test_molecules`
  (rows whose only cross-pool match is stereo-insensitive), and a "Same
  Connectivity, Different Stereochemistry" section in the text report. The
  Tanimoto check now skips pairs that share a connectivity block, since
  they're reported there.
- New SMILES column aliases: `standardized_smiles`, `standardised_smiles`,
  `std_smiles`, `isomeric_smiles` and `cxsmiles` (#15, #7).
- `py.typed` marker, so type checkers use the package's inline type hints.
- A warning is logged before the train/test Tanimoto check when it will
  compare more than 100 million pairs (roughly a 10k × 10k split), so a
  long run isn't mistaken for a hang.
- `challenge_data_checker.report_types`: `TypedDict` definitions of the
  report schema. `audit()`, `cli.run()` and `build_report()` are now
  annotated as returning `Report`. The returned value is still a plain
  dict, so this is no behaviour change.

### Changed

- **Breaking: severity tiers and a three-state result (#11, #4).** Every
  finding is now either blocking (`FAIL`) or flagged for review (`WARN`),
  following four rules:
  - chemistry errors always FAIL
  - data-curation issues WARN in train and FAIL in test
  - exact leakage FAILs and non-exact leakage WARNs
  - heuristic quality flags WARN

  The README has the full table. The overall result is `PASS`, `WARN` or
  `FAIL` instead of `PASS`/`FAIL`, so, for example, duplicates within the
  training set or a racemate/enantiomer pair across pools no longer fail an
  audit on their own.
  - New report fields: `summary.status`, `summary.blocking` and
    `summary.flagged`, with `findings`, `rows` and a `by_type` breakdown.
  - Findings are counted as distinct problems and rows, not as checks that
    fired: one duplicate seen through raw SMILES, canonical SMILES and
    InChIKey is one finding (#4). The old banner ("N issue(s) flagged") is
    replaced by blocking and flagged finding and row counts.
  - The dashboard colours each count by tier (red ✗ blocking, yellow ⚠
    flagged) and adds a findings-by-severity table. Text report findings
    are tagged `[FAIL]` or `[WARN]`.
  - **CLI exit codes:** `2` when the result is `FAIL`, or `WARN` with the
    new `--strict` flag. `1` still means a configuration or input error.
    Previously the CLI exited `0` whenever the audit ran.
  - The Tanimoto check is WARN at the default threshold of `1.0`, where it
    only catches fingerprint collisions, and FAIL when you lower the
    threshold.
- **Breaking (dependency floors):** dependencies now declare tested lower
  bounds: `pandas>=2.2`, `numpy>=1.26`, `rdkit>=2024.3`, `rich>=13`, and in
  the extras `pyarrow>=14`, `huggingface_hub>=0.24`, `requests>=2.31`,
  `pytest>=8`, `pytest-cov>=5`. Environments with older versions will be
  upgraded, or refuse to install.
- Python 3.13, 3.14 and 3.15 are now supported: the `<3.13` cap on
  `requires-python` is removed. CI tests every supported Python with the
  newest dependencies, and Python 3.12 with the declared minimum versions.
- `environment.yml`: `rich` moved from the dev tools to the runtime
  dependencies, and its pin updated from 13.6.0 to 15.0.0.
- Salt and solvate recognition rewritten (#20). Counterions are now
  recognised whether drawn ionic or as the neutral acid/base (`.Cl`, `.Br`,
  TFA, mesylate, maleate, citrate, tartrate, pamoate, …), and hydrates and
  common crystallisation solvents are recognised. An entry is flagged under
  `salts_or_metal_complexes` when it contains a metal, or when it is a
  parent compound whose other fragments are all known counterions or
  solvents. Behaviour changes:
  - neutral-acid salts and hydrates (e.g. `CCN(CC)CC.Cl`) are now flagged
  - a lone counterion entry (e.g. a bare `C#N`) is no longer flagged as a
    salt; it is still a suspicious fragment
  - an entry made only of counterion-like fragments with no parent compound
    (e.g. `CC(=O)O.CC(=O)O`) is no longer flagged as a salt
  - an entry with an unrecognised extra fragment alongside a counterion is
    no longer flagged as a salt; it is still a mixture
- Every processed row now records its salt- and solvent-stripped,
  neutralised parent compound (`MoleculeRecord.parent_smiles` /
  `parent_inchikey`) and the classified stripped fragments, ready for
  parent-level leakage matching (#5).
- A SMILES with a control character inside it (e.g. `"CC\nO"`) is now an
  unparseable `control_characters` failure. RDKit stops reading at the
  first whitespace, so it used to be silently read as `CC`.
- RDKit's logging is now routed through Python's `rdkit` logger
  (`rdBase.LogToPythonLogger()`) and kept silent, as before, outside the
  checker's own per-row capture. Processing failures after a successful
  parse log their traceback at DEBUG (`-v`) and record the exception type
  (#7).
- SMILES column aliases now have a preference order (`smiles`,
  `canonical_smiles`, `standardized_smiles`, … `molecule`, `mol`). A file
  with several alias columns (e.g. `smi` and `structure`) used to be an
  error; now the most preferred one is used, and the warning names the
  columns passed over. Two columns that differ only by case are still an
  error.
- Files downloaded from a plain (non-HuggingFace) URL are now deleted once
  loaded, and a partially downloaded file is deleted if the connection
  fails. They used to be left in the system temp directory (#7).
  HuggingFace Hub downloads stay in `huggingface_hub`'s own cache as
  before.
- The suspicious-fragment check (≤3 heavy atoms) no longer flags
  legitimate small molecules on a new allowlist: methanol, ethanol,
  acetonitrile, formaldehyde, acetaldehyde, formic acid, formamide, simple
  amines and ethers, small hydrocarbons, halomethanes, dichloromethane,
  methanethiol, CO₂ and CS₂ (#3). Bare `C`, `C#N`, lone ions and lone
  heteroatoms (e.g. water) are still flagged.
- The report's `generated_at` timestamp is now in UTC (`+00:00`) instead of
  the machine's local time zone.
- `__version__` is read from the installed package metadata, so
  `pyproject.toml` is the single place the version is set.
- Ruff now enforces lint rules (`E`, `F`, `W`, `I`, `UP`, `B`, `BLE`, `SIM`,
  `RUF`). Existing code updated to pass; no behaviour change.

## [0.3.3] - 2026-09-15

### Changed

- `pydoclint` now requires type hints in docstring `Parameters`/`Returns`
  sections, checks return-type consistency against function signatures, and
  requires an explicit `Returns` section (documenting `None`) for functions
  that return nothing. All docstrings updated accordingly. No behaviour
  change.

## [0.3.2] - 2026-09-15

### Changed

- All docstrings reformatted from Google to NumPy style, and `pydoclint`'s
  configured style switched to match. No behaviour change.

## [0.3.1] - 2026-09-14

### Fixed

- The SMILES column can no longer silently resolve to a different column
  than configured: falling back to an alias match (`smiles`, `smi`,
  `structure`, `canonical_smiles`, `molecule`, `mol`) now logs a warning and
  the resolved column is recorded per source file under
  `config.resolved_smiles_columns` in the report.

## [0.3.0] - 2026-08-28

### Added

- Support for HuggingFace Hub and generic URL train/test data sources, via the
  new `remote` extra.

## [0.2.2] - 2026-07-14

### Changed

- Renamed leakage labels to "in train/test overlap" for clarity.

## [0.2.1] - 2026-07-14

### Added

- The audit run timestamp is now included in the report.

## [0.2.0] - 2026-07-14

### Added

- Rich colour-coded console dashboard summarising audit results.
- Direct Python API (`audit()`) for running audits without a TOML config.
- `report_format` is inferred from the `report_output` file extension when not
  set explicitly.

## [0.1.0] - 2026-07-10

Initial release.

[0.3.0]: https://github.com/OpenADMET/challenge-data-checker/releases/tag/v0.3.0
[0.2.2]: https://github.com/OpenADMET/challenge-data-checker/releases/tag/v0.2.2
[0.2.1]: https://github.com/OpenADMET/challenge-data-checker/releases/tag/v0.2.1
[0.2.0]: https://github.com/OpenADMET/challenge-data-checker/releases/tag/v0.2.0
[0.1.0]: https://github.com/OpenADMET/challenge-data-checker/releases/tag/v0.1.0
