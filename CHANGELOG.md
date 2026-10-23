# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); tagged releases use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `py.typed` marker, so type checkers use the package's inline type hints.
- A warning is logged before the train/test Tanimoto check when it will
  compare more than 100 million pairs (roughly a 10k × 10k split), so a
  long run isn't mistaken for a hang.
- `challenge_data_checker.report_types`: `TypedDict` definitions of the
  report schema. `audit()`, `cli.run()` and `build_report()` are now
  annotated as returning `Report`. The returned value is still a plain
  dict, so this is no behaviour change.

### Changed

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
