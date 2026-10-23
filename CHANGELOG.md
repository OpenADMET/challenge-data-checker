# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); tagged releases use
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-10-23

A large release: severity tiers with a three-state result and CI-friendly
exit codes, salt-form and stereoisomer leakage detection, a single-dataset
`check` mode, and much finer-grained chemistry and identifier checks.

### Breaking changes

- **Overall result is `PASS` / `WARN` / `FAIL`**, from blocking and flagged
  severity tiers (see Changed). Findings that used to fail an audit, such
  as duplicates within the training set, may now only warn, and some new
  findings fail.
- **CLI exit codes:** `2` for `FAIL` (or `WARN` with `--strict`), `1` for a
  configuration, input or usage error. Previously the CLI exited `0`
  whenever the audit ran.
- **Report schema:**
  - `summary` gains `status`, `blocking`, `flagged` and
    `total_quality_flags`
  - reports carry `"mode": "audit"` (or `"check"`)
  - unparseable entries gain `category` and `resolvable_via_identifier`
  - identifier findings gain `conflict_type`
  - the summary's issue total is now distinct findings and rows, not
    checks fired
- **`mixtures` and `salts_or_metal_complexes` are deprecated** totals, no
  longer tiered. The specific categories that replace them are listed under
  Changed.
- **`strip_salts` is on by default**, so a salt and its free base across
  pools are now reported as leakage (FAIL).
- **Identifier values are compared with surrounding whitespace stripped**,
  which can surface identifier leakage that was previously hidden.
- **Dependency floors:** `pandas>=2.2`, `numpy>=1.26`, `rdkit>=2024.3`,
  `rich>=13` (and floors on the extras). Python 3.13-3.15 are now
  supported.

### Added

- **Single-dataset `check` mode (#19):** `challenge-data-checker check
  --files …` (or `check --config check.toml`) and `check()` in Python run
  every check that doesn't need a train/test split (unparseable SMILES,
  duplicates, identifier consistency, quality flags) on one dataset, with
  the same severity tiers, exit codes and `--strict`.
  - Duplicates within the checked dataset are WARN.
  - A check TOML takes `[paths] files` with an optional `report_output`,
    and every CLI flag maps to a TOML key.
  - Combining `--config` with flags, or passing the wrong kind of config to
    either command, is a clear error.
- **Salt-form leakage (#5):** a new `strip_salts` setting (on by default)
  compares each molecule's salt- and solvent-stripped, neutralised parent
  compound. A salt in one pool then matches its free base or another salt
  form in the other (e.g. `CCN(CC)CC.Cl` vs `CCN(CC)CC`), reported under
  the `parent_inchikey` representation and as salt-form-only overlap.
- **Stereoisomer matching (#10):** the InChIKey connectivity block
  (`inchikey_connectivity`) finds racemate/enantiomer pairs and other
  stereoisomers within and across pools, reported as stereo-only overlap
  (WARN).
  - These matches and salt-form matches are only reported when the rows
    don't all share one full InChIKey, so exact matches aren't repeated.
  - The Tanimoto check skips pairs already reported this way.
- **Parse-failure diagnosis (#23):** unparseable SMILES keep RDKit's own
  message, with a `category` for the likely root cause: `empty`,
  `control_characters`, `truncated`, `placeholder`, `markush`,
  `stoichiometry_text`, `valence`, `kekulisation`, `ring_closure`,
  `processing` or `other`.
  - A SMILES that only parses after stripping surrounding whitespace (e.g.
    a CXSMILES with a leading space) is kept and flagged
    (`smiles_whitespace`, WARN).
  - RDKit warnings on valid SMILES (e.g. conflicting double-bond
    directions) are flagged with their message (`rdkit_warnings`, WARN).
- **Identifier hygiene (#24, #25):**
  - `identifier_whitespace` (WARN) for padded values
  - a new `identifier_case_insensitive` setting
  - near-duplicate identifiers that differ only in case or `-`/`_` (WARN)
  - identifiers with both a valid and a broken SMILES (WARN)
  - empty SMILES that can be filled in from another row with the same
    identifier (WARN; `resolvable_via_identifier` on the unparseable entry)
- **Quality flags (#26, #19):**
  - `salt_or_solvate`, `contains_metal`, `multi_component` (with a new
    `multi_component_min_heavy_atoms` setting) and `unclassified_mixture`
    (WARN)
  - `homodimer_without_counterion` (FAIL: a 2:1 salt whose acid was
    stripped, so the stoichiometry is wrong)
  - `net_charge` (WARN: an entry drawn without its counterion, e.g. a
    sodium salt drawn as the bare anion, or a quaternary ammonium without
    its anion)
  - `undefined_stereo` (WARN: unspecified stereocentres or stereo double
    bonds)
- **Inconsistent SMILES (WARN):** a structure written as two or more
  different SMILES strings, across all files and pools, with the files
  using each form. This usually means the data went through different
  processing pipelines, which re-canonicalising would otherwise hide.
  Reported under `inconsistent_smiles`, with `summary.total_inconsistent_smiles`.
- **Input:** `.tsv`, `.csv.gz` and `.tsv.gz` files, locally or from a URL
  (#14), and the SMILES column aliases `standardized_smiles`,
  `standardised_smiles`, `std_smiles`, `isomeric_smiles` and `cxsmiles`
  (#15).
- `skills/challenge-data-checker/SKILL.md`, an Agent Skills file teaching
  an AI agent to run the checker and triage findings into ground-truth
  leaks, test-set membership disclosure, or expected noise (#16).
- `challenge_data_checker.report_types`: `TypedDict` definitions of the
  report schema, and a `py.typed` marker.
- A warning before a train/test Tanimoto comparison of more than 100
  million pairs, so a long run isn't mistaken for a hang.

### Changed

- **Severity tiers (#11).** Every finding is blocking (`FAIL`) or flagged
  for review (`WARN`), following four rules:
  - chemistry errors always FAIL
  - data-curation issues WARN in train and FAIL in test
  - exact leakage FAILs and non-exact leakage WARNs
  - heuristic quality flags WARN

  The README has the full table. The Tanimoto check is WARN at the default
  threshold of `1.0` and FAIL when you lower it. The dashboard colours
  each count by tier and adds a findings-by-severity table, and text
  report findings are tagged `[FAIL]` or `[WARN]`.
- **Counting (#4):** findings are counted as distinct problems and
  affected rows, not checks fired. One duplicate seen through raw SMILES,
  canonical SMILES and InChIKey is one finding.
- **Identifier conflicts are classified (#27):** each "one identifier,
  several structures" finding has a `conflict_type`:
  - `salt_form`
  - `stereo`
  - `tautomer`
  - `different_compound`, with the highest Tanimoto similarity

  Only `different_compound` is FAIL. "One compound, several identifiers"
  now groups by the parent compound's connectivity, so it's found across
  salt forms and stereo annotations.
- **Mixtures are split (#26):** multi-fragment entries are classified as
  salts/solvates, multi-component, homodimers or other mixtures. Fragments
  identical to the parent are parent copies, so a 2:1 salt stays a salt.
- SMILES column aliases have a preference order: a file with several alias
  columns uses the most preferred, instead of erroring.
- The report's `generated_at` timestamp is in UTC.
- The dashboard's identifier table is one row per check and one column per
  identifier column.
- Processing failures after a successful parse log their traceback at
  DEBUG (`-v`) and record the exception type (#7).
- Python 3.13, 3.14 and 3.15 are supported. CI tests every supported
  Python with the newest dependencies, and Python 3.12 with the declared
  minimum versions (#12).
- Tooling (#7):
  - ruff enforces lint rules
  - `__version__` comes from the package metadata
  - in `environment.yml`, `rich` moved to the runtime dependencies (pinned
    at 15.0.0)

### Fixed

- **Salts drawn as the neutral acid or as hydrates** (`.Cl`, `.Br`, TFA,
  mesylate, esylate, isethionate, triflate, sulfosalicylate, maleate, `.O`,
  …) are now recognised as salts and solvates
  rather than only as mixtures (#20). Counterions are matched whether
  drawn ionic or neutral. The parent is never a large counterion such as
  pamoate.
- **Legitimate small molecules are no longer flagged as suspicious
  fragments** (#3). Methanol, ethanol, acetonitrile, DCM and other common
  small molecules are allowlisted. Bare `C`, `C#N`, lone ions and lone
  heteroatoms are still flagged.
- **A SMILES with a control character inside it** (e.g. `"CC\nO"`) is now
  an unparseable `control_characters` failure. RDKit used to silently read
  it as `CC`.
- **Temporary files are cleaned up:** files downloaded from a plain URL are
  deleted once loaded, and partial downloads are deleted on failure (#7).
- **Identifier column names show in the dashboard again:** they're
  escaped, so rich no longer swallows `identifier[col]`.

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
