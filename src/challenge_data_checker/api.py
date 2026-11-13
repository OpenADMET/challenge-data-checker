"""Direct Python API for running an audit without a TOML config file.

Intended for interactive use (e.g. a Jupyter notebook): pass train/test data
as file paths and/or already-loaded pandas DataFrames, mixed freely, and get
the same report dict the CLI produces, without needing files on disk.

``audit`` compares a train/test split, ``check`` vets a single dataset, and
``search_public`` asks whether a blinded test set is already public.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from challenge_data_checker.config import (
    CheckConfig,
    CheckPathsConfig,
    ColumnsConfig,
    Config,
    PathsConfig,
    PublicSearchConfig,
    PublicSearchPathsConfig,
    SettingsConfig,
    infer_report_format,
    parse_public_search,
    parse_settings,
)
from challenge_data_checker.core import run_audit, run_check, run_public_search
from challenge_data_checker.io_utils import DataSource
from challenge_data_checker.quality_filters import DEFAULT_MULTI_COMPONENT_MIN_HEAVY_ATOMS
from challenge_data_checker.report_types import CheckReport, PublicSearchReport, Report

# A single train/test argument: one source, or a list of sources (paths and/or
# DataFrames freely mixed).
type DataInput = DataSource | Sequence[DataSource]


def _as_source_list(data: DataInput) -> list[DataSource]:
    """Normalize a single source or a sequence of sources into a list.

    Parameters
    ----------
    data : DataInput
        A single path/DataFrame, or a sequence of them.

    Returns
    -------
    list[DataSource]
        ``data`` as a list, wrapping a lone path/DataFrame in a list.
    """
    if isinstance(data, (str, Path, pd.DataFrame)):
        return [data]
    return list(data)


def audit(
    train: DataInput,
    test: DataInput,
    smiles_column: str = "SMILES",
    identifier_columns: Sequence[str] | None = None,
    tautomer_standardisation: bool = False,
    strip_salts: bool = True,
    identifier_case_insensitive: bool = False,
    multi_component_min_heavy_atoms: int = DEFAULT_MULTI_COMPONENT_MIN_HEAVY_ATOMS,
    max_train_test_similarity: float = 1.0,
    fp_radius: int = 2,
    fp_n_bits: int = 2048,
    report_output: str | Path | None = None,
    report_format: str | None = None,
    print_report: bool = True,
) -> Report:
    """Run a challenge-data-checker audit directly from Python.

    A code-first alternative to the TOML-config CLI (``challenge-data-checker
    config.toml``): pass every setting as a keyword argument instead of
    writing a config file, and pass train/test data as file paths and/or
    DataFrames already in memory.

    Parameters
    ----------
    train : DataInput
        Training data: a single data file path (.csv, .tsv, optionally
        gzipped, or .parquet), URL or DataFrame, or a list mixing these.
    test : DataInput
        Test data, in the same shapes as ``train``.
    smiles_column : str
        Name of the SMILES column, resolved per source (exact match, then
        case-insensitive, then a common alias like ``smi`` or ``structure``).
    identifier_columns : Sequence[str] | None
        Optional compound identifier column names to cross-check for
        leakage/namespace issues.
    tautomer_standardisation : bool
        Whether to canonicalise tautomers before comparing molecules.
    strip_salts : bool
        Whether to also compare molecules by their salt- and
        solvent-stripped parent compound, so a salt in one pool matches
        its free base (or another salt form) in the other.
    identifier_case_insensitive : bool
        Whether identifier values that differ only in letter case count as
        the same identifier.
    multi_component_min_heavy_atoms : int
        Minimum heavy atoms for an unrecognised fragment to count as a
        separate compound in the ``multi_component`` quality flag.
    max_train_test_similarity : float
        Tanimoto similarity threshold (0.0-1.0, inclusive) at or above
        which a train/test pair is flagged as possible leakage.
    fp_radius : int
        Morgan fingerprint radius.
    fp_n_bits : int
        Morgan fingerprint bit-vector length.
    report_output : str | Path | None
        Optional path to also write the full report to. If omitted, the
        report is only returned, not written to disk.
    report_format : str | None
        ``"json"`` or ``"txt"``. If omitted, it's inferred from
        ``report_output``'s extension (``.txt`` -> ``"txt"``, otherwise
        ``"json"``); has no effect if ``report_output`` is omitted.
    print_report : bool
        Whether to print the colour-coded summary dashboard to stdout.

    Returns
    -------
    Report
        The full report dict (same shape as the CLI's report file/the
        JSON written by ``report.save_report``), regardless of
        ``report_output``/``print_report``. ``report["summary"]["status"]``
        is the overall result: ``"PASS"``, ``"WARN"`` (flagged findings
        only) or ``"FAIL"`` (blocking findings).
    """
    config = Config(
        paths=PathsConfig(
            train_files=_as_source_list(train),
            test_files=_as_source_list(test),
            report_output=report_output,
        ),
        columns=ColumnsConfig(
            smiles_column=smiles_column,
            identifier_columns=list(identifier_columns) if identifier_columns else [],
        ),
        settings=SettingsConfig(
            tautomer_standardisation=tautomer_standardisation,
            strip_salts=strip_salts,
            identifier_case_insensitive=identifier_case_insensitive,
            multi_component_min_heavy_atoms=multi_component_min_heavy_atoms,
            max_train_test_similarity=max_train_test_similarity,
            fp_radius=fp_radius,
            fp_n_bits=fp_n_bits,
            report_format=infer_report_format(report_format, report_output),
        ),
        config_path=Path("<python-api>"),
    )

    return run_audit(config, save=report_output is not None, print_report=print_report)


def check(
    data: DataInput,
    smiles_column: str = "SMILES",
    identifier_columns: Sequence[str] | None = None,
    tautomer_standardisation: bool = False,
    strip_salts: bool = True,
    identifier_case_insensitive: bool = False,
    multi_component_min_heavy_atoms: int = DEFAULT_MULTI_COMPONENT_MIN_HEAVY_ATOMS,
    report_output: str | Path | None = None,
    report_format: str | None = None,
    print_report: bool = True,
) -> CheckReport:
    """Check a single dataset, with no train/test split, directly from Python.

    Runs every check that doesn't need a split (unparseable SMILES,
    duplicates, identifier consistency, quality flags), for example on a
    dataset before splitting, publishing or accepting it.

    Parameters
    ----------
    data : DataInput
        The dataset: a single data file path, URL or DataFrame, or a list
        mixing these, pooled into one dataset.
    smiles_column : str
        Name of the SMILES column, resolved per source as in ``audit``.
    identifier_columns : Sequence[str] | None
        Optional compound identifier column names to check.
    tautomer_standardisation : bool
        Whether to canonicalise tautomers before comparing molecules.
    strip_salts : bool
        Whether a salt and its free base (or another salt form) count as
        duplicates of the same compound.
    identifier_case_insensitive : bool
        Whether identifier values that differ only in letter case count as
        the same identifier.
    multi_component_min_heavy_atoms : int
        Minimum heavy atoms for an unrecognised fragment to count as a
        separate compound in the ``multi_component`` quality flag.
    report_output : str | Path | None
        Optional path to also write the full report to.
    report_format : str | None
        ``"json"`` or ``"txt"``; inferred from ``report_output`` if
        omitted.
    print_report : bool
        Whether to print the colour-coded summary dashboard to stdout.

    Returns
    -------
    CheckReport
        The full check report. ``report["summary"]["status"]`` is the
        overall result: ``"PASS"``, ``"WARN"`` or ``"FAIL"``.
    """
    settings_raw: dict[str, object] = {
        "tautomer_standardisation": tautomer_standardisation,
        "strip_salts": strip_salts,
        "identifier_case_insensitive": identifier_case_insensitive,
        "multi_component_min_heavy_atoms": multi_component_min_heavy_atoms,
    }
    if report_format is not None:
        settings_raw["report_format"] = report_format
    config = CheckConfig(
        paths=CheckPathsConfig(files=_as_source_list(data), report_output=report_output),
        columns=ColumnsConfig(
            smiles_column=smiles_column,
            identifier_columns=list(identifier_columns) if identifier_columns else [],
        ),
        settings=parse_settings(settings_raw, report_output),
        config_path=None,
    )
    return run_check(config, save=report_output is not None, print_report=print_report)


def search_public(
    test: DataInput,
    chembl_sqlite: str | Path,
    assay_chembl_ids: Sequence[str] | None = None,
    target_chembl_ids: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    standard_types: Sequence[str] | None = None,
    target_types: Sequence[str] | None = None,
    match_connectivity: bool = True,
    smiles_column: str = "SMILES",
    identifier_columns: Sequence[str] | None = None,
    tautomer_standardisation: bool = False,
    strip_salts: bool = True,
    report_output: str | Path | None = None,
    report_format: str | None = None,
    print_report: bool = True,
) -> PublicSearchReport:
    """Search a public database for a blinded test set, directly from Python.

    Answers a different question from ``audit``: not "does train overlap
    test?" but "is this test set already public?". A test compound simply
    being in ChEMBL is not leakage, so at least one of
    ``assay_chembl_ids``, ``target_chembl_ids`` or ``target_names`` must be
    given to say what the challenge endpoint is.

    The search runs against a **local** ChEMBL release rather than the REST
    API, so a blinded test set never leaves the machine and the report can
    cite a release number.

    Parameters
    ----------
    test : DataInput
        The (blinded) test set: a single data file path, URL or DataFrame,
        or a list mixing these.
    chembl_sqlite : str | Path
        Path to a local ChEMBL SQLite release.
    assay_chembl_ids : Sequence[str] | None
        ChEMBL assay IDs that measure the challenge endpoint. A match is
        blocking.
    target_chembl_ids : Sequence[str] | None
        ChEMBL target IDs for the challenge endpoint. A match is blocking.
    target_names : Sequence[str] | None
        Target names or gene symbols (e.g. ``CYP3A4``) resolved against
        the release. A match is flagged, not blocking.
    standard_types : Sequence[str] | None
        Allowed ``standard_type`` values for a ``target_names`` match
        (e.g. ``IC50``, ``Ki``), compared case-insensitively. Empty
        allows any.
    target_types : Sequence[str] | None
        ChEMBL ``target_type`` values a ``target_names`` entry may resolve
        to. Defaults to single proteins only.
    match_connectivity : bool
        Whether to also match on the InChIKey connectivity block, catching
        differing or undefined stereochemistry. On by default; such a match
        is flagged, not blocking, since it may be a deliberate stereo
        split.
    smiles_column : str
        Name of the SMILES column, resolved per source as in ``audit``.
    identifier_columns : Sequence[str] | None
        Optional identifier column names, echoed into the report. The
        public lookup is structural, so these aren't matched on.
    tautomer_standardisation : bool
        Whether to canonicalise tautomers before comparing molecules.
    strip_salts : bool
        Whether to also look compounds up by their salt-stripped parent,
        so a test free base matches a ChEMBL salt form.
    report_output : str | Path | None
        Optional path to also write the full report to.
    report_format : str | None
        ``"json"`` or ``"txt"``; inferred from ``report_output`` if
        omitted.
    print_report : bool
        Whether to print the colour-coded summary dashboard to stdout.

    Returns
    -------
    PublicSearchReport
        The full report. ``report["summary"]["status"]`` is the overall
        result: ``"PASS"``, ``"WARN"`` or ``"FAIL"``.
    """
    config = PublicSearchConfig(
        paths=PublicSearchPathsConfig(
            test_files=_as_source_list(test), report_output=report_output
        ),
        columns=ColumnsConfig(
            smiles_column=smiles_column,
            identifier_columns=list(identifier_columns) if identifier_columns else [],
        ),
        settings=SettingsConfig(
            tautomer_standardisation=tautomer_standardisation,
            strip_salts=strip_salts,
            report_format=infer_report_format(report_format, report_output),
        ),
        public_search=parse_public_search(
            {
                "chembl_sqlite": str(chembl_sqlite),
                "match_connectivity": match_connectivity,
                "assay_chembl_ids": list(assay_chembl_ids) if assay_chembl_ids else [],
                "target_chembl_ids": list(target_chembl_ids) if target_chembl_ids else [],
                "target_names": list(target_names) if target_names else [],
                "standard_types": list(standard_types) if standard_types else [],
                **({"target_types": list(target_types)} if target_types is not None else {}),
            }
        ),
        config_path=None,
    )
    return run_public_search(config, save=report_output is not None, print_report=print_report)
