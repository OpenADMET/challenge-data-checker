"""Shared execution pipeline: load pools, process molecules, build/save/print a report.

Used by both the CLI (``cli.run`` / ``cli.run_check``) and the direct Python
API (``api.audit`` / ``api.check``), which differ only in how they build the
configuration passed in.
"""

from challenge_data_checker.chemistry import MoleculeProcessor, process_pool
from challenge_data_checker.config import CheckConfig, Config
from challenge_data_checker.io_utils import load_pool
from challenge_data_checker.report import (
    build_check_report,
    build_report,
    print_summary,
    save_report,
)
from challenge_data_checker.report_types import CheckReport, PerPool, Report


def run_audit(config: Config, *, save: bool = True, print_report: bool = True) -> Report:
    """Run the full audit pipeline for an already-built configuration.

    Parameters
    ----------
    config : Config
        The configuration to audit with, however it was built (from
        a TOML file, or directly via ``api.audit``).
    save : bool
        Whether to write the report to ``config.paths.report_output``.
        Has no effect if ``report_output`` is ``None``.
    print_report : bool
        Whether to print the stdout summary dashboard.

    Returns
    -------
    Report
        The JSON-serialisable report dict, as returned by
        ``report.build_report``.
    """
    train_df, train_identifier_cols, train_smiles_cols = load_pool(
        config.paths.train_files, config.columns.smiles_column, config.columns.identifier_columns
    )
    test_df, test_identifier_cols, test_smiles_cols = load_pool(
        config.paths.test_files, config.columns.smiles_column, config.columns.identifier_columns
    )
    # A column may be entirely absent from one pool's files (e.g. only present in
    # train). Each pool is only processed with the identifier columns it actually
    # has; the union is used afterwards for cross-pool checks and reporting, since
    # MoleculeRecord.identifiers.get(col) gracefully returns None for a pool that
    # never saw that column.
    identifier_columns = [
        col
        for col in config.columns.identifier_columns
        if col in train_identifier_cols or col in test_identifier_cols
    ]

    processor = MoleculeProcessor(config.settings)
    train_records = process_pool(train_df, "train", train_identifier_cols, processor)
    test_records = process_pool(test_df, "test", test_identifier_cols, processor)

    resolved_smiles_columns: PerPool[dict[str, str]] = {
        "train": train_smiles_cols,
        "test": test_smiles_cols,
    }
    report = build_report(
        config, train_records, test_records, identifier_columns, resolved_smiles_columns
    )

    if save and config.paths.report_output is not None:
        save_report(report, config.paths.report_output, config.settings.report_format)
    if print_report:
        print_summary(report, config.paths.report_output)
    return report


def run_check(config: CheckConfig, *, save: bool = True, print_report: bool = True) -> CheckReport:
    """Run the single-dataset check pipeline for an already-built configuration.

    Parameters
    ----------
    config : CheckConfig
        The check configuration to run with, however it was built (from a
        TOML file, CLI flags, or ``api.check``).
    save : bool
        Whether to write the report to ``config.paths.report_output``.
        Has no effect if ``report_output`` is ``None``.
    print_report : bool
        Whether to print the stdout summary dashboard.

    Returns
    -------
    CheckReport
        The JSON-serialisable check report, as returned by
        ``report.build_check_report``.
    """
    df, identifier_columns, smiles_columns = load_pool(
        config.paths.files, config.columns.smiles_column, config.columns.identifier_columns
    )
    processor = MoleculeProcessor(config.settings)
    records = process_pool(df, "dataset", identifier_columns, processor)
    report = build_check_report(config, records, identifier_columns, smiles_columns)

    if save and config.paths.report_output is not None:
        save_report(report, config.paths.report_output, config.settings.report_format)
    if print_report:
        print_summary(report, config.paths.report_output)
    return report
