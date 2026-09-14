"""Shared execution pipeline: load pools, process molecules, build/save/print a report.

Used by both the TOML-config CLI (``cli.run``) and the direct Python API
(``api.audit``), which differ only in how they build the ``Config`` passed in.
"""

from challenge_data_checker.chemistry import MoleculeProcessor, process_pool
from challenge_data_checker.config import Config
from challenge_data_checker.io_utils import load_pool
from challenge_data_checker.report import build_report, print_summary, save_report


def run_audit(config: Config, *, save: bool = True, print_report: bool = True) -> dict:
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
    dict
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

    resolved_smiles_columns = {"train": train_smiles_cols, "test": test_smiles_cols}
    report = build_report(
        config, train_records, test_records, identifier_columns, resolved_smiles_columns
    )

    if save and config.paths.report_output is not None:
        save_report(report, config.paths.report_output, config.settings.report_format)
    if print_report:
        print_summary(report, config.paths.report_output)
    return report
