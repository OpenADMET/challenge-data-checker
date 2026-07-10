"""Command-line entry point for challenge-data-checker."""

import argparse
import logging
import sys

from challenge_data_checker.chemistry import MoleculeProcessor, process_pool
from challenge_data_checker.config import ConfigError, load_config
from challenge_data_checker.io_utils import ColumnResolutionError, load_pool
from challenge_data_checker.report import build_report, print_summary, save_report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured argument parser.

    """
    parser = argparse.ArgumentParser(
        prog="challenge-data-checker",
        description=(
            "Audit ML benchmarking train/test splits for chemical data leakage, "
            "duplicates, and data quality issues."
        ),
    )
    parser.add_argument("config", help="Path to the TOML configuration file.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging."
    )
    return parser


def run(config_path: str) -> dict:
    """Run the full audit for a given config path.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        The JSON-serialisable report dict, as returned by
        ``report.build_report``.

    """
    config = load_config(config_path)

    train_df, train_identifier_cols = load_pool(
        config.paths.train_files, config.columns.smiles_column, config.columns.identifier_columns
    )
    test_df, test_identifier_cols = load_pool(
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

    report = build_report(config, train_records, test_records, identifier_columns)
    save_report(report, config.paths.report_output, config.settings.report_format)
    print_summary(report, config.paths.report_output)
    return report


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the audit as a command-line program.

    Args:
        argv: Command-line arguments (excluding the program name), or
            ``None`` to use ``sys.argv``.

    Returns:
        The process exit code: ``0`` on success, ``1`` if a configuration or
        input error occurred.

    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        run(args.config)
    except (ConfigError, ColumnResolutionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
