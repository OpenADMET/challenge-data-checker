"""Command-line entry point for challenge-data-checker."""

import argparse
import logging
import sys

from challenge_data_checker.config import ConfigError, load_config
from challenge_data_checker.core import run_audit
from challenge_data_checker.io_utils import ColumnResolutionError
from challenge_data_checker.report_types import Report

# Process exit codes. 1 means the audit couldn't run at all; 2 means it ran
# and found something that should stop the split being released.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FINDINGS = 2


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="challenge-data-checker",
        description=(
            "Audit ML benchmarking train/test splits for chemical data leakage, "
            "duplicates, and data quality issues."
        ),
        epilog=(
            "Exit codes: 0 = PASS (or WARN without --strict); 1 = configuration or "
            "input error; 2 = blocking findings (FAIL), or flagged findings (WARN) "
            "with --strict."
        ),
    )
    parser.add_argument("config", help="Path to the TOML configuration file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also exit with code 2 when there are only flagged (WARN) findings.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging."
    )
    return parser


def run(config_path: str) -> Report:
    """Run the full audit for a given config path.

    Parameters
    ----------
    config_path : str
        Path to the TOML configuration file.

    Returns
    -------
    Report
        The JSON-serialisable report dict, as returned by
        ``report.build_report``.
    """
    config = load_config(config_path)
    return run_audit(config)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the audit as a command-line program.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (excluding the program name), or ``None``
        to use ``sys.argv``.

    Returns
    -------
    int
        The process exit code: ``0`` if the audit passed (or only has
        flagged findings, without ``--strict``), ``1`` if a configuration
        or input error stopped it running, ``2`` if it found blocking
        findings (or flagged findings, with ``--strict``).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        report = run(args.config)
    except (ConfigError, ColumnResolutionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    status = report["summary"]["status"]
    if status == "FAIL" or (status == "WARN" and args.strict):
        return EXIT_FINDINGS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
