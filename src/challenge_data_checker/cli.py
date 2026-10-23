"""Command-line entry point for challenge-data-checker.

Two commands share one executable:

- ``challenge-data-checker config.toml`` audits a train/test split.
- ``challenge-data-checker check ...`` checks a single dataset, configured
  either with ``--config check.toml`` or entirely with flags.
"""

import argparse
import logging
import sys
from typing import NoReturn

from challenge_data_checker.config import (
    AUDIT_ONLY_SETTINGS,
    CheckConfig,
    CheckPathsConfig,
    ConfigError,
    load_check_config,
    load_config,
    parse_columns,
    parse_settings,
)
from challenge_data_checker.core import run_audit, run_check
from challenge_data_checker.io_utils import ColumnResolutionError
from challenge_data_checker.report_types import CheckReport, Report, Status

# Process exit codes. 1 means the audit couldn't run at all (including a
# command-line usage error); 2 means it ran and found something that should
# stop the data being released.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FINDINGS = 2

_EXIT_CODES_HELP = (
    "Exit codes: 0 = PASS (or WARN without --strict); 1 = configuration, input or "
    "usage error; 2 = blocking findings (FAIL), or flagged findings (WARN) with --strict."
)

# Errors that mean the audit or check couldn't run, reported as exit code 1.
_RUN_ERRORS = (ConfigError, ColumnResolutionError, FileNotFoundError, ValueError)


class _ArgumentParser(argparse.ArgumentParser):
    """An argument parser whose usage errors exit with ``EXIT_ERROR``.

    argparse exits with 2 by default, which here means "blocking findings".
    """

    def error(self, message: str) -> NoReturn:
        """Print the usage and a usage error, then exit with ``EXIT_ERROR``.

        Parameters
        ----------
        message : str
            The error message.

        Returns
        -------
        NoReturn
            Never returns; exits the process.
        """
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by the audit and the check (not data/settings).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add them to.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also exit with code 2 when there are only flagged (WARN) findings.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the train/test audit.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    parser = _ArgumentParser(
        prog="challenge-data-checker",
        description=(
            "Audit ML benchmarking train/test splits for chemical data leakage, "
            "duplicates, and data quality issues. To check a single dataset "
            "without a split, run `challenge-data-checker check --help`."
        ),
        epilog=_EXIT_CODES_HELP,
    )
    parser.add_argument("config", help="Path to the TOML configuration file.")
    _add_run_options(parser)
    return parser


def build_check_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the single-dataset check.

    Every data/setting flag maps 1:1 to a key of the check TOML config, so
    either form can express any check.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    parser = _ArgumentParser(
        prog="challenge-data-checker check",
        description=(
            "Check a single dataset (no train/test split) for unparseable SMILES, "
            "duplicates, identifier problems and data quality issues. Configure it "
            "with --config, or with the data and setting flags, not both."
        ),
        epilog=_EXIT_CODES_HELP,
    )
    parser.add_argument("--config", help="Path to a check TOML configuration file.")
    data = parser.add_argument_group("data and settings (instead of --config)")
    data.add_argument(
        "--files", nargs="+", help="Data files or URLs, pooled into one dataset (paths.files)."
    )
    data.add_argument("--smiles-column", help="SMILES column name (default: SMILES).")
    data.add_argument("--identifier-columns", nargs="+", help="Identifier column names.")
    data.add_argument(
        "--tautomer-standardisation",
        action="store_const",
        const=True,
        help="Canonicalise tautomers before comparing molecules.",
    )
    data.add_argument(
        "--strip-salts",
        action=argparse.BooleanOptionalAction,
        help="Treat salt forms of one compound as duplicates (default: on).",
    )
    data.add_argument(
        "--identifier-case-insensitive",
        action="store_const",
        const=True,
        help="Treat identifier values differing only in case as the same.",
    )
    data.add_argument(
        "--multi-component-min-heavy-atoms",
        type=int,
        help="Minimum heavy atoms for a fragment to count as a separate compound (default: 6).",
    )
    data.add_argument("--report-output", help="Path to write the full report to (optional).")
    data.add_argument("--report-format", help='"json" or "txt" (default: from --report-output).')
    _add_run_options(parser)
    return parser


# Check flags that set data or settings, and so can't be combined with --config.
_CHECK_DATA_FLAGS = (
    "files",
    "smiles_column",
    "identifier_columns",
    "tautomer_standardisation",
    "strip_salts",
    "identifier_case_insensitive",
    "multi_component_min_heavy_atoms",
    "report_output",
    "report_format",
)


def check_config_from_args(args: argparse.Namespace) -> CheckConfig:
    """Build a check configuration from parsed ``check`` flags.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed ``check`` arguments, without ``--config``.

    Returns
    -------
    CheckConfig
        The validated configuration, built with the same parsers as a TOML
        config.

    Raises
    ------
    ConfigError
        If ``--files`` is missing or a setting is invalid.
    """
    if not args.files:
        raise ConfigError("check needs either --config or --files.")
    settings_raw = {
        key: getattr(args, key)
        for key in (
            "tautomer_standardisation",
            "strip_salts",
            "identifier_case_insensitive",
            "multi_component_min_heavy_atoms",
            "report_format",
        )
        if getattr(args, key) is not None
    }
    assert not AUDIT_ONLY_SETTINGS & set(settings_raw)  # no such flags exist
    return CheckConfig(
        paths=CheckPathsConfig(files=args.files, report_output=args.report_output),
        columns=parse_columns(
            {
                "smiles_column": args.smiles_column or "SMILES",
                "identifier_columns": args.identifier_columns or [],
            }
        ),
        settings=parse_settings(settings_raw, args.report_output),
        config_path=None,
    )


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


def run_check_from_args(args: argparse.Namespace) -> CheckReport:
    """Run a single-dataset check from parsed ``check`` arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed ``check`` arguments.

    Returns
    -------
    CheckReport
        The JSON-serialisable check report.
    """
    config = load_check_config(args.config) if args.config else check_config_from_args(args)
    return run_check(config)


def _exit_code(status: Status, strict: bool) -> int:
    """Map an audit or check result to the process exit code.

    Parameters
    ----------
    status : Status
        The report's overall status.
    strict : bool
        Whether flagged (WARN) findings should also fail.

    Returns
    -------
    int
        ``EXIT_FINDINGS`` for FAIL, or WARN when ``strict``; otherwise
        ``EXIT_OK``.
    """
    if status == "FAIL" or (status == "WARN" and strict):
        return EXIT_FINDINGS
    return EXIT_OK


def _configure_logging(verbose: bool) -> None:
    """Set up logging for a command-line run.

    Parameters
    ----------
    verbose : bool
        Whether to log at DEBUG rather than WARNING level.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def check_main(argv: list[str]) -> int:
    """Parse ``check`` arguments and run the single-dataset check.

    Parameters
    ----------
    argv : list[str]
        Arguments after ``check``.

    Returns
    -------
    int
        The process exit code (see ``main``).
    """
    parser = build_check_arg_parser()
    args = parser.parse_args(argv)
    if args.config:
        clashing = [f for f in _CHECK_DATA_FLAGS if getattr(args, f) is not None]
        if clashing:
            flags = ", ".join("--" + f.replace("_", "-") for f in clashing)
            print(
                f"Error: --config can't be combined with data or setting flags ({flags}).",
                file=sys.stderr,
            )
            return EXIT_ERROR
    _configure_logging(args.verbose)

    try:
        report = run_check_from_args(args)
    except _RUN_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _exit_code(report["summary"]["status"], args.strict)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the audit (or, after ``check``, the single-dataset check).

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (excluding the program name), or ``None``
        to use ``sys.argv``.

    Returns
    -------
    int
        The process exit code: ``0`` if the run passed (or only has
        flagged findings, without ``--strict``), ``1`` if a configuration,
        input or usage error stopped it running, ``2`` if it found
        blocking findings (or flagged findings, with ``--strict``).
    """
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "check":
        return check_main(argv[1:])

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        report = run(args.config)
    except _RUN_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _exit_code(report["summary"]["status"], args.strict)


if __name__ == "__main__":
    sys.exit(main())
