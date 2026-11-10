"""Command-line entry point for challenge-data-checker.

Three commands share one executable:

- ``challenge-data-checker config.toml`` audits a train/test split.
- ``challenge-data-checker check ...`` checks a single dataset, configured
  either with ``--config check.toml`` or entirely with flags.
- ``challenge-data-checker public-search ...`` searches a public database
  for a blinded test set, to find out whether it is already public.
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
    PublicSearchConfig,
    PublicSearchPathsConfig,
    load_check_config,
    load_config,
    load_public_search_config,
    parse_columns,
    parse_public_search,
    parse_settings,
)
from challenge_data_checker.core import run_audit, run_check, run_public_search
from challenge_data_checker.io_utils import ColumnResolutionError
from challenge_data_checker.report_types import (
    CheckReport,
    PublicSearchReport,
    Report,
    Status,
)

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
            "without a split, run `challenge-data-checker check --help`. To find "
            "out whether a blinded test set is already public, run "
            "`challenge-data-checker public-search --help`."
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
    if argv and argv[0] == "public-search":
        return public_search_main(argv[1:])

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        report = run(args.config)
    except _RUN_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _exit_code(report["summary"]["status"], args.strict)


def build_public_search_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the public-source search.

    Every data/setting flag maps 1:1 to a key of the public-search TOML
    config, so either form can express any search.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    parser = _ArgumentParser(
        prog="challenge-data-checker public-search",
        description=(
            "Search a public database (a local ChEMBL release) for a blinded test "
            "set, to find out whether its measurements are already public. A test "
            "compound merely being in ChEMBL is not leakage, so name the challenge "
            "endpoint with at least one of --assay-chembl-ids, --target-chembl-ids "
            "or --target-names. Configure it with --config, or with the data and "
            "setting flags, not both."
        ),
        epilog=_EXIT_CODES_HELP,
    )
    parser.add_argument("--config", help="Path to a public-search TOML configuration file.")
    data = parser.add_argument_group("data and settings (instead of --config)")
    data.add_argument(
        "--test-files", nargs="+", help="Test-set data files or URLs (paths.test_files)."
    )
    data.add_argument("--chembl-sqlite", help="Path to a local ChEMBL SQLite release.")
    data.add_argument(
        "--assay-chembl-ids",
        nargs="+",
        help="ChEMBL assay IDs measuring the challenge endpoint (a match is blocking).",
    )
    data.add_argument(
        "--target-chembl-ids",
        nargs="+",
        help="ChEMBL target IDs for the challenge endpoint (a match is blocking).",
    )
    data.add_argument(
        "--target-names",
        nargs="+",
        help="Target names or gene symbols to resolve, e.g. CYP3A4 (a match is flagged).",
    )
    data.add_argument(
        "--standard-types",
        nargs="+",
        help="Allowed standard_type values for a --target-names match, e.g. IC50 Ki.",
    )
    data.add_argument(
        "--target-types",
        nargs="+",
        help=(
            "ChEMBL target types a --target-names entry may resolve to "
            '(default: "SINGLE PROTEIN").'
        ),
    )
    data.add_argument(
        "--match-connectivity",
        action="store_const",
        const=True,
        help="Also match on InChIKey connectivity block (catches differing stereo; slower).",
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
        help="Also match a compound by its salt-stripped parent (default: on).",
    )
    data.add_argument("--report-output", help="Path to write the full report to (optional).")
    data.add_argument("--report-format", help='"json" or "txt" (default: from --report-output).')
    _add_run_options(parser)
    return parser


# public-search flags that set data or settings, and so can't be combined
# with --config.
_PUBLIC_SEARCH_DATA_FLAGS = (
    "test_files",
    "chembl_sqlite",
    "assay_chembl_ids",
    "target_chembl_ids",
    "target_names",
    "standard_types",
    "target_types",
    "match_connectivity",
    "smiles_column",
    "identifier_columns",
    "tautomer_standardisation",
    "strip_salts",
    "report_output",
    "report_format",
)


def public_search_config_from_args(args: argparse.Namespace) -> PublicSearchConfig:
    """Build a public-search configuration from parsed ``public-search`` flags.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed ``public-search`` arguments, without ``--config``.

    Returns
    -------
    PublicSearchConfig
        The validated configuration, built with the same parsers as a TOML
        config.

    Raises
    ------
    ConfigError
        If ``--test-files`` or ``--chembl-sqlite`` is missing, no endpoint
        matcher is given, or a setting is invalid.
    """
    if not args.test_files:
        raise ConfigError("public-search needs either --config or --test-files.")
    if not args.chembl_sqlite:
        raise ConfigError("public-search needs --chembl-sqlite (a local ChEMBL release).")
    settings_raw = {
        key: getattr(args, key)
        for key in (
            "tautomer_standardisation",
            "strip_salts",
            "report_format",
        )
        if getattr(args, key) is not None
    }
    return PublicSearchConfig(
        paths=PublicSearchPathsConfig(test_files=args.test_files, report_output=args.report_output),
        columns=parse_columns(
            {
                "smiles_column": args.smiles_column or "SMILES",
                "identifier_columns": args.identifier_columns or [],
            }
        ),
        settings=parse_settings(settings_raw, args.report_output),
        public_search=parse_public_search(
            {
                "chembl_sqlite": args.chembl_sqlite,
                "match_connectivity": bool(args.match_connectivity),
                "assay_chembl_ids": args.assay_chembl_ids or [],
                "target_chembl_ids": args.target_chembl_ids or [],
                "target_names": args.target_names or [],
                "standard_types": args.standard_types or [],
                **({"target_types": args.target_types} if args.target_types else {}),
            }
        ),
        config_path=None,
    )


def run_public_search_from_args(args: argparse.Namespace) -> PublicSearchReport:
    """Run a public-source search from parsed ``public-search`` arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed ``public-search`` arguments.

    Returns
    -------
    PublicSearchReport
        The JSON-serialisable public-search report.
    """
    config = (
        load_public_search_config(args.config)
        if args.config
        else public_search_config_from_args(args)
    )
    return run_public_search(config)


def public_search_main(argv: list[str]) -> int:
    """Parse ``public-search`` arguments and run the public-source search.

    Parameters
    ----------
    argv : list[str]
        Arguments after ``public-search``.

    Returns
    -------
    int
        The process exit code (see ``main``).
    """
    parser = build_public_search_arg_parser()
    args = parser.parse_args(argv)
    if args.config:
        clashing = [f for f in _PUBLIC_SEARCH_DATA_FLAGS if getattr(args, f) is not None]
        if clashing:
            flags = ", ".join("--" + f.replace("_", "-") for f in clashing)
            print(
                f"Error: --config can't be combined with data or setting flags ({flags}).",
                file=sys.stderr,
            )
            return EXIT_ERROR
    _configure_logging(args.verbose)

    try:
        report = run_public_search_from_args(args)
    except _RUN_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _exit_code(report["summary"]["status"], args.strict)


if __name__ == "__main__":
    sys.exit(main())
