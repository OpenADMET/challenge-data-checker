"""Command-line entry point for challenge-data-checker."""

import argparse
import logging
import sys

from challenge_data_checker.config import ConfigError, load_config
from challenge_data_checker.core import run_audit
from challenge_data_checker.io_utils import ColumnResolutionError


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
    return run_audit(config)


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
