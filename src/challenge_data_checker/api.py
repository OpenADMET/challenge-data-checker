"""Direct Python API for running an audit without a TOML config file.

Intended for interactive use (e.g. a Jupyter notebook): pass train/test data
as file paths and/or already-loaded pandas DataFrames, mixed freely, and get
the same report dict the CLI produces, without needing files on disk.
"""

from pathlib import Path
from typing import Sequence

import pandas as pd

from challenge_data_checker.config import (
    ColumnsConfig,
    Config,
    PathsConfig,
    SettingsConfig,
    infer_report_format,
)
from challenge_data_checker.core import run_audit
from challenge_data_checker.io_utils import DataSource

# A single train/test argument: one source, or a list of sources (paths and/or
# DataFrames freely mixed).
type DataInput = DataSource | Sequence[DataSource]


def _as_source_list(data: DataInput) -> list[DataSource]:
    """Normalize a single source or a sequence of sources into a list.

    Parameters
    ----------
    data
        A single path/DataFrame, or a sequence of them.

    Returns
    -------
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
    max_train_test_similarity: float = 1.0,
    fp_radius: int = 2,
    fp_n_bits: int = 2048,
    report_output: str | Path | None = None,
    report_format: str | None = None,
    print_report: bool = True,
) -> dict:
    """Run a challenge-data-checker audit directly from Python.

    A code-first alternative to the TOML-config CLI (``challenge-data-checker
    config.toml``): pass every setting as a keyword argument instead of
    writing a config file, and pass train/test data as file paths and/or
    DataFrames already in memory.

    Parameters
    ----------
    train
        Training data: a single .csv/.parquet path or DataFrame, or a list
        mixing either.
    test
        Test data, in the same shapes as ``train``.
    smiles_column
        Name of the SMILES column, resolved per source (exact match, then
        case-insensitive, then a common alias like ``smi`` or ``structure``).
    identifier_columns
        Optional compound identifier column names to cross-check for
        leakage/namespace issues.
    tautomer_standardisation
        Whether to canonicalise tautomers before comparing molecules.
    max_train_test_similarity
        Tanimoto similarity threshold (0.0-1.0, inclusive) at or above
        which a train/test pair is flagged as possible leakage.
    fp_radius
        Morgan fingerprint radius.
    fp_n_bits
        Morgan fingerprint bit-vector length.
    report_output
        Optional path to also write the full report to. If omitted, the
        report is only returned, not written to disk.
    report_format
        ``"json"`` or ``"txt"``. If omitted, it's inferred from
        ``report_output``'s extension (``.txt`` -> ``"txt"``, otherwise
        ``"json"``); has no effect if ``report_output`` is omitted.
    print_report
        Whether to print the colour-coded summary dashboard to stdout.

    Returns
    -------
    The full report dict (same shape as the CLI's report file/the JSON
    written by ``report.save_report``), regardless of ``report_output``/
    ``print_report``.
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
            max_train_test_similarity=max_train_test_similarity,
            fp_radius=fp_radius,
            fp_n_bits=fp_n_bits,
            report_format=infer_report_format(report_format, report_output),
        ),
        config_path=Path("<python-api>"),
    )

    return run_audit(config, save=report_output is not None, print_report=print_report)
