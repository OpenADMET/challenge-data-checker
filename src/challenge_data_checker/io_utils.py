"""File loading, pooling, and fuzzy column resolution."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from challenge_data_checker.remote import fetch_remote_source, is_remote_source

logger = logging.getLogger(__name__)

# A single train/test data source: a file path, a URL to a remote file
# (HuggingFace Hub or plain HTTP(S)), or an already-loaded DataFrame (e.g.
# passed in directly from a Jupyter notebook via the Python API).
type DataSource = str | Path | pd.DataFrame

# Known alternate names for a SMILES column, used only when neither an exact
# nor a case-insensitive match to the configured name is found. Ordered by
# preference: if a file has several, the earliest one in this tuple wins.
SMILES_COLUMN_ALIASES = (
    "smiles",
    "canonical_smiles",
    "standardized_smiles",
    "standardised_smiles",
    "std_smiles",
    "isomeric_smiles",
    "cxsmiles",
    "smi",
    "structure",
    "molecule",
    "mol",
)

# Supported data file extensions, matched case-insensitively against the end
# of the file name. Delimited text may also be gzip-compressed (pandas infers
# the compression from the ".gz" extension).
SUPPORTED_FILE_EXTENSIONS = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet")

SOURCE_FILE_COL = "__source_file__"
ROW_INDEX_COL = "__row_index__"
RESOLVED_SMILES_COL = "__smiles__"


class ColumnResolutionError(ValueError):
    """Raised when a required column cannot be unambiguously resolved."""


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a .csv, .tsv (optionally gzipped) or .parquet file into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to a data file with one of ``SUPPORTED_FILE_EXTENSIONS``.

    Returns
    -------
    pd.DataFrame
        The loaded data as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` has an unsupported file extension.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input data file not found: {path}")
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    if name.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t")
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file format '{''.join(path.suffixes)}' for {path}. "
        f"Use one of: {', '.join(SUPPORTED_FILE_EXTENSIONS)}."
    )


def describe_source(source: DataSource, index: int) -> str:
    """Build a human-readable label identifying a train/test data source.

    Used both as the ``source_file`` tag on every row pooled from ``source``
    and (via ``report.build_report``) in the report's echoed configuration,
    so a finding's location and the config section refer to the source the
    same way.

    Parameters
    ----------
    source : DataSource
        A file path, or an in-memory DataFrame.
    index : int
        The source's position in its train/test list, used to distinguish
        multiple in-memory DataFrames from one another.

    Returns
    -------
    str
        The path as a string, or ``"<in-memory dataframe #N (R rows)>"``
        for a DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        return f"<in-memory dataframe #{index} ({len(source)} rows)>"
    return str(source)


def resolve_smiles_column(
    columns: list[str], configured_name: str, source_label: str = "<unknown source>"
) -> str:
    """Resolve the SMILES column name, falling back to case-insensitive/alias matching.

    Resolution order:
        1. Exact match to ``configured_name``.
        2. Case-insensitive match to ``configured_name``.
        3. A column whose lowercased name is in ``SMILES_COLUMN_ALIASES``,
           preferring the alias listed earliest when several are present.

    Falling back to an alias match (step 3) is logged as a warning, since it
    means the configured column name wasn't found at all and a different
    column is being audited in its place. The warning also names any other
    alias columns that were passed over.

    Parameters
    ----------
    columns : list[str]
        Column names available in the loaded DataFrame.
    configured_name : str
        The ``smiles_column`` value from the config.
    source_label : str
        Label identifying the source, used only in the warning message
        logged on alias fallback.

    Returns
    -------
    str
        The resolved column name, exactly as it appears in ``columns``.

    Raises
    ------
    ColumnResolutionError
        If no column can be resolved as the SMILES column, or the chosen
        name matches more than one column case-insensitively.
    """
    if configured_name in columns:
        return configured_name

    lower_map: dict[str, list[str]] = {}
    for col in columns:
        lower_map.setdefault(col.lower(), []).append(col)

    configured_lower = configured_name.lower()
    if configured_lower in lower_map:
        matches = lower_map[configured_lower]
        if len(matches) == 1:
            return matches[0]
        raise ColumnResolutionError(
            f"Could not uniquely resolve SMILES column '{configured_name}': multiple "
            f"columns match case-insensitively: {matches}."
        )

    alias_candidates = [col for col in columns if col.lower() in SMILES_COLUMN_ALIASES]
    for alias in SMILES_COLUMN_ALIASES:
        matches = lower_map.get(alias, [])
        if not matches:
            continue
        if len(matches) > 1:
            raise ColumnResolutionError(
                f"Could not uniquely resolve SMILES column '{configured_name}': "
                f"multiple columns match the alias '{alias}' case-insensitively: {matches}. "
                "Please set 'smiles_column' explicitly to the exact column name."
            )
        resolved = matches[0]
        passed_over = [col for col in alias_candidates if col != resolved]
        logger.warning(
            "SMILES column '%s' not found in %s; falling back to alias match '%s'%s. "
            "Set 'smiles_column' explicitly to silence this warning if that's intended.",
            configured_name,
            source_label,
            resolved,
            f" (preferred over {passed_over})" if passed_over else "",
        )
        return resolved

    raise ColumnResolutionError(
        f"Could not find SMILES column '{configured_name}' (or a case-insensitive/alias "
        f"match) among available columns: {columns}."
    )


def check_identifier_columns(
    columns: list[str], identifier_columns: list[str], file_path: str | Path
) -> list[str]:
    """Return the subset of identifier columns present in ``columns``.

    Logs a warning (does not raise) for any configured identifier column
    missing from this particular file.

    Parameters
    ----------
    columns : list[str]
        Column names available in the loaded DataFrame.
    identifier_columns : list[str]
        Identifier column names from the config.
    file_path : str | Path
        Path of the file being checked, used only in the warning message.

    Returns
    -------
    list[str]
        The subset of ``identifier_columns`` that are present in
        ``columns``.
    """
    present = []
    for col in identifier_columns:
        if col in columns:
            present.append(col)
        else:
            logger.warning(
                "Identifier column '%s' not found in %s; skipping this identifier "
                "check for this file.",
                col,
                file_path,
            )
    return present


def load_pool(
    sources: Sequence[DataSource], smiles_column: str, identifier_columns: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Load and concatenate a list of data sources into a single pool.

    Each source's SMILES column is independently resolved (exact/case-insensitive/
    alias fuzzy match) and renamed to a fixed internal name so downstream code
    doesn't need to know the original column name. Each row is tagged with its
    source label (see ``describe_source``) and original row index within that
    source. Identifier columns missing from a given source are logged as a
    warning and simply left absent (pandas fills them with NaN after
    concatenation).

    Parameters
    ----------
    sources : Sequence[DataSource]
        Paths to local data files, URLs to remote data files (HuggingFace
        Hub or plain HTTP(S)), and/or already-loaded DataFrames to load (if
        needed) and pool together. See ``SUPPORTED_FILE_EXTENSIONS``.
    smiles_column : str
        The configured SMILES column name to resolve in each source.
    identifier_columns : list[str]
        Identifier column names from the config.

    Returns
    -------
    pd.DataFrame
        The pooled DataFrame.
    list[str]
        The identifier columns present in at least one of the sources
        (used for later checks).
    dict[str, str]
        A mapping of source label to the SMILES column actually resolved
        for that source (so the report can record it even when it
        silently differs from ``smiles_column``).
    """
    frames = []
    identifier_columns_seen: list[str] = []
    resolved_smiles_columns: dict[str, str] = {}
    for i, source in enumerate(sources):
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, str) and is_remote_source(source):
            with fetch_remote_source(source) as local_path:
                df = load_table(local_path)
        else:
            df = load_table(source)
        label = describe_source(source, i)
        resolved_smiles_col = resolve_smiles_column(list(df.columns), smiles_column, label)
        resolved_smiles_columns[label] = resolved_smiles_col
        df = df.rename(columns={resolved_smiles_col: RESOLVED_SMILES_COL})
        present = check_identifier_columns(list(df.columns), identifier_columns, label)
        for col in present:
            if col not in identifier_columns_seen:
                identifier_columns_seen.append(col)
        df[SOURCE_FILE_COL] = label
        df[ROW_INDEX_COL] = df.index
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, identifier_columns_seen, resolved_smiles_columns
