"""File loading, pooling, and fuzzy column resolution."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Known alternate names for a SMILES column, used only when neither an exact
# nor a case-insensitive match to the configured name is found.
SMILES_COLUMN_ALIASES = {
    "smiles",
    "smi",
    "structure",
    "canonical_smiles",
    "molecule",
    "mol",
}

SOURCE_FILE_COL = "__source_file__"
ROW_INDEX_COL = "__row_index__"
RESOLVED_SMILES_COL = "__smiles__"


class ColumnResolutionError(ValueError):
    """Raised when a required column cannot be unambiguously resolved."""


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a .csv or .parquet file into a DataFrame.

    Args:
        path: Path to a .csv or .parquet data file.

    Returns:
        The loaded data as a DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` has an unsupported file extension.

    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format '{suffix}' for {path}. Use .csv or .parquet.")


def resolve_smiles_column(columns: list[str], configured_name: str) -> str:
    """Resolve the SMILES column name, falling back to case-insensitive/alias matching.

    Resolution order:
        1. Exact match to ``configured_name``.
        2. Case-insensitive match to ``configured_name``.
        3. A unique column whose lowercased name is in ``SMILES_COLUMN_ALIASES``.

    Args:
        columns: Column names available in the loaded DataFrame.
        configured_name: The ``smiles_column`` value from the config.

    Returns:
        The resolved column name, exactly as it appears in ``columns``.

    Raises:
        ColumnResolutionError: If no single column can be unambiguously
            resolved as the SMILES column.

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
    if len(alias_candidates) == 1:
        return alias_candidates[0]

    if len(alias_candidates) > 1:
        raise ColumnResolutionError(
            f"Could not uniquely resolve SMILES column '{configured_name}': "
            f"multiple candidate columns found via fuzzy matching: {alias_candidates}. "
            "Please set 'smiles_column' explicitly to the exact column name."
        )
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

    Args:
        columns: Column names available in the loaded DataFrame.
        identifier_columns: Identifier column names from the config.
        file_path: Path of the file being checked, used only in the warning
            message.

    Returns:
        The subset of ``identifier_columns`` that are present in ``columns``.

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
    files: list[str], smiles_column: str, identifier_columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Load and concatenate a list of data files into a single pool.

    Each file's SMILES column is independently resolved (exact/case-insensitive/
    alias fuzzy match) and renamed to a fixed internal name so downstream code
    doesn't need to know the original column name. Each row is tagged with its
    source file and original row index within that file. Identifier columns
    missing from a given file are logged as a warning and simply left absent
    (pandas fills them with NaN after concatenation).

    Args:
        files: Paths to the .csv/.parquet files to load and pool together.
        smiles_column: The configured SMILES column name to resolve in each
            file.
        identifier_columns: Identifier column names from the config.

    Returns:
        A tuple of the pooled DataFrame and the list of identifier columns
        present in at least one of the files (used for later checks).

    """
    frames = []
    identifier_columns_seen: list[str] = []
    for file in files:
        df = load_table(file)
        df = df.copy()
        resolved_smiles_col = resolve_smiles_column(list(df.columns), smiles_column)
        df = df.rename(columns={resolved_smiles_col: RESOLVED_SMILES_COL})
        present = check_identifier_columns(list(df.columns), identifier_columns, file)
        for col in present:
            if col not in identifier_columns_seen:
                identifier_columns_seen.append(col)
        df[SOURCE_FILE_COL] = str(file)
        df[ROW_INDEX_COL] = df.index
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, identifier_columns_seen
