"""TOML configuration loading and validation for challenge-data-checker."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from challenge_data_checker.io_utils import DataSource


class ConfigError(ValueError):
    """Raised when the TOML configuration is missing or malformed."""


VALID_REPORT_FORMATS = {"json", "txt"}


@dataclass(frozen=True)
class PathsConfig:
    """Train/test data sources and report destination.

    Attributes:
        train_files: Training data sources: paths to .csv/.parquet files
            and/or already-loaded DataFrames (the latter only when built
            directly via the Python API, never from a TOML config).
        test_files: Test data sources, in the same shapes as ``train_files``.
        report_output: Path the audit report will be written to, in the
            format given by ``settings.report_format``, or ``None`` to skip
            writing a report file (Python API only; a TOML config always
            requires this).

    """

    train_files: Sequence[DataSource]
    test_files: Sequence[DataSource]
    report_output: str | Path | None


@dataclass(frozen=True)
class ColumnsConfig:
    """Column configuration declared in the ``[columns]`` section of the config.

    Attributes:
        smiles_column: Name of the column containing SMILES strings.
        identifier_columns: Names of columns holding compound identifiers.

    """

    smiles_column: str
    identifier_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SettingsConfig:
    """Optional settings declared in the ``[settings]`` section of the config.

    Attributes:
        tautomer_standardisation: Whether to canonicalise tautomers before
            comparing molecules.
        max_train_test_similarity: Tanimoto similarity threshold (0.0-1.0) at
            or above which a train/test pair is flagged as possible leakage.
        fp_radius: Morgan fingerprint radius.
        fp_n_bits: Morgan fingerprint bit-vector length.
        report_format: Output format for the audit report, ``"json"`` or
            ``"txt"``. If not set explicitly in the config, it is inferred
            from ``paths.report_output``'s file extension: ``.txt`` implies
            ``"txt"``, anything else implies ``"json"``.

    """

    tautomer_standardisation: bool = False
    max_train_test_similarity: float = 1.0
    fp_radius: int = 2
    fp_n_bits: int = 2048
    report_format: str = "json"


@dataclass(frozen=True)
class Config:
    """Fully parsed and validated challenge-data-checker configuration.

    Attributes:
        paths: Input/output file paths.
        columns: SMILES and identifier column configuration.
        settings: Optional chemistry/matching settings.
        config_path: Path the configuration was loaded from.

    """

    paths: PathsConfig
    columns: ColumnsConfig
    settings: SettingsConfig
    config_path: Path


def infer_report_format(explicit: str | None, report_output: str | Path | None) -> str:
    """Resolve the report format, inferring it from the output path when unset.

    Args:
        explicit: The user-specified ``report_format``, or ``None`` if not
            set.
        report_output: The path the report will be written to, or ``None``
            if it won't be written to disk.

    Returns:
        ``explicit`` (lowercased) if given; otherwise ``"txt"`` if
        ``report_output`` ends in ``.txt``, otherwise ``"json"``.

    """
    if explicit is not None:
        return explicit.lower()
    if report_output is not None and Path(report_output).suffix.lower() == ".txt":
        return "txt"
    return "json"


def _require(section: dict, key: str, section_name: str) -> object:
    """Fetch a required key from a parsed TOML section.

    Args:
        section: The parsed TOML section.
        key: The key to look up within ``section``.
        section_name: Name of the section, used only in error messages.

    Returns:
        The value stored under ``key``.

    Raises:
        ConfigError: If ``key`` is not present in ``section``.

    """
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in [{section_name}] section.")
    return section[key]


def _require_str_list(section: dict, key: str, section_name: str) -> list[str]:
    """Fetch and validate a required, non-empty list-of-strings key.

    Args:
        section: The parsed TOML section.
        key: The key to look up within ``section``.
        section_name: Name of the section, used only in error messages.

    Returns:
        The list of strings stored under ``key``.

    Raises:
        ConfigError: If the key is missing, not a list of strings, or empty.

    """
    value = _require(section, key, section_name)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"'{key}' in [{section_name}] must be a list of strings.")
    if not value:
        raise ConfigError(f"'{key}' in [{section_name}] must not be empty.")
    return value


def load_config(config_path: str | Path) -> Config:
    """Load and validate a challenge-data-checker TOML configuration file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        The parsed and validated configuration.

    Raises:
        ConfigError: If the file is missing, not valid TOML, missing a
            required section or key, or contains an out-of-range setting.

    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Failed to parse TOML config {config_path}: {exc}") from exc

    if "paths" not in raw:
        raise ConfigError("Config is missing required [paths] section.")
    if "columns" not in raw:
        raise ConfigError("Config is missing required [columns] section.")

    paths_raw = raw["paths"]
    columns_raw = raw["columns"]
    settings_raw = raw.get("settings", {})

    paths = PathsConfig(
        train_files=_require_str_list(paths_raw, "train_files", "paths"),
        test_files=_require_str_list(paths_raw, "test_files", "paths"),
        report_output=str(_require(paths_raw, "report_output", "paths")),
    )

    smiles_column = _require(columns_raw, "smiles_column", "columns")
    if not isinstance(smiles_column, str) or not smiles_column:
        raise ConfigError("'smiles_column' in [columns] must be a non-empty string.")

    identifier_columns_raw = columns_raw.get("identifier_columns", [])
    if not isinstance(identifier_columns_raw, list) or not all(
        isinstance(v, str) for v in identifier_columns_raw
    ):
        raise ConfigError("'identifier_columns' in [columns] must be a list of strings.")

    columns = ColumnsConfig(
        smiles_column=smiles_column,
        identifier_columns=list(identifier_columns_raw),
    )

    explicit_format = (
        str(settings_raw["report_format"]) if "report_format" in settings_raw else None
    )
    report_format = infer_report_format(explicit_format, paths.report_output)

    settings = SettingsConfig(
        tautomer_standardisation=bool(settings_raw.get("tautomer_standardisation", False)),
        max_train_test_similarity=float(settings_raw.get("max_train_test_similarity", 1.0)),
        fp_radius=int(settings_raw.get("fp_radius", 2)),
        fp_n_bits=int(settings_raw.get("fp_n_bits", 2048)),
        report_format=report_format,
    )

    if not 0.0 <= settings.max_train_test_similarity <= 1.0:
        raise ConfigError(
            "'max_train_test_similarity' must be between 0.0 and 1.0, "
            f"got {settings.max_train_test_similarity}."
        )
    if settings.fp_radius < 0:
        raise ConfigError(f"'fp_radius' must be >= 0, got {settings.fp_radius}.")
    if settings.fp_n_bits <= 0:
        raise ConfigError(f"'fp_n_bits' must be > 0, got {settings.fp_n_bits}.")
    if settings.report_format not in VALID_REPORT_FORMATS:
        raise ConfigError(
            f"'report_format' must be one of {sorted(VALID_REPORT_FORMATS)}, "
            f"got '{settings.report_format}'."
        )

    return Config(paths=paths, columns=columns, settings=settings, config_path=config_path)
