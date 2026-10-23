"""TOML configuration loading and validation for challenge-data-checker."""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from challenge_data_checker.io_utils import DataSource
from challenge_data_checker.quality_filters import DEFAULT_MULTI_COMPONENT_MIN_HEAVY_ATOMS


class ConfigError(ValueError):
    """Raised when the TOML configuration is missing or malformed."""


VALID_REPORT_FORMATS = {"json", "txt"}


@dataclass(frozen=True)
class PathsConfig:
    """Train/test data sources and report destination.

    Attributes
    ----------
    train_files : Sequence[DataSource]
        Training data sources: paths to data files, URLs to remote data
        files (HuggingFace Hub or plain HTTP(S)),
        and/or already-loaded DataFrames (the latter only when built
        directly via the Python API, never from a TOML config).
    test_files : Sequence[DataSource]
        Test data sources, in the same shapes as ``train_files``.
    report_output : str | Path | None
        Path the audit report will be written to, in the format given by
        ``settings.report_format``, or ``None`` to skip writing a report
        file (Python API only; a TOML config always requires this).
    """

    train_files: Sequence[DataSource]
    test_files: Sequence[DataSource]
    report_output: str | Path | None


@dataclass(frozen=True)
class ColumnsConfig:
    """Column configuration declared in the ``[columns]`` section of the config.

    Attributes
    ----------
    smiles_column : str
        Name of the column containing SMILES strings.
    identifier_columns : list[str]
        Names of columns holding compound identifiers.
    """

    smiles_column: str
    identifier_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SettingsConfig:
    """Optional settings declared in the ``[settings]`` section of the config.

    Attributes
    ----------
    tautomer_standardisation : bool
        Whether to canonicalise tautomers before comparing molecules.
    strip_salts : bool
        Whether to also compare molecules by their salt- and
        solvent-stripped, neutralised parent compound, so a salt in one
        pool matches its free base (or another salt form) in the other.
    identifier_case_insensitive : bool
        Whether identifier values that differ only in letter case count as
        the same identifier. Off by default; case-only differences are
        then still reported as near-duplicate identifiers (WARN).
    multi_component_min_heavy_atoms : int
        Minimum heavy atoms for an unrecognised fragment to count as a
        separate compound in the ``multi_component`` quality flag.
    max_train_test_similarity : float
        Tanimoto similarity threshold (0.0-1.0) at or above which a
        train/test pair is flagged as possible leakage.
    fp_radius : int
        Morgan fingerprint radius.
    fp_n_bits : int
        Morgan fingerprint bit-vector length.
    report_format : str
        Output format for the audit report, ``"json"`` or ``"txt"``. If
        not set explicitly in the config, it is inferred from
        ``paths.report_output``'s file extension: ``.txt`` implies
        ``"txt"``, anything else implies ``"json"``.
    """

    tautomer_standardisation: bool = False
    strip_salts: bool = True
    identifier_case_insensitive: bool = False
    multi_component_min_heavy_atoms: int = DEFAULT_MULTI_COMPONENT_MIN_HEAVY_ATOMS
    max_train_test_similarity: float = 1.0
    fp_radius: int = 2
    fp_n_bits: int = 2048
    report_format: str = "json"


@dataclass(frozen=True)
class Config:
    """Fully parsed and validated challenge-data-checker configuration.

    Attributes
    ----------
    paths : PathsConfig
        Input/output file paths.
    columns : ColumnsConfig
        SMILES and identifier column configuration.
    settings : SettingsConfig
        Optional chemistry/matching settings.
    config_path : Path
        Path the configuration was loaded from.
    """

    paths: PathsConfig
    columns: ColumnsConfig
    settings: SettingsConfig
    config_path: Path


def infer_report_format(explicit: str | None, report_output: str | Path | None) -> str:
    """Resolve the report format, inferring it from the output path when unset.

    Parameters
    ----------
    explicit : str | None
        The user-specified ``report_format``, or ``None`` if not set.
    report_output : str | Path | None
        The path the report will be written to, or ``None`` if it won't
        be written to disk.

    Returns
    -------
    str
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

    Parameters
    ----------
    section : dict
        The parsed TOML section.
    key : str
        The key to look up within ``section``.
    section_name : str
        Name of the section, used only in error messages.

    Returns
    -------
    object
        The value stored under ``key``.

    Raises
    ------
    ConfigError
        If ``key`` is not present in ``section``.
    """
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in [{section_name}] section.")
    return section[key]


def _require_str_list(section: dict, key: str, section_name: str) -> list[str]:
    """Fetch and validate a required, non-empty list-of-strings key.

    Parameters
    ----------
    section : dict
        The parsed TOML section.
    key : str
        The key to look up within ``section``.
    section_name : str
        Name of the section, used only in error messages.

    Returns
    -------
    list[str]
        The list of strings stored under ``key``.

    Raises
    ------
    ConfigError
        If the key is missing, not a list of strings, or empty.
    """
    value = _require(section, key, section_name)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"'{key}' in [{section_name}] must be a list of strings.")
    if not value:
        raise ConfigError(f"'{key}' in [{section_name}] must not be empty.")
    return value


def _load_toml(config_path: Path) -> dict:
    """Read a TOML config file and check it has the required sections.

    Parameters
    ----------
    config_path : Path
        Path to the TOML configuration file.

    Returns
    -------
    dict
        The parsed TOML document.

    Raises
    ------
    ConfigError
        If the file is missing, not valid TOML, or lacks a ``[paths]`` or
        ``[columns]`` section.
    """
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
    return raw


def parse_columns(columns_raw: Mapping[str, object]) -> ColumnsConfig:
    """Parse and validate the ``[columns]`` section.

    Parameters
    ----------
    columns_raw : Mapping[str, object]
        The ``[columns]`` section (or the equivalent CLI flags).

    Returns
    -------
    ColumnsConfig
        The validated column configuration.

    Raises
    ------
    ConfigError
        If ``smiles_column`` is missing or empty, or ``identifier_columns``
        isn't a list of strings.
    """
    smiles_column = _require(dict(columns_raw), "smiles_column", "columns")
    if not isinstance(smiles_column, str) or not smiles_column:
        raise ConfigError("'smiles_column' in [columns] must be a non-empty string.")
    identifier_columns_raw = columns_raw.get("identifier_columns", [])
    if not isinstance(identifier_columns_raw, list) or not all(
        isinstance(v, str) for v in identifier_columns_raw
    ):
        raise ConfigError("'identifier_columns' in [columns] must be a list of strings.")
    return ColumnsConfig(
        smiles_column=smiles_column, identifier_columns=list(identifier_columns_raw)
    )


# Settings that only apply to the train/test audit (they configure the
# train-vs-test Tanimoto comparison), so a check config may not set them.
AUDIT_ONLY_SETTINGS = frozenset({"max_train_test_similarity", "fp_radius", "fp_n_bits"})


def parse_settings(
    settings_raw: Mapping[str, object], report_output: str | Path | None
) -> SettingsConfig:
    """Parse and validate the ``[settings]`` section.

    Used for TOML configs and CLI flags alike, so both are validated the
    same way.

    Parameters
    ----------
    settings_raw : Mapping[str, object]
        The ``[settings]`` section (or the equivalent CLI flags); missing
        keys take their defaults.
    report_output : str | Path | None
        Where the report will be written, used to infer ``report_format``
        when it isn't set.

    Returns
    -------
    SettingsConfig
        The validated settings.

    Raises
    ------
    ConfigError
        If a setting is out of range or ``report_format`` is unknown.
    """
    defaults = SettingsConfig()

    def get(key: str) -> object:
        return settings_raw.get(key, getattr(defaults, key))

    explicit_format = settings_raw.get("report_format")
    settings = SettingsConfig(
        tautomer_standardisation=bool(get("tautomer_standardisation")),
        strip_salts=bool(get("strip_salts")),
        identifier_case_insensitive=bool(get("identifier_case_insensitive")),
        multi_component_min_heavy_atoms=int(str(get("multi_component_min_heavy_atoms"))),
        max_train_test_similarity=float(str(get("max_train_test_similarity"))),
        fp_radius=int(str(get("fp_radius"))),
        fp_n_bits=int(str(get("fp_n_bits"))),
        report_format=infer_report_format(
            None if explicit_format is None else str(explicit_format), report_output
        ),
    )

    if not 0.0 <= settings.max_train_test_similarity <= 1.0:
        raise ConfigError(
            "'max_train_test_similarity' must be between 0.0 and 1.0, "
            f"got {settings.max_train_test_similarity}."
        )
    if settings.multi_component_min_heavy_atoms < 1:
        raise ConfigError(
            "'multi_component_min_heavy_atoms' must be >= 1, "
            f"got {settings.multi_component_min_heavy_atoms}."
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
    return settings


def load_config(config_path: str | Path) -> Config:
    """Load and validate a challenge-data-checker (train/test audit) TOML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the TOML configuration file.

    Returns
    -------
    Config
        The parsed and validated configuration.

    Raises
    ------
    ConfigError
        If the file is missing, not valid TOML, missing a required
        section or key, is a single-dataset ``check`` config, or contains
        an out-of-range setting.
    """
    config_path = Path(config_path)
    raw = _load_toml(config_path)
    paths_raw = raw["paths"]
    if "files" in paths_raw:
        raise ConfigError(
            f"{config_path} is a single-dataset check config ([paths] files = ...). "
            f"Run it with: challenge-data-checker check --config {config_path}"
        )

    paths = PathsConfig(
        train_files=_require_str_list(paths_raw, "train_files", "paths"),
        test_files=_require_str_list(paths_raw, "test_files", "paths"),
        report_output=str(_require(paths_raw, "report_output", "paths")),
    )
    return Config(
        paths=paths,
        columns=parse_columns(raw["columns"]),
        settings=parse_settings(raw.get("settings", {}), paths.report_output),
        config_path=config_path,
    )


@dataclass(frozen=True)
class CheckPathsConfig:
    """Data sources and report destination for a single-dataset check.

    Attributes
    ----------
    files : Sequence[DataSource]
        The dataset's sources (paths, URLs and/or DataFrames), pooled into
        one dataset.
    report_output : str | Path | None
        Where to write the report, or ``None`` to skip writing a file.
    """

    files: Sequence[DataSource]
    report_output: str | Path | None


@dataclass(frozen=True)
class CheckConfig:
    """Fully parsed and validated configuration for a single-dataset check.

    Attributes
    ----------
    paths : CheckPathsConfig
        Input/output file paths.
    columns : ColumnsConfig
        SMILES and identifier column configuration.
    settings : SettingsConfig
        Chemistry/matching settings (the audit-only Tanimoto settings keep
        their defaults and aren't used).
    config_path : Path | None
        Path the configuration was loaded from, or ``None`` if it was built
        from CLI flags or the Python API.
    """

    paths: CheckPathsConfig
    columns: ColumnsConfig
    settings: SettingsConfig
    config_path: Path | None


def reject_audit_only_settings(settings_raw: Mapping[str, object]) -> None:
    """Refuse train/test-only settings in a single-dataset check config.

    Parameters
    ----------
    settings_raw : Mapping[str, object]
        The ``[settings]`` section of a check config.

    Returns
    -------
    None

    Raises
    ------
    ConfigError
        If any of ``AUDIT_ONLY_SETTINGS`` is set.
    """
    audit_only = sorted(AUDIT_ONLY_SETTINGS & set(settings_raw))
    if audit_only:
        raise ConfigError(
            f"Setting(s) {audit_only} only apply to the train/test audit, not to a check."
        )


def load_check_config(config_path: str | Path) -> CheckConfig:
    """Load and validate a single-dataset ``check`` TOML configuration file.

    The schema matches the audit's, except that ``[paths]`` takes ``files``
    instead of ``train_files``/``test_files``, ``report_output`` is
    optional, and the train/test-only settings aren't allowed.

    Parameters
    ----------
    config_path : str | Path
        Path to the TOML configuration file.

    Returns
    -------
    CheckConfig
        The parsed and validated configuration.

    Raises
    ------
    ConfigError
        If the file is missing, not valid TOML, missing a required section
        or key, is a train/test audit config, or contains an out-of-range
        or audit-only setting.
    """
    config_path = Path(config_path)
    raw = _load_toml(config_path)
    paths_raw = raw["paths"]
    if "train_files" in paths_raw or "test_files" in paths_raw:
        raise ConfigError(
            f"{config_path} is a train/test audit config ([paths] train_files/test_files). "
            f"Run it with: challenge-data-checker {config_path}"
        )
    settings_raw = raw.get("settings", {})
    reject_audit_only_settings(settings_raw)

    report_output = paths_raw.get("report_output")
    paths = CheckPathsConfig(
        files=_require_str_list(paths_raw, "files", "paths"),
        report_output=None if report_output is None else str(report_output),
    )
    return CheckConfig(
        paths=paths,
        columns=parse_columns(raw["columns"]),
        settings=parse_settings(settings_raw, paths.report_output),
        config_path=config_path,
    )
