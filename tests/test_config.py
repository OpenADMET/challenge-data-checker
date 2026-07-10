import pytest

from challenge_data_checker.config import ConfigError, load_config

BASE_TOML = """
[paths]
train_files = ["train.csv"]
test_files = ["test.csv"]
report_output = "report.json"

[columns]
smiles_column = "SMILES"
identifier_columns = ["compound_id"]
"""


def write_toml(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    return path


def test_load_config_applies_defaults(tmp_path):
    config = load_config(write_toml(tmp_path, BASE_TOML))
    assert config.paths.train_files == ["train.csv"]
    assert config.paths.test_files == ["test.csv"]
    assert config.paths.report_output == "report.json"
    assert config.columns.smiles_column == "SMILES"
    assert config.columns.identifier_columns == ["compound_id"]
    assert config.settings.tautomer_standardisation is False
    assert config.settings.max_train_test_similarity == 1.0
    assert config.settings.fp_radius == 2
    assert config.settings.fp_n_bits == 2048
    assert config.settings.report_format == "json"


def test_load_config_overrides_settings(tmp_path):
    content = BASE_TOML + """
[settings]
tautomer_standardisation = true
max_train_test_similarity = 0.85
fp_radius = 3
fp_n_bits = 1024
report_format = "txt"
"""
    config = load_config(write_toml(tmp_path, content))
    assert config.settings.tautomer_standardisation is True
    assert config.settings.max_train_test_similarity == 0.85
    assert config.settings.fp_radius == 3
    assert config.settings.fp_n_bits == 1024
    assert config.settings.report_format == "txt"


def test_report_format_is_case_insensitive(tmp_path):
    content = BASE_TOML + """
[settings]
report_format = "TXT"
"""
    config = load_config(write_toml(tmp_path, content))
    assert config.settings.report_format == "txt"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.toml")


def test_missing_paths_section_raises(tmp_path):
    content = """
[columns]
smiles_column = "SMILES"
"""
    with pytest.raises(ConfigError, match=r"\[paths\]"):
        load_config(write_toml(tmp_path, content))


def test_missing_columns_section_raises(tmp_path):
    content = """
[paths]
train_files = ["train.csv"]
test_files = ["test.csv"]
report_output = "report.json"
"""
    with pytest.raises(ConfigError, match=r"\[columns\]"):
        load_config(write_toml(tmp_path, content))


def test_empty_train_files_raises(tmp_path):
    content = """
[paths]
train_files = []
test_files = ["test.csv"]
report_output = "report.json"

[columns]
smiles_column = "SMILES"
"""
    with pytest.raises(ConfigError, match="must not be empty"):
        load_config(write_toml(tmp_path, content))


def test_identifier_columns_defaults_to_empty(tmp_path):
    content = """
[paths]
train_files = ["train.csv"]
test_files = ["test.csv"]
report_output = "report.json"

[columns]
smiles_column = "SMILES"
"""
    config = load_config(write_toml(tmp_path, content))
    assert config.columns.identifier_columns == []


def test_invalid_similarity_threshold_raises(tmp_path):
    content = BASE_TOML + """
[settings]
max_train_test_similarity = 1.5
"""
    with pytest.raises(ConfigError, match="max_train_test_similarity"):
        load_config(write_toml(tmp_path, content))


def test_invalid_fp_n_bits_raises(tmp_path):
    content = BASE_TOML + """
[settings]
fp_n_bits = 0
"""
    with pytest.raises(ConfigError, match="fp_n_bits"):
        load_config(write_toml(tmp_path, content))


def test_invalid_report_format_raises(tmp_path):
    content = BASE_TOML + """
[settings]
report_format = "yaml"
"""
    with pytest.raises(ConfigError, match="report_format"):
        load_config(write_toml(tmp_path, content))
