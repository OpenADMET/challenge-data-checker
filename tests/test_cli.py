import json

import pandas as pd
import pytest

from challenge_data_checker.cli import main, run


def write_config(tmp_path, report_filename="report.json", **overrides):
    settings_lines = "\n".join(f"{k} = {v}" for k, v in overrides.items())
    content = f"""
[paths]
train_files = ["{tmp_path / "train.csv"}"]
test_files = ["{tmp_path / "test.csv"}"]
report_output = "{tmp_path / report_filename}"

[columns]
smiles_column = "SMILES"
identifier_columns = ["compound_id", "external_reg_no"]

[settings]
{settings_lines}
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(content)
    return config_path


@pytest.fixture
def mock_dataset(tmp_path):
    # Intentional leaks/edge cases:
    # - CCO appears in both train (C1) and test (T1) -> exact + identifier leakage
    # - benzene appears as C2 in train and T2 in test with a *different* id -> namespace mismatch
    # - CCO duplicated within train itself (C1 / C1b) -> internal duplicate
    # - a salt, a mixture, and suspicious tiny fragments for quality filters
    # - test.csv uses "structure" as a fuzzy alias for the configured "SMILES" column
    # - external_reg_no is only present in train.csv, missing entirely from test.csv
    train_df = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCO", "c1ccccc1", "CC(=O)O.CC(=O)O", "[Na+].[Cl-]", "C#N"],
            "compound_id": ["C1", "C1b", "C2", "C3", "C4", "C5"],
            "external_reg_no": ["R1", "R1", "R2", "R3", "R4", "R5"],
        }
    )
    test_df = pd.DataFrame(
        {
            "structure": ["CCO", "c1ccccc1", "CCN"],
            "compound_id": ["T1", "T2", "SAME_ID"],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return tmp_path


def test_run_end_to_end_report_contents(mock_dataset):
    config_path = write_config(mock_dataset)
    report = run(str(config_path))

    assert report["summary"]["total_train_rows"] == 6
    assert report["summary"]["total_test_rows"] == 3
    assert report["summary"]["total_unparseable"] == 0

    # exact leakage: CCO and benzene both appear in train and test
    assert len(report["train_test_leakage"]["exact"]["inchikey"]) == 2

    # external_reg_no exists only in train.csv (missing from test.csv, which logs a
    # warning); it must still be tracked for train-only checks without crashing.
    assert "external_reg_no" in report["config"]["identifier_columns"]
    assert "compound_id" in report["config"]["identifier_columns"]
    assert report["internal_duplicates"]["train"]["identifiers"]["external_reg_no"]

    # internal duplicate: CCO appears twice within train alone
    assert len(report["internal_duplicates"]["train"]["inchikey"]) == 1

    # quality filters:
    # - mixtures: the diacetic acid dimer and NaCl (both contain a ".") -> 2
    # - salts/metals: NaCl (metal) and the bare cyanide (matches the counterion
    #   list used to catch stripped-metal-complex artefacts) -> 2
    # - suspicious tiny fragments: both CCO rows + the bare cyanide (<=3 heavy atoms) -> 3
    quality_train = report["quality_flags"]["train"]
    assert len(quality_train["mixtures"]) == 2
    assert len(quality_train["salts_or_metal_complexes"]) == 2
    assert len(quality_train["suspicious_fragments"]) == 3

    # report file written to disk
    report_output = mock_dataset / "report.json"
    assert report_output.is_file()
    on_disk = json.loads(report_output.read_text())
    assert on_disk["summary"]["total_train_rows"] == 6


def test_main_prints_dashboard(mock_dataset, capsys):
    config_path = write_config(mock_dataset)
    exit_code = main([str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "challenge-data-checker audit summary" in captured.out
    assert "Rows processed" in captured.out


def test_main_returns_error_code_for_bad_config(tmp_path, capsys):
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text("[paths]\n")  # missing required keys

    exit_code = main([str(bad_config)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error" in captured.err


def test_main_returns_error_code_for_missing_config_file(tmp_path):
    exit_code = main([str(tmp_path / "nope.toml")])
    assert exit_code == 1


def test_tautomer_standardisation_setting_is_honored(mock_dataset):
    config_path = write_config(mock_dataset, tautomer_standardisation="true")
    report = run(str(config_path))
    assert report["config"]["tautomer_standardisation"] is True


def test_report_format_txt_writes_human_readable_report(mock_dataset):
    config_path = write_config(mock_dataset, report_filename="report.txt", report_format='"txt"')
    report = run(str(config_path))
    assert report["config"]["report_format"] == "txt"

    report_output = mock_dataset / "report.txt"
    assert report_output.is_file()
    text = report_output.read_text()

    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert "challenge-data-checker audit report" in text
    assert "Train-Test Leakage: Exact Matches" in text
    assert "Quality Flags: Train" in text


def test_report_format_defaults_to_json(mock_dataset):
    config_path = write_config(mock_dataset)
    report = run(str(config_path))
    assert report["config"]["report_format"] == "json"
    assert (mock_dataset / "report.json").is_file()
