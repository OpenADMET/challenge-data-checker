import json

import pandas as pd
import pytest

from challenge_data_checker import audit
from challenge_data_checker.api import audit as audit_from_module


@pytest.fixture
def train_test_dfs():
    # Same leakage/duplicate shape as the CLI end-to-end fixture: CCO leaks
    # exactly and is duplicated within train; benzene leaks with mismatched ids.
    train_df = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCO", "c1ccccc1"],
            "compound_id": ["C1", "C1b", "C2"],
        }
    )
    test_df = pd.DataFrame(
        {
            "SMILES": ["CCO", "c1ccccc1", "CCN"],
            "compound_id": ["T1", "T2", "T3"],
        }
    )
    return train_df, test_df


def test_audit_is_exported_from_package_root():
    assert audit is audit_from_module


def test_audit_accepts_dataframes_directly(train_test_dfs):
    train_df, test_df = train_test_dfs
    report = audit(
        train=train_df,
        test=test_df,
        identifier_columns=["compound_id"],
        print_report=False,
    )

    assert report["summary"]["total_train_rows"] == 3
    assert report["summary"]["total_test_rows"] == 3
    assert len(report["train_test_leakage"]["exact"]["inchikey"]) == 2
    assert len(report["internal_duplicates"]["train"]["inchikey"]) == 1


def test_audit_labels_dataframe_sources_in_config_echo(train_test_dfs):
    train_df, test_df = train_test_dfs
    report = audit(train=train_df, test=test_df, print_report=False)

    assert report["config"]["train_files"] == ["<in-memory dataframe #0 (3 rows)>"]
    assert report["config"]["test_files"] == ["<in-memory dataframe #0 (3 rows)>"]


def test_audit_accepts_mixed_paths_and_dataframes(tmp_path, train_test_dfs):
    train_df, test_df = train_test_dfs
    extra_path = tmp_path / "extra_train.csv"
    pd.DataFrame({"SMILES": ["CCC"], "compound_id": ["C9"]}).to_csv(extra_path, index=False)

    report = audit(
        train=[train_df, str(extra_path)],
        test=test_df,
        identifier_columns=["compound_id"],
        print_report=False,
    )

    assert report["summary"]["total_train_rows"] == 4
    assert report["config"]["train_files"] == [
        "<in-memory dataframe #0 (3 rows)>",
        str(extra_path),
    ]


def test_audit_does_not_write_report_by_default(train_test_dfs, tmp_path, monkeypatch):
    train_df, test_df = train_test_dfs
    monkeypatch.chdir(tmp_path)
    audit(train=train_df, test=test_df, print_report=False)

    assert list(tmp_path.iterdir()) == []


def test_audit_writes_report_when_output_given(train_test_dfs, tmp_path):
    train_df, test_df = train_test_dfs
    output_path = tmp_path / "report.json"

    report = audit(train=train_df, test=test_df, report_output=output_path, print_report=False)

    assert output_path.is_file()
    on_disk = json.loads(output_path.read_text())
    assert on_disk["summary"]["total_train_rows"] == report["summary"]["total_train_rows"]


def test_audit_infers_txt_format_from_output_extension(train_test_dfs, tmp_path):
    train_df, test_df = train_test_dfs
    output_path = tmp_path / "report.txt"

    report = audit(train=train_df, test=test_df, report_output=output_path, print_report=False)

    assert report["config"]["report_format"] == "txt"
    text = output_path.read_text()
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert "challenge-data-checker audit report" in text


def test_audit_explicit_report_format_overrides_inference(train_test_dfs, tmp_path):
    train_df, test_df = train_test_dfs
    output_path = tmp_path / "report.txt"

    report = audit(
        train=train_df,
        test=test_df,
        report_output=output_path,
        report_format="json",
        print_report=False,
    )

    assert report["config"]["report_format"] == "json"
    assert json.loads(output_path.read_text())["summary"]["total_train_rows"] == 3


def test_audit_prints_summary_by_default(train_test_dfs, capsys):
    train_df, test_df = train_test_dfs
    audit(train=train_df, test=test_df)

    captured = capsys.readouterr()
    assert "challenge-data-checker audit summary" in captured.out


def test_audit_can_suppress_printed_summary(train_test_dfs, capsys):
    train_df, test_df = train_test_dfs
    audit(train=train_df, test=test_df, print_report=False)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_audit_single_dataframe_and_single_path_are_not_required_to_be_lists(
    tmp_path, train_test_dfs
):
    train_df, test_df = train_test_dfs
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    report = audit(train=train_df, test=str(test_path), print_report=False)

    assert report["summary"]["total_train_rows"] == 3
    assert report["summary"]["total_test_rows"] == 3
