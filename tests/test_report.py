import json
from datetime import datetime

import pytest

from challenge_data_checker.config import Config, ColumnsConfig, PathsConfig, SettingsConfig
from challenge_data_checker.models import MoleculeRecord
from challenge_data_checker.report import (
    build_report,
    format_report_as_text,
    print_summary,
    save_report,
)


def make_config(tmp_path):
    return Config(
        paths=PathsConfig(
            train_files=["train.csv"],
            test_files=["test.csv"],
            report_output=str(tmp_path / "report.json"),
        ),
        columns=ColumnsConfig(smiles_column="SMILES", identifier_columns=["compound_id"]),
        settings=SettingsConfig(),
        config_path=tmp_path / "config.toml",
    )


def make_record(pool, row_index, raw_smiles, canonical_smiles, inchikey, identifiers=None, **flags):
    return MoleculeRecord(
        pool=pool,
        source_file=f"{pool}.csv",
        row_index=row_index,
        raw_smiles=raw_smiles,
        identifiers=identifiers or {},
        canonical_smiles=canonical_smiles,
        inchikey=inchikey,
        fingerprint=None,
        heavy_atom_count=10,
        num_fragments=1,
        **flags,
    )


def test_build_report_contains_expected_top_level_keys(tmp_path):
    config = make_config(tmp_path)
    train = [make_record("train", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"})]
    test = [make_record("test", 0, "CCN", "CCN", "IK2", {"compound_id": "C2"})]

    report = build_report(config, train, test, ["compound_id"])

    for key in (
        "generated_at",
        "config",
        "summary",
        "unparseable_smiles",
        "train_test_leakage",
        "internal_duplicates",
        "identifier_namespace_issues",
        "quality_flags",
    ):
        assert key in report

    assert report["summary"]["total_train_rows"] == 1
    assert report["summary"]["total_test_rows"] == 1
    assert report["config"]["report_format"] == "json"


def test_build_report_records_resolved_smiles_columns(tmp_path):
    config = make_config(tmp_path)

    report = build_report(
        config,
        [],
        [],
        [],
        {"train": {"train.csv": "canonical_smiles"}, "test": {"test.csv": "SMILES"}},
    )

    assert report["config"]["resolved_smiles_columns"] == {
        "train": {"train.csv": "canonical_smiles"},
        "test": {"test.csv": "SMILES"},
    }


def test_build_report_resolved_smiles_columns_defaults_to_empty(tmp_path):
    config = make_config(tmp_path)

    report = build_report(config, [], [], [])

    assert report["config"]["resolved_smiles_columns"] == {"train": {}, "test": {}}


def test_build_report_generated_at_is_a_valid_recent_timestamp(tmp_path):
    config = make_config(tmp_path)
    before = datetime.now().astimezone().replace(microsecond=0)

    report = build_report(config, [], [], [])

    after = datetime.now().astimezone()
    generated_at = datetime.fromisoformat(report["generated_at"])
    assert before <= generated_at <= after


def test_save_report_writes_valid_json(tmp_path):
    config = make_config(tmp_path)
    train = [make_record("train", 0, "CCO", "CCO", "IK1")]
    test = [make_record("test", 0, "CCN", "CCN", "IK2")]
    report = build_report(config, train, test, [])

    output_path = tmp_path / "nested" / "report.json"
    save_report(report, output_path)

    assert output_path.is_file()
    loaded = json.loads(output_path.read_text())
    assert loaded["summary"]["total_train_rows"] == 1


def test_save_report_writes_valid_json_when_format_explicit(tmp_path):
    config = make_config(tmp_path)
    train = [make_record("train", 0, "CCO", "CCO", "IK1")]
    test = [make_record("test", 0, "CCN", "CCN", "IK2")]
    report = build_report(config, train, test, [])

    output_path = tmp_path / "report.json"
    save_report(report, output_path, "json")

    loaded = json.loads(output_path.read_text())
    assert loaded["summary"]["total_train_rows"] == 1


def test_save_report_writes_txt(tmp_path):
    config = make_config(tmp_path)
    train = [make_record("train", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"})]
    test = [make_record("test", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"})]
    report = build_report(config, train, test, ["compound_id"])

    output_path = tmp_path / "report.txt"
    save_report(report, output_path, "txt")

    assert output_path.is_file()
    text = output_path.read_text()
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert "challenge-data-checker audit report" in text
    assert "Train-Test Leakage: Exact Matches" in text
    assert "canonical_smiles = 'CCO'" in text
    assert "[train] train.csv (row 0): 'CCO'" in text


def test_save_report_invalid_format_raises(tmp_path):
    config = make_config(tmp_path)
    report = build_report(config, [], [], [])

    with pytest.raises(ValueError, match="report_format"):
        save_report(report, tmp_path / "report.bad", "yaml")


def test_format_report_as_text_renders_all_sections(tmp_path):
    config = make_config(tmp_path)
    train = [
        make_record("train", 0, "CCO", "CCO", "IK1", {"compound_id": "ID_A"}),
        make_record("train", 1, "CCO", "CCO", "IK1", {"compound_id": "ID_A"}),
    ]
    test = [make_record("test", 0, "CCO", "CCO", "IK1", {"compound_id": "ID_B"})]
    report = build_report(config, train, test, ["compound_id"])

    text = format_report_as_text(report)

    assert text.startswith("=" * 72)
    for heading in (
        "Configuration",
        "challenge-data-checker audit summary",
        "Unparseable SMILES",
        "Train-Test Leakage: Exact Matches",
        "Train-Test Leakage: Identifier Overlap",
        "Train-Test Leakage: Tanimoto Similarity",
        "Internal Duplicates: Train",
        "Internal Duplicates: Test",
        "Identifier Namespace Issues",
        "Quality Flags: Train",
        "Quality Flags: Test",
    ):
        assert heading in text
    # internal train duplicate (two identical CCO rows) rendered
    assert "identifier[compound_id] = 'ID_A'" in text
    # namespace mismatch: same structure, two different ids across pools
    assert "multiple identifiers" in text
    assert f"Generated: {report['generated_at']}" in text


def test_print_summary_runs_without_error(tmp_path, capsys):
    config = make_config(tmp_path)
    train = [make_record("train", 0, "CCO", "CCO", "IK1")]
    test = [make_record("test", 0, "CCN", "CCN", "IK2")]
    report = build_report(config, train, test, [])

    print_summary(report, config.paths.report_output)
    captured = capsys.readouterr()
    assert "challenge-data-checker audit summary" in captured.out
    assert "Rows processed" in captured.out
    assert report["generated_at"] in captured.out
