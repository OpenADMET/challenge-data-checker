import json
from datetime import UTC, datetime, timedelta

import pytest

from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import ColumnsConfig, Config, PathsConfig, SettingsConfig
from challenge_data_checker.models import MoleculeRecord
from challenge_data_checker.report import (
    build_report,
    format_report_as_text,
    print_summary,
    save_report,
)
from challenge_data_checker.report_types import (
    QUALITY_FLAG_CATEGORIES,
    REPRESENTATIONS,
    InternalDuplicates,
    NamespaceIssues,
    QualityFlags,
    Report,
    ReportConfig,
    Summary,
    TrainTestLeakage,
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


def test_build_report_keys_match_typed_schema(tmp_path):
    # mypy only checks the code that builds the report; this guards the
    # TypedDict schema against drifting from what's actually emitted.
    config = make_config(tmp_path)
    train = [
        make_record("train", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"}),
        make_record("train", 1, "CCO", "CCO", "IK1", {"compound_id": "C1"}),
    ]
    test = [make_record("test", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"})]

    report = build_report(config, train, test, ["compound_id"])

    assert set(report) == set(Report.__annotations__)
    assert set(report["config"]) == set(ReportConfig.__annotations__)
    assert set(report["summary"]) == set(Summary.__annotations__)
    assert set(report["train_test_leakage"]) == set(TrainTestLeakage.__annotations__)
    for pool in ("train", "test"):
        assert set(report["internal_duplicates"][pool]) == set(InternalDuplicates.__annotations__)
        assert set(report["quality_flags"][pool]) == set(QualityFlags.__annotations__)
    assert set(report["internal_duplicates"]["train"]) == {*REPRESENTATIONS, "identifiers"}
    assert set(report["quality_flags"]["train"]) == set(QUALITY_FLAG_CATEGORIES)
    assert set(report["identifier_namespace_issues"]["compound_id"]) == set(
        NamespaceIssues.__annotations__
    )


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
    before = datetime.now(UTC).replace(microsecond=0)

    report = build_report(config, [], [], [])

    after = datetime.now(UTC)
    generated_at = datetime.fromisoformat(report["generated_at"])
    assert before <= generated_at <= after
    assert generated_at.utcoffset() == timedelta(0)


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


def test_build_report_counts_stereo_only_overlap_separately(tmp_path):
    # Train has L-alanine; test has L-alanine (exact) and the racemate
    # (same connectivity, different stereo): one exact and one stereo-only
    # test row, and the train row only counts as exact.
    processor = MoleculeProcessor(SettingsConfig())
    train = [processor.process_row("train", "train.csv", 0, "C[C@H](N)C(=O)O", {})]
    test = [
        processor.process_row("test", "test.csv", 0, "C[C@H](N)C(=O)O", {}),
        processor.process_row("test", "test.csv", 1, "CC(N)C(=O)O", {}),
    ]

    report = build_report(make_config(tmp_path), train, test, [])

    s = report["summary"]
    assert (s["total_exact_leaked_train_molecules"], s["total_exact_leaked_test_molecules"]) == (
        1,
        1,
    )
    assert (s["total_stereo_leaked_train_molecules"], s["total_stereo_leaked_test_molecules"]) == (
        0,
        1,
    )
    text = format_report_as_text(report)
    stereo_section = text.split("Same Connectivity, Different Stereochemistry")[1]
    assert "'CC(N)C(=O)O'" in stereo_section.split("Tanimoto")[0]


def test_build_report_counts_salt_form_only_overlap_separately(tmp_path):
    processor = MoleculeProcessor(SettingsConfig())
    train = [processor.process_row("train", "train.csv", 0, "CCN(CC)CC.Cl", {})]
    test = [processor.process_row("test", "test.csv", 0, "CCN(CC)CC", {})]

    report = build_report(make_config(tmp_path), train, test, [])

    s = report["summary"]
    assert s["total_exact_leaked_test_molecules"] == 0
    assert s["total_salt_form_leaked_train_molecules"] == 1
    assert s["total_salt_form_leaked_test_molecules"] == 1
    assert s["total_stereo_leaked_test_molecules"] == 0
    assert report["config"]["strip_salts"] is True
    text = format_report_as_text(report)
    salt_section = text.split("Same Parent Compound, Different Salt Form")[1].split("Same Conn")[0]
    assert "'CCN(CC)CC.Cl'" in salt_section
    assert "strip_salts              : True" in text
