import logging
from pathlib import Path

import pytest
from rdkit import Chem

from challenge_data_checker.chemistry import MoleculeProcessor, capture_rdkit_log, parse_smiles
from challenge_data_checker.config import ColumnsConfig, Config, PathsConfig, SettingsConfig
from challenge_data_checker.parse_errors import classify_parse_failure
from challenge_data_checker.report import build_report, format_report_as_text


@pytest.mark.parametrize(
    ("smiles", "category", "message_fragment"),
    [
        ("C1CC", "ring_closure", "unclosed ring"),
        ("CN(C)(C)(C)C", "valence", "valence"),
        ("C" * 255 + "(", "truncated", "syntax error"),
        ("CC(C", "truncated", "extra open parentheses"),
        # Newer RDKit also reports the position; only "syntax error" is stable.
        ("CC[R]", "markush", "syntax error"),
        ("CCR1", "markush", ""),
        ("CCO.[0.5H2O]", "stoichiometry_text", "syntax error"),
        ("c1cccc1", "kekulisation", "kekulize"),
        ("NA", "placeholder", ""),
        ("[Gramicidin]", "placeholder", ""),
        ("###", "placeholder", ""),
        ("CC\nO", "control_characters", "control characters"),
        ("CC<NL>O", "control_characters", "control characters"),
        ("", "empty", "Empty"),
    ],
)
def test_parse_failures_are_classified_with_rdkit_message(smiles, category, message_fragment):
    result = parse_smiles(smiles)
    assert result.mol is None
    assert result.error_category == category
    assert message_fragment in result.error
    assert "Failed parsing SMILES" not in result.error  # the noisy summary line is dropped


def test_embedded_newline_is_not_silently_truncated():
    # RDKit itself would read only "CC" and return ethane.
    assert parse_smiles("CC\nO").mol is None
    # Surrounding whitespace, e.g. a trailing newline from a cell, is fine.
    assert parse_smiles("CCO\n").mol is not None


def test_cxsmiles_with_leading_whitespace_is_recovered():
    result = parse_smiles(" C[C@H](N)C(=O)O |&1:1|")
    assert result.mol is not None
    assert result.whitespace_recovered is True


def test_plain_smiles_with_whitespace_is_not_flagged():
    # RDKit already accepts it, so there's nothing to recover.
    result = parse_smiles("  CCO ")
    assert result.mol is not None
    assert result.whitespace_recovered is False


def test_rdkit_warning_is_captured_for_a_molecule_that_parses():
    result = parse_smiles("C/C=C(/F)/C=C/F")
    assert result.mol is not None
    assert "Conflicting single bond directions" in result.warning


def test_capture_restores_the_rdkit_logger():
    rdkit_logger = logging.getLogger("rdkit")
    before = (list(rdkit_logger.handlers), rdkit_logger.level, rdkit_logger.propagate)
    with capture_rdkit_log() as lines:
        Chem.MolFromSmiles("C1CC")
    captured = len(lines)
    Chem.MolFromSmiles("C1CC")  # outside the block: nothing more is captured
    after = (list(rdkit_logger.handlers), rdkit_logger.level, rdkit_logger.propagate)
    assert before == after
    assert captured > 0
    assert len(lines) == captured


def test_classify_prefers_raw_string_evidence_over_rdkit_message():
    # A truncated value that RDKit reports as a ring problem is still truncated.
    assert classify_parse_failure("C1" + "C" * 253, "unclosed ring") == "truncated"
    assert classify_parse_failure("CC1CC", "unclosed ring") == "ring_closure"
    assert classify_parse_failure("CC=?", "") == "other"


def test_process_row_records_category_warning_and_whitespace():
    processor = MoleculeProcessor(SettingsConfig())
    failed = processor.process_row("train", "f.csv", 0, "C1CC", {})
    assert failed.parse_error_category == "ring_closure"
    warned = processor.process_row("train", "f.csv", 1, "C/C=C(/F)/C=C/F", {})
    assert warned.is_parsed and warned.rdkit_warning
    recovered = processor.process_row("train", "f.csv", 2, " C[C@H](N)C(=O)O |&1:1|", {})
    assert recovered.is_parsed and recovered.smiles_whitespace_recovered


def _report(train_smiles, test_smiles):
    processor = MoleculeProcessor(SettingsConfig())
    config = Config(
        paths=PathsConfig(train_files=["train.csv"], test_files=["test.csv"], report_output=None),
        columns=ColumnsConfig(smiles_column="SMILES", identifier_columns=[]),
        settings=SettingsConfig(),
        config_path=Path("<test>"),
    )
    train = [
        processor.process_row("train", "train.csv", i, s, {}) for i, s in enumerate(train_smiles)
    ]
    test = [processor.process_row("test", "test.csv", i, s, {}) for i, s in enumerate(test_smiles)]
    return build_report(config, train, test, [])


def test_report_counts_parse_failures_by_category():
    report = _report(["C1CC", "CC[R]", "c1ccccc1CO"], ["C1CC"])
    assert report["summary"]["total_unparseable_by_category"] == {"markush": 1, "ring_closure": 2}
    assert {e["category"] for e in report["unparseable_smiles"]} == {"markush", "ring_closure"}
    assert report["summary"]["status"] == "FAIL"
    text = format_report_as_text(report)
    assert "-- ring_closure: unclosed ring" in text


def test_whitespace_recovery_and_rdkit_warnings_only_warn():
    report = _report([" C[C@H](N)C(=O)O |&1:1|"], ["C/C=C(/F)/C=C/F"])
    s = report["summary"]
    assert s["status"] == "WARN"
    assert set(s["flagged"]["by_type"]) == {"smiles_whitespace", "rdkit_warning"}
    assert s["total_smiles_whitespace"] == {"train": 1, "test": 0}
    assert s["total_rdkit_warnings"] == {"train": 0, "test": 1}
    text = format_report_as_text(report)
    assert "Conflicting single bond directions" in text
