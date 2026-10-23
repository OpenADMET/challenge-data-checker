import pandas as pd

from challenge_data_checker import audit
from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import SettingsConfig
from challenge_data_checker.report import format_report_as_text


def run(train, test, **settings):
    """Audit (smiles, id) pairs with a single "id" identifier column."""
    return audit(
        train=pd.DataFrame(train, columns=["SMILES", "id"]),
        test=pd.DataFrame(test, columns=["SMILES", "id"]),
        identifier_columns=["id"],
        print_report=False,
        **settings,
    )


def flagged(report):
    return report["summary"]["flagged"]["by_type"]


def blocking(report):
    return report["summary"]["blocking"]["by_type"]


def test_whitespace_padded_id_is_the_same_id_within_a_pool():
    # The issue's reproduction: used to report one structure with two ids.
    report = run([("CCO", "CPD-1"), ("CCO", " CPD-1")], [("c1ccccc1", "T1")])
    namespace = report["identifier_namespace_issues"]["id"]
    assert namespace["one_structure_multiple_ids"] == []
    assert [e["raw_value"] for e in namespace["identifier_whitespace"]] == [" CPD-1"]
    assert namespace["identifier_whitespace"][0]["value"] == "CPD-1"
    assert "identifier_whitespace" in flagged(report)


def test_whitespace_padded_id_still_leaks_across_pools():
    report = run([("CCO", "CPD-1")], [("c1ccccc1", "CPD-1 ")])
    assert len(report["train_test_leakage"]["identifiers"]["id"]) == 1
    assert "identifier_overlap" in blocking(report)  # same id: full tier


def test_case_only_difference_is_a_near_duplicate_warning_by_default():
    report = run([("CCO", "Molport-001")], [("c1ccccc1", "MolPort-001")])
    assert report["train_test_leakage"]["identifiers"]["id"] == []
    (group,) = report["identifier_namespace_issues"]["id"]["near_duplicates"]
    assert [i["identifier"] for i in group["identifiers"]] == ["MolPort-001", "Molport-001"]
    assert report["summary"]["status"] == "WARN"
    assert set(flagged(report)) == {"identifier_near_duplicate"}


def test_case_insensitive_setting_makes_case_variants_the_same_id():
    report = run(
        [("CCO", "Molport-001")],
        [("c1ccccc1", "MolPort-001")],
        identifier_case_insensitive=True,
    )
    assert len(report["train_test_leakage"]["identifiers"]["id"]) == 1
    assert {"identifier_overlap", "one_id_multiple_structures"} <= set(blocking(report))
    assert report["identifier_namespace_issues"]["id"]["near_duplicates"] == []
    assert report["config"]["identifier_case_insensitive"] is True


def test_hyphen_and_underscore_variants_are_near_duplicates():
    report = run([("CCO", "CPD-1"), ("CCN", "CPD_1")], [("c1ccccc1", "T1")])
    (group,) = report["identifier_namespace_issues"]["id"]["near_duplicates"]
    assert group["key"] == "cpd1"
    # Not the same id, so neither a conflict nor a duplicate.
    assert report["identifier_namespace_issues"]["id"]["one_id_multiple_structures"] == []
    assert report["summary"]["status"] == "WARN"


def test_near_duplicates_include_rows_whose_smiles_did_not_parse():
    report = run([("CCO", "CPD-1"), ("C1CC", "cpd-1")], [("c1ccccc1", "T1")])
    (group,) = report["identifier_namespace_issues"]["id"]["near_duplicates"]
    rows = {o["row_index"] for i in group["identifiers"] for o in i["occurrences"]}
    assert rows == {0, 1}


def test_whitespace_only_identifier_is_treated_as_missing():
    processor = MoleculeProcessor(SettingsConfig())
    record = processor.process_row("train", "f.csv", 0, "CCO", {"id": "   "})
    assert record.identifiers == {"id": None}
    assert record.raw_identifiers == {"id": "   "}


def test_text_report_lists_identifier_hygiene_findings():
    report = run([("CCO", " CPD-1"), ("CCN", "cpd-1")], [("c1ccccc1", "T1")])
    text = format_report_as_text(report)
    assert "[WARN] id: ' CPD-1' has surrounding whitespace, compared as 'CPD-1'" in text
    assert "[WARN] id: near-duplicate identifiers 'CPD-1', 'cpd-1'" in text
