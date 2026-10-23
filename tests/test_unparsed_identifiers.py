import pandas as pd

from challenge_data_checker import audit
from challenge_data_checker.report import format_report_as_text


def run(train, test=(("c1ccccc1", "T1"),)):
    return audit(
        train=pd.DataFrame(train, columns=["SMILES", "id"]),
        test=pd.DataFrame(list(test), columns=["SMILES", "id"]),
        identifier_columns=["id"],
        print_report=False,
    )


def test_issue_reproduction_reports_broken_and_empty_rows_for_the_id():
    report = run([("CCO", "CPD-1"), ("#####", "CPD-1"), ("", "CPD-1")])
    issues = report["identifier_namespace_issues"]["id"]

    (broken,) = issues["id_with_unparseable_and_valid_structures"]
    assert broken["identifier"] == "CPD-1"
    assert [st["canonical_smiles"] for st in broken["structures"]] == ["CCO"]
    assert [(r["raw_smiles"], r["category"]) for r in broken["unparseable"]] == [
        ("#####", "placeholder")
    ]

    (missing,) = issues["missing_structure_resolvable_by_id"]
    assert missing["ambiguous"] is False
    assert [loc["row_index"] for loc in missing["missing"]] == [2]
    assert [c["canonical_smiles"] for c in missing["candidates"]] == ["CCO"]


def test_unparseable_entries_are_annotated_when_resolvable():
    report = run([("CCO", "CPD-1"), ("", "CPD-1"), ("", "CPD-2"), ("#####", "CPD-1")])
    by_row = {e["row_index"]: e["resolvable_via_identifier"] for e in report["unparseable_smiles"]}
    assert by_row == {1: "id", 2: None, 3: None}  # only the empty row with a known id
    assert report["summary"]["total_unparseable_resolvable_via_identifier"] == 1


def test_resolution_works_across_pools():
    report = run([("", "CPD-1")], test=[("CCO", "CPD-1")])
    assert report["unparseable_smiles"][0]["resolvable_via_identifier"] == "id"


def test_ambiguous_when_the_id_has_several_structures():
    report = run([("CCO", "CPD-1"), ("CCN", "CPD-1"), ("", "CPD-1")])
    (missing,) = report["identifier_namespace_issues"]["id"]["missing_structure_resolvable_by_id"]
    assert missing["ambiguous"] is True
    assert len(missing["candidates"]) == 2
    assert report["unparseable_smiles"][0]["resolvable_via_identifier"] is None


def test_new_findings_warn_but_unparseable_rows_still_block():
    report = run([("CCO", "CPD-1"), ("", "CPD-1"), ("#####", "CPD-1")])
    s = report["summary"]
    assert s["status"] == "FAIL"
    assert "unparseable_smiles" in s["blocking"]["by_type"]
    assert {
        "id_with_unparseable_and_valid_structures",
        "missing_structure_resolvable_by_id",
    } <= set(s["flagged"]["by_type"])


def test_text_report_describes_the_findings():
    text = format_report_as_text(run([("CCO", "CPD-1"), ("", "CPD-1"), ("#####", "CPD-1")]))
    assert "identifier 'CPD-1' has valid and unparseable SMILES" in text
    assert "identifier 'CPD-1' has empty SMILES (resolvable)" in text
    assert "(resolvable via identifier column 'id')" in text
