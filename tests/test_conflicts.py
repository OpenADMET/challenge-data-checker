import pandas as pd
import pytest

from challenge_data_checker import audit
from challenge_data_checker.report import format_report_as_text

R_ALA, S_ALA, RAC_ALA = "C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O", "CC(N)C(=O)O"


def run(rows, **settings):
    return audit(
        train=pd.DataFrame(rows, columns=["SMILES", "id"]),
        test=pd.DataFrame([("c1ccccc1", "T1")], columns=["SMILES", "id"]),
        identifier_columns=["id"],
        print_report=False,
        **settings,
    )


def conflict(report):
    (finding,) = report["identifier_namespace_issues"]["id"]["one_id_multiple_structures"]
    return finding


@pytest.mark.parametrize(
    ("smiles", "conflict_type"),
    [
        (["CCN(CC)CC", "CCN(CC)CC.Cl"], "salt_form"),
        ([RAC_ALA, R_ALA, S_ALA], "stereo"),
        ([f"{RAC_ALA}.Cl", R_ALA], "stereo"),  # salt form and stereo both differ
        (["Oc1ccccn1", "O=c1cccc[nH]1"], "tautomer"),  # 2-hydroxypyridine / 2-pyridone
        (["c1ccccc1CCN", "c1ccc2ccccc2c1"], "different_compound"),
    ],
)
def test_one_id_conflicts_are_classified(smiles, conflict_type):
    report = run([(s, "CPD-1") for s in smiles])
    assert conflict(report)["conflict_type"] == conflict_type


def test_only_different_compounds_block():
    variants = run([("CCN(CC)CC", "CPD-1"), ("CCN(CC)CC.Cl", "CPD-1")])
    different = run([("c1ccccc1CCN", "CPD-1"), ("c1ccc2ccccc2c1", "CPD-1")])
    assert "one_id_structure_variants" in variants["summary"]["flagged"]["by_type"]
    assert "one_id_multiple_structures" not in variants["summary"]["blocking"]["by_type"]
    assert "one_id_multiple_structures" in different["summary"]["blocking"]["by_type"]


def test_different_compound_reports_max_similarity():
    report = run([("c1ccccc1CCN", "CPD-1"), ("c1ccccc1CCO", "CPD-1"), ("CCCCCC", "CPD-1")])
    finding = conflict(report)
    assert finding["conflict_type"] == "different_compound"
    assert 0.0 < finding["max_similarity"] < 1.0


def test_salt_forms_are_different_compounds_without_strip_salts():
    report = run([("CCN(CC)CC", "CPD-1"), ("CCN(CC)CC.Cl", "CPD-1")], strip_salts=False)
    assert conflict(report)["conflict_type"] == "different_compound"


def test_summary_counts_conflicts_by_type():
    report = run([("CCN(CC)CC", "A"), ("CCN(CC)CC.Cl", "A"), ("c1ccccc1CCN", "B"), ("CCCCCC", "B")])
    counts = report["summary"]["total_identifier_namespace_issues"]["id"]
    assert counts["one_id_multiple_structures"] == 2
    assert counts["one_id_multiple_structures_by_type"] == {
        "different_compound": 1,
        "salt_form": 1,
    }


@pytest.mark.parametrize(
    ("rows", "conflict_type"),
    [
        ([("CCN(CC)CC", "A"), ("CCN(CC)CC", "B")], "same_structure"),
        ([("CCN(CC)CC", "A"), ("CCN(CC)CC.Cl", "B")], "salt_form"),
        ([(R_ALA, "A"), (S_ALA, "B")], "stereo"),
    ],
)
def test_one_compound_with_several_ids_is_found_across_salt_and_stereo(rows, conflict_type):
    report = run(rows)
    (finding,) = report["identifier_namespace_issues"]["id"]["one_structure_multiple_ids"]
    assert finding["conflict_type"] == conflict_type
    assert [i["identifier"] for i in finding["identifiers"]] == ["A", "B"]
    assert report["summary"]["status"] == "WARN"


def test_text_report_shows_conflict_type_and_tier():
    text = format_report_as_text(
        run([("CCN(CC)CC", "A"), ("CCN(CC)CC.Cl", "A"), ("c1ccccc1CCN", "B"), ("CCCCCC", "B")])
    )
    assert "[WARN] id: identifier 'A' -> multiple structures (salt_form" in text
    assert "[FAIL] id: identifier 'B' -> multiple structures (different_compound" in text
