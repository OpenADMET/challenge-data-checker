import pandas as pd

from challenge_data_checker import audit, check
from challenge_data_checker.report import format_report_as_text


def test_same_structure_from_two_pipelines_is_flagged(tmp_path):
    # The same compounds, canonicalised by two different pipelines.
    a, b = tmp_path / "pipeline_a.csv", tmp_path / "pipeline_b.csv"
    pd.DataFrame({"SMILES": ["OCc1ccccc1", "c1ccccc1CCN"]}).to_csv(a, index=False)
    pd.DataFrame({"SMILES": ["C(O)c1ccccc1", "NCCc1ccccc1"]}).to_csv(b, index=False)

    report = check([str(a), str(b)], print_report=False)

    findings = report["inconsistent_smiles"]
    assert len(findings) == 2
    for finding in findings:
        assert [r["source_files"] for r in finding["representations"]] in (
            [[str(a)], [str(b)]],
            [[str(b)], [str(a)]],
        )
    assert report["summary"]["total_inconsistent_smiles"] == 2
    assert report["summary"]["flagged"]["by_type"]["inconsistent_smiles"] == {
        "findings": 2,
        "rows": 4,
    }


def test_identical_strings_and_whitespace_only_differences_are_not_flagged():
    report = check(pd.DataFrame({"SMILES": ["CCO", "CCO", " CCO"]}), print_report=False)
    assert report["inconsistent_smiles"] == []


def test_different_stereo_is_not_an_inconsistent_representation():
    # Different structures (full InChIKey differs), reported elsewhere.
    report = check(
        pd.DataFrame({"SMILES": ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]}), print_report=False
    )
    assert report["inconsistent_smiles"] == []


def test_inconsistent_smiles_across_pools_in_an_audit():
    report = audit(
        train=pd.DataFrame({"SMILES": ["OCc1ccccc1"]}),
        test=pd.DataFrame({"SMILES": ["C(O)c1ccccc1"]}),
        print_report=False,
    )
    assert report["summary"]["total_inconsistent_smiles"] == 1
    text = format_report_as_text(report)
    assert "Inconsistent SMILES Representations" in text
    assert "is written 2 ways" in text
