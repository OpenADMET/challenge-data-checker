from pathlib import Path

import pytest

from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import ColumnsConfig, Config, PathsConfig, SettingsConfig
from challenge_data_checker.models import MoleculeRecord
from challenge_data_checker.report import build_report, format_report_as_text, print_summary
from challenge_data_checker.severity import (
    TIERS,
    Severity,
    TierContext,
    _drop_redundant,
    severity_of,
)

_PROCESSOR = MoleculeProcessor(SettingsConfig())


def audit(train, test, identifier_columns=(), **settings):
    """Build a report from (smiles, identifiers) rows, without touching disk."""
    config = Config(
        paths=PathsConfig(train_files=["train.csv"], test_files=["test.csv"], report_output=None),
        columns=ColumnsConfig(smiles_column="SMILES", identifier_columns=list(identifier_columns)),
        settings=SettingsConfig(**settings),
        config_path=Path("<test>"),
    )

    def records(pool, rows):
        return [
            _PROCESSOR.process_row(pool, f"{pool}.csv", i, smiles, ids)
            for i, (smiles, ids) in enumerate(rows)
        ]

    return build_report(
        config, records("train", train), records("test", test), list(identifier_columns)
    )


def rows(*smiles):
    return [(s, {}) for s in smiles]


def test_one_overlap_and_one_duplicate_count_as_two_problems_not_eleven():
    # #4: ethanol twice in train, once in test, plus unrelated benzene. Two real
    # problems (one train/test overlap, one train duplicate), each seen through
    # raw SMILES, canonical SMILES and InChIKey.
    report = audit(rows("CCO", "CCO", "c1ccccc1"), rows("CCO"), identifier_columns=[])
    s = report["summary"]

    assert s["status"] == "FAIL"
    assert s["blocking"]["findings"] == 1
    assert s["blocking"]["by_type"] == {"exact_overlap": {"findings": 1, "rows": 3}}
    assert s["flagged"]["findings"] == 1
    assert s["flagged"]["by_type"] == {"duplicate_structure": {"findings": 1, "rows": 2}}


def test_clean_split_passes():
    report = audit(rows("c1ccccc1CCN"), rows("c1ccccc1CCO"))
    s = report["summary"]
    assert s["status"] == "PASS"
    assert s["blocking"] == {"findings": 0, "rows": 0, "by_type": {}}
    assert s["flagged"] == {"findings": 0, "rows": 0, "by_type": {}}


def test_duplicates_warn_in_train_but_fail_in_test():
    in_train = audit(rows("c1ccccc1CCN", "c1ccccc1CCN"), rows("c1ccccc1CCO"))
    in_test = audit(rows("c1ccccc1CCO"), rows("c1ccccc1CCN", "c1ccccc1CCN"))
    assert in_train["summary"]["status"] == "WARN"
    assert in_test["summary"]["status"] == "FAIL"
    assert "duplicate_structure" in in_test["summary"]["blocking"]["by_type"]


def test_duplicate_identifiers_warn_in_train_but_fail_in_test():
    train = [("c1ccccc1CCN", {"id": "A"}), ("c1ccccc1CCO", {"id": "A"})]
    test = [("c1ccccc1CCC", {"id": "B"})]
    in_train = audit(train, test, identifier_columns=["id"])
    in_test = audit(test, train, identifier_columns=["id"])
    assert "duplicate_identifier" in in_train["summary"]["flagged"]["by_type"]
    assert "duplicate_identifier" in in_test["summary"]["blocking"]["by_type"]


def test_stereoisomer_overlap_and_duplicates_only_warn():
    across = audit(rows("CC(N)C(=O)O"), rows("C[C@H](N)C(=O)O"))
    within_test = audit(rows("c1ccccc1"), rows("CC(N)C(=O)O", "C[C@H](N)C(=O)O"))
    assert across["summary"]["status"] == "WARN"
    assert set(across["summary"]["flagged"]["by_type"]) == {"stereo_overlap"}
    assert within_test["summary"]["status"] == "WARN"
    assert set(within_test["summary"]["flagged"]["by_type"]) == {"duplicate_stereoisomer"}


def test_salt_form_overlap_fails():
    report = audit(rows("CCN(CC)CC.Cl"), rows("CCN(CC)CC"))
    assert report["summary"]["status"] == "FAIL"
    assert "salt_form_overlap" in report["summary"]["blocking"]["by_type"]


@pytest.mark.parametrize(("threshold", "tier"), [(1.0, "flagged"), (0.3, "blocking")])
def test_tanimoto_tier_depends_on_threshold(threshold, tier):
    # A fingerprint collision between different structures is reported at 1.0;
    # a close analogue only at a lowered threshold.
    smiles_train, smiles_test = ("CCCCO", "CCCCC") if threshold < 1 else ("CCCCCCCC", "CCCCCCCCC")
    report = audit(rows(smiles_train), rows(smiles_test), max_train_test_similarity=threshold)
    by_type = report["summary"][tier]["by_type"]
    assert "tanimoto_similarity" in by_type


@pytest.mark.parametrize("pool", ["train", "test"])
def test_unparseable_smiles_fail_in_either_pool(pool):
    good, bad = rows("c1ccccc1CCO"), rows("not a smiles")
    train, test = (bad, good) if pool == "train" else (good, bad)
    report = audit(train, test)
    assert report["summary"]["status"] == "FAIL"
    assert "unparseable_smiles" in report["summary"]["blocking"]["by_type"]


def test_quality_flags_only_warn():
    report = audit(rows("c1ccccc1CCN.Cl"), rows("C#N"))
    s = report["summary"]
    assert s["status"] == "WARN"
    assert set(s["flagged"]["by_type"]) >= {"salt_or_metal_complex", "suspicious_fragment"}


def test_one_id_multiple_structures_fails_and_one_structure_multiple_ids_warns():
    conflict = audit(
        [("c1ccccc1CCN", {"id": "A"})], [("c1ccccc1CCO", {"id": "A"})], identifier_columns=["id"]
    )
    assert "one_id_multiple_structures" in conflict["summary"]["blocking"]["by_type"]
    renamed = audit(
        [("c1ccccc1CCN", {"id": "A"}), ("c1ccccc1CCN", {"id": "B"})],
        [("c1ccccc1CCO", {"id": "C"})],
        identifier_columns=["id"],
    )
    assert "one_structure_multiple_ids" in renamed["summary"]["flagged"]["by_type"]


def test_severity_of_per_pool_rule_needs_a_pool():
    assert severity_of("duplicate_structure", "train", TierContext()) is Severity.WARN
    assert severity_of("duplicate_structure", "test", TierContext()) is Severity.FAIL
    with pytest.raises(ValueError, match="needs a pool"):
        severity_of("duplicate_structure", None, TierContext())


def test_every_tier_rule_resolves():
    for finding_type in TIERS:
        for pool in ("train", "test"):
            assert severity_of(finding_type, pool, TierContext()) in Severity


def test_drop_redundant_keeps_only_maximal_row_sets():
    a = frozenset({("train", "f", 0), ("train", "f", 1)})
    subset = frozenset({("train", "f", 0)})
    other = frozenset({("train", "f", 2), ("train", "f", 3)})
    assert sorted(_drop_redundant([a, subset, a, other]), key=sorted) == sorted(
        [a, other], key=sorted
    )


def test_text_report_tags_findings_with_their_tier():
    report = audit(rows("c1ccccc1CCN", "c1ccccc1CCN"), rows("c1ccccc1CCN"))
    text = format_report_as_text(report)
    assert "[FAIL] inchikey = " in text  # exact overlap
    assert "[WARN] inchikey = " in text  # train duplicate
    assert "Overall status        : FAIL" in text


@pytest.mark.parametrize(
    ("train", "test", "banner"),
    [
        (["c1ccccc1CCN"], ["c1ccccc1CCO"], "OVERALL PASS"),
        (["c1ccccc1CCN", "c1ccccc1CCN"], ["c1ccccc1CCO"], "OVERALL WARN - 1 flagged finding(s)"),
        (["c1ccccc1CCN"], ["c1ccccc1CCN"], "OVERALL FAIL - 1 blocking finding(s) in 2 row(s)"),
    ],
)
def test_dashboard_banner_is_three_state(capsys, train, test, banner):
    print_summary(audit(rows(*train), rows(*test)), None)
    assert banner in capsys.readouterr().out


def test_records_are_plain_molecule_records():
    # Guards the helper above against silently building a different type.
    assert isinstance(_PROCESSOR.process_row("train", "f", 0, "C", {}), MoleculeRecord)
