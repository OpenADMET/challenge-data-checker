import pandas as pd
import pytest

from challenge_data_checker import audit
from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import SettingsConfig
from challenge_data_checker.report import format_report_as_text


@pytest.mark.parametrize(
    ("smiles", "centres", "bonds"),
    [
        ("CC(N)C(=O)O", 1, 0),  # racemic / unspecified alanine
        ("C[C@H](N)C(=O)O", 0, 0),
        ("CC=CC", 0, 1),  # 2-butene, E/Z unspecified
        ("C/C=C/C", 0, 0),
        ("CC1CCC(C)CC1", 2, 0),  # ring cis/trans
        ("CS(=O)CC", 1, 0),  # sulfoxide
        ("CCO", 0, 0),
    ],
)
def test_undefined_stereo_counts(smiles, centres, bonds):
    record = MoleculeProcessor(SettingsConfig()).process_row("train", "f.csv", 0, smiles, {})
    assert (record.undefined_stereocentres, record.undefined_stereo_bonds) == (centres, bonds)


def test_counted_before_tautomer_standardisation():
    processor = MoleculeProcessor(SettingsConfig(tautomer_standardisation=True))
    record = processor.process_row("train", "f.csv", 0, "CC(N)C(=O)O", {})
    assert record.undefined_stereocentres == 1


def test_undefined_stereo_is_flagged_in_both_pools():
    report = audit(
        train=pd.DataFrame({"SMILES": ["CC(N)C(=O)O"]}),
        test=pd.DataFrame({"SMILES": ["CC=CC"]}),
        print_report=False,
    )
    flags = report["quality_flags"]
    assert flags["train"]["undefined_stereo"][0]["stereocentres"] == 1
    assert flags["test"]["undefined_stereo"][0]["stereo_bonds"] == 1
    assert report["summary"]["status"] == "WARN"
    assert report["summary"]["flagged"]["by_type"]["undefined_stereo"]["rows"] == 2
    text = format_report_as_text(report)
    assert "1 undefined stereocentre(s), 0 undefined stereo double bond(s)" in text
