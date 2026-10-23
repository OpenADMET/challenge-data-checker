import pandas as pd
import pytest

from challenge_data_checker import audit
from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import ConfigError, SettingsConfig, load_config

IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
NICOTINE = "CN1CCC[C@H]1c1cccnc1"


def record(smiles, **settings):
    return MoleculeProcessor(SettingsConfig(**settings)).process_row(
        "train", "f.csv", 0, smiles, {}
    )


def flags(r):
    return {
        name
        for name, on in {
            "salt_or_solvate": r.is_salt_or_solvate,
            "multi_component": r.is_multi_component,
            "homodimer_without_counterion": r.is_homodimer_without_counterion,
            "unclassified_mixture": r.is_unclassified_mixture,
            "contains_metal": r.contains_metal,
        }.items()
        if on
    }


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (f"{IBUPROFEN}.{NICOTINE}", {"multi_component"}),  # two drugs concatenated
        ("CCN.CCN", {"homodimer_without_counterion"}),  # 2:1 salt, acid stripped
        ("CCN(CC)CC.CCN(CC)CC.OS(=O)(=O)O", {"salt_or_solvate"}),  # genuine 2:1 salt
        ("c1ccccc1CCN.Cl", {"salt_or_solvate"}),
        ("c1ccccc1CCN.C1CC1", {"unclassified_mixture"}),  # small unknown fragment
        ("Cl.O", {"unclassified_mixture"}),  # no parent compound at all
        # Sodium acetate: acetate is itself a known counterion, so there's no parent compound.
        ("CC(=O)[O-].[Na+]", {"unclassified_mixture", "contains_metal"}),
        ("C[Hg]Cl", {"contains_metal"}),  # organometallic, single fragment
        (IBUPROFEN, set()),
    ],
)
def test_mixture_flags(smiles, expected):
    assert flags(record(smiles)) == expected


def test_every_multi_fragment_entry_gets_exactly_one_mixture_category():
    for smiles in [
        f"{IBUPROFEN}.{NICOTINE}",
        "CCN.CCN",
        "CCN(CC)CC.Cl",
        "c1ccccc1CCN.C1CC1",
        "Cl.O",
        f"{IBUPROFEN}.{NICOTINE}.Cl",
    ]:
        r = record(smiles)
        mixture_flags = flags(r) - {"contains_metal"}
        assert len(mixture_flags) == 1, (smiles, mixture_flags)


@pytest.mark.parametrize(
    ("smiles", "charge"),
    [
        ("CC(=O)[O-].CC(=O)[O-]", -2),  # dicarboxylate with its Ca2+ stripped
        ("C[N+](C)(C)C", 1),  # quaternary ammonium without its anion
        ("[NH3+]CC(=O)[O-]", 0),  # zwitterion: charge-separated but neutral
        ("C[N+](=O)[O-]", 0),  # nitro group
        ("CCN(CC)CC.Cl", 0),
    ],
)
def test_net_charge(smiles, charge):
    assert record(smiles).net_charge == charge


def test_multi_component_threshold_is_configurable():
    smiles = f"{IBUPROFEN}.C1CCCC1"  # cyclopentane: 5 heavy atoms, unrecognised
    assert "multi_component" not in flags(record(smiles))
    assert "multi_component" in flags(record(smiles, multi_component_min_heavy_atoms=5))


def _report(smiles):
    return audit(
        train=pd.DataFrame({"SMILES": smiles}),
        test=pd.DataFrame({"SMILES": ["c1ccccc1"]}),
        print_report=False,
    )


@pytest.mark.parametrize(
    ("smiles", "status", "finding_type"),
    [
        ("CCN.CCN", "FAIL", "homodimer_without_counterion"),
        # A missing counterion is a representation problem: flagged, not blocking.
        ("CC(=O)[O-].CC(=O)[O-]", "WARN", "net_charge"),
        ("C[N+](C)(C)CCCCCCCCCCCC", "WARN", "net_charge"),  # quaternary, no anion
        (f"{IBUPROFEN}.{NICOTINE}", "WARN", "multi_component"),
        ("c1ccccc1CCN.Cl", "WARN", "salt_or_solvate"),
        ("C[Hg]Cl", "WARN", "contains_metal"),
    ],
)
def test_mixture_flag_tiers(smiles, status, finding_type):
    report = _report([smiles])
    tier = "blocking" if status == "FAIL" else "flagged"
    assert report["summary"]["status"] == status
    assert finding_type in report["summary"][tier]["by_type"]


def test_deprecated_totals_are_kept_but_not_tiered():
    report = _report(["c1ccccc1CCN.Cl", "CCN.CCN"])
    totals = report["summary"]["total_quality_flags"]
    assert totals["mixtures"]["train"] == 2
    assert report["summary"]["total_mixtures"]["train"] == 2
    tiered = set(report["summary"]["blocking"]["by_type"]) | set(
        report["summary"]["flagged"]["by_type"]
    )
    assert not tiered & {"mixture", "mixtures", "salt_or_metal_complex"}


def test_multi_component_min_heavy_atoms_must_be_positive(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""
[paths]
train_files = ["train.csv"]
test_files = ["test.csv"]
report_output = "report.json"

[columns]
smiles_column = "SMILES"

[settings]
multi_component_min_heavy_atoms = 0
""")
    with pytest.raises(ConfigError, match="multi_component_min_heavy_atoms"):
        load_config(config)


@pytest.mark.parametrize(
    "counterion",
    ["CCS(=O)(=O)O", "OCCS(=O)(=O)O", "OS(=O)(=O)C(F)(F)F", "OC(=O)c1cc(S(=O)(=O)O)ccc1O"],
)
def test_esylate_isethionate_triflate_sulfosalicylate_salts(counterion):
    assert flags(record(f"{IBUPROFEN}.{counterion}")) == {"salt_or_solvate"}
