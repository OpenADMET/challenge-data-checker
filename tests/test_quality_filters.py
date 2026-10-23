import pytest
from rdkit import Chem

from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import SettingsConfig
from challenge_data_checker.quality_filters import (
    _COUNTERION_SMILES_RAW,
    _SMALL_MOLECULE_ALLOWLIST_RAW,
    _SOLVENT_SMILES_RAW,
    COUNTERION_KEYS,
    SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES,
    SOLVENT_KEYS,
    classify_fragments,
    contains_metal,
    has_multiple_fragments,
    is_salt_or_metal_complex,
    is_suspicious_small_fragment,
)


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_has_multiple_fragments_dot_in_smiles():
    m = mol("CC(=O)O.CC(=O)O")
    assert has_multiple_fragments(m, "CC(=O)O.CC(=O)O") is True


def test_has_multiple_fragments_single_fragment_is_false():
    m = mol("CCO")
    assert has_multiple_fragments(m, "CCO") is False


def test_contains_metal_true_for_transition_metal():
    m = mol("[Fe+2]")
    assert contains_metal(m) is True


def test_contains_metal_false_for_organic():
    m = mol("CCO")
    assert contains_metal(m) is False


def test_is_salt_or_metal_complex_for_metal_containing_mol():
    m = mol("[Na+].[Cl-]")
    assert is_salt_or_metal_complex(m) is True


def test_is_salt_or_metal_complex_for_organic_counterion_fragment():
    m = mol("c1ccccc1CN.CC(=O)[O-]")  # amine acetate salt
    assert is_salt_or_metal_complex(m) is True


def test_is_salt_or_metal_complex_false_for_plain_organic():
    m = mol("c1ccccc1")
    assert is_salt_or_metal_complex(m) is False


def test_is_suspicious_small_fragment_flags_bare_cyanide():
    m = mol("C#N")
    assert is_suspicious_small_fragment(m, m.GetNumHeavyAtoms()) is True


def test_is_suspicious_small_fragment_flags_lone_ion():
    m = mol("[Cl-]")
    assert is_suspicious_small_fragment(m, m.GetNumHeavyAtoms()) is True


def test_is_suspicious_small_fragment_false_for_larger_molecule():
    m = mol("c1ccccc1CCN")
    assert is_suspicious_small_fragment(m, m.GetNumHeavyAtoms()) is False


def test_ethanol_is_not_a_suspicious_fragment():
    m = mol("CCO")
    assert not is_suspicious_small_fragment(m, m.GetNumHeavyAtoms())


@pytest.mark.parametrize("smiles", ["O", "N", "C"])
def test_is_suspicious_small_fragment_still_flags_bare_atoms(smiles):
    m = mol(smiles)
    assert is_suspicious_small_fragment(m, m.GetNumHeavyAtoms()) is True


@pytest.mark.parametrize("tautomer_standardisation", [False, True])
@pytest.mark.parametrize("smiles", sorted(_SMALL_MOLECULE_ALLOWLIST_RAW))
def test_allowlisted_small_molecules_are_not_suspicious(smiles, tautomer_standardisation):
    # Runs the real pipeline, so an allowlist entry whose tautomer-canonical
    # form differs from its plain canonical form would be caught here.
    processor = MoleculeProcessor(SettingsConfig(tautomer_standardisation=tautomer_standardisation))
    record = processor.process_row("train", "f.csv", 0, smiles, {})
    assert record.is_parsed
    assert record.heavy_atom_count is not None and record.heavy_atom_count <= 3
    assert record.is_suspicious_fragment is False


def test_allowlist_entries_all_parse():
    assert len(SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES) == len(_SMALL_MOLECULE_ALLOWLIST_RAW)


def test_allowlist_excludes_stripped_pipeline_artefacts():
    assert "C" not in SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES
    assert "C#N" not in SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES


@pytest.mark.parametrize(
    "smiles",
    [
        "CCN(CC)CC.Cl",  # hydrochloride, neutral acid form
        "CCN(CC)CC.Br",  # hydrobromide
        "CCN(CC)CC.OC(=O)C(F)(F)F",  # TFA
        "CCN(CC)CC.CS(=O)(=O)O",  # mesylate, neutral form
        "CCN(CC)CC.OC(=O)/C=C\\C(=O)O",  # maleate
        "CCN(CC)CC.O",  # hydrate
        "CCN(CC)CC.[Cl-]",  # ionic chloride
        "CC[NH+](CC)CC.[Cl-]",  # fully ionic drawing
        "CCN(CC)CC.Cl.O",  # hydrochloride hydrate
    ],
)
def test_neutral_and_ionic_counterions_are_salts(smiles):
    assert is_salt_or_metal_complex(mol(smiles)) is True
    assert Chem.MolToSmiles(classify_fragments(mol(smiles)).parent) == "CCN(CC)CC"


def test_parent_is_not_a_large_counterion():
    pamoate = "OC(=O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O"
    result = classify_fragments(mol(f"CNC(C)Cc1ccccc1.{pamoate}"))
    assert Chem.MolToSmiles(result.parent) == "CNC(C)Cc1ccccc1"
    assert result.is_salt_or_solvate


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1.c1ccccc1",  # homodimer: no counterion to strip
        "CCN(CC)CC.Cl.c1ccncc1",  # an unknown organic fragment alongside the salt
        "CC(=O)O.CC(=O)O",  # only counterion-like fragments, no parent compound
        "Cl.O",
        "C#N",  # a lone counterion is a suspicious fragment, not a salt
        "CCN(CC)CC",
    ],
)
def test_not_salts(smiles):
    assert is_salt_or_metal_complex(mol(smiles)) is False


def test_stripped_fragments_are_classified():
    result = classify_fragments(mol("CCN(CC)CC.Cl.O.c1ccncc1"))
    assert sorted(result.stripped) == [
        ("Cl", "counterion"),
        ("O", "solvent"),
        ("c1ccncc1", "unknown"),
    ]


def test_counterion_and_solvent_sets_parse_and_are_disjoint():
    assert len(COUNTERION_KEYS) == len(_COUNTERION_SMILES_RAW)
    assert len(SOLVENT_KEYS) == len(_SOLVENT_SMILES_RAW)
    assert not COUNTERION_KEYS & SOLVENT_KEYS


@pytest.mark.parametrize("tautomer_standardisation", [False, True])
@pytest.mark.parametrize("fragment", sorted(_COUNTERION_SMILES_RAW | _SOLVENT_SMILES_RAW))
def test_every_known_fragment_makes_a_salt_through_the_pipeline(fragment, tautomer_standardisation):
    # Runs the real pipeline (which standardises before classifying), so an
    # entry whose tautomer-canonical form no longer matches would fail here.
    processor = MoleculeProcessor(SettingsConfig(tautomer_standardisation=tautomer_standardisation))
    free_base = processor.process_row("train", "f.csv", 0, "CCN(CC)CC", {})
    salt = processor.process_row("train", "f.csv", 1, f"CCN(CC)CC.{fragment}", {})
    assert salt.is_parsed
    assert salt.is_salt_or_metal is True
    assert salt.parent_smiles == free_base.canonical_smiles
    assert salt.parent_inchikey == free_base.inchikey
    assert [kind for _, kind in salt.stripped_fragments] != ["unknown"]
