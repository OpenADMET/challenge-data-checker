import pytest
from rdkit import Chem

from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import SettingsConfig
from challenge_data_checker.quality_filters import (
    _SMALL_MOLECULE_ALLOWLIST_RAW,
    COUNTERION_CANONICAL_SMILES,
    SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES,
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


def test_allowlist_and_counterion_set_are_disjoint():
    assert not SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES & COUNTERION_CANONICAL_SMILES
