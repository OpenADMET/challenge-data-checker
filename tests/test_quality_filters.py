from rdkit import Chem

from challenge_data_checker.quality_filters import (
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
