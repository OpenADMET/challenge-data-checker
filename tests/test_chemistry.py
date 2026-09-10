from rdkit import DataStructs

from challenge_data_checker.chemistry import MoleculeProcessor, parse_smiles
from challenge_data_checker.config import SettingsConfig


def make_processor(**overrides):
    settings = SettingsConfig(**overrides)
    return MoleculeProcessor(settings)


def test_parse_smiles_valid():
    mol, error = parse_smiles("CCO")
    assert mol is not None
    assert error is None


def test_parse_smiles_invalid():
    mol, error = parse_smiles("not_a_smiles(((")
    assert mol is None
    assert error is not None


def test_parse_smiles_empty_string():
    mol, error = parse_smiles("")
    assert mol is None
    assert "Empty" in error


def test_process_row_success_fields():
    processor = make_processor()
    record = processor.process_row("train", "f.csv", 0, "CCO", {"compound_id": "C1"})
    assert record.parse_error is None
    assert record.canonical_smiles == "CCO"
    assert record.inchikey is not None
    assert record.fingerprint is not None
    assert record.heavy_atom_count == 3
    assert record.num_fragments == 1


def test_process_row_unparseable():
    processor = make_processor()
    record = processor.process_row("train", "f.csv", 0, "garbage(((", {})
    assert record.parse_error is not None
    assert not record.is_parsed


def test_fingerprint_ignores_stereochemistry():
    processor = make_processor()
    # (R)- and (S)-alanine: same graph, different stereocentre.
    r_ala = processor.process_row("train", "f.csv", 0, "C[C@H](N)C(=O)O", {})
    s_ala = processor.process_row("train", "f.csv", 1, "C[C@@H](N)C(=O)O", {})

    assert r_ala.inchikey != s_ala.inchikey  # InChIKey still distinguishes stereo
    similarity = DataStructs.TanimotoSimilarity(r_ala.fingerprint, s_ala.fingerprint)
    assert similarity == 1.0  # fingerprint is stereo-blind


def test_tautomer_standardisation_canonicalises_when_enabled():
    processor_on = make_processor(tautomer_standardisation=True)
    processor_off = make_processor(tautomer_standardisation=False)

    keto = processor_on.process_row("train", "f.csv", 0, "CC(=O)CC(=O)C", {})
    # enol/keto tautomer pair should standardise to the same canonical form when enabled
    enol = processor_on.process_row("train", "f.csv", 1, "CC(O)=CC(=O)C", {})
    assert keto.canonical_smiles == enol.canonical_smiles

    keto_off = processor_off.process_row("train", "f.csv", 0, "CC(=O)CC(=O)C", {})
    enol_off = processor_off.process_row("train", "f.csv", 1, "CC(O)=CC(=O)C", {})
    assert keto_off.canonical_smiles != enol_off.canonical_smiles
