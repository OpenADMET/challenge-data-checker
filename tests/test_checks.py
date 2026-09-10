from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from challenge_data_checker.checks import (
    identifier_leakage,
    identifier_namespace_issues,
    internal_duplicates,
    tanimoto_leakage,
    train_test_leakage,
    unparseable_entries,
)
from challenge_data_checker.models import MoleculeRecord

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def fp(smiles):
    return _FP_GEN.GetFingerprint(Chem.MolFromSmiles(smiles))


def make_record(
    pool,
    source_file,
    row_index,
    raw_smiles,
    canonical_smiles=None,
    inchikey=None,
    identifiers=None,
    fingerprint=None,
    parse_error=None,
):
    return MoleculeRecord(
        pool=pool,
        source_file=source_file,
        row_index=row_index,
        raw_smiles=raw_smiles,
        identifiers=identifiers or {},
        parse_error=parse_error,
        canonical_smiles=canonical_smiles if not parse_error else None,
        inchikey=inchikey if not parse_error else None,
        fingerprint=fingerprint if not parse_error else None,
    )


def test_unparseable_entries():
    records = [
        make_record("train", "f.csv", 0, "CCO", "CCO", "IK1"),
        make_record("train", "f.csv", 1, "garbage(((", parse_error="RDKit failed to parse SMILES"),
    ]
    result = unparseable_entries(records)
    assert len(result) == 1
    assert result[0]["row_index"] == 1
    assert result[0]["error"] == "RDKit failed to parse SMILES"


def test_train_test_leakage_detects_shared_inchikey():
    train = [make_record("train", "train.csv", 0, "CCO", "CCO", "IK_ETHANOL")]
    test = [make_record("test", "test.csv", 0, "CCO", "CCO", "IK_ETHANOL")]
    result = train_test_leakage(train, test)
    assert len(result["inchikey"]) == 1
    assert result["inchikey"][0]["value"] == "IK_ETHANOL"
    assert len(result["raw_smiles"]) == 1
    assert len(result["canonical_smiles"]) == 1


def test_train_test_leakage_no_overlap():
    train = [make_record("train", "train.csv", 0, "CCO", "CCO", "IK_ETHANOL")]
    test = [make_record("test", "test.csv", 0, "CCN", "CCN", "IK_ETHYLAMINE")]
    result = train_test_leakage(train, test)
    assert result["inchikey"] == []
    assert result["raw_smiles"] == []


def test_identifier_leakage_detects_shared_id_across_pools():
    train = [make_record("train", "train.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"})]
    test = [make_record("test", "test.csv", 0, "CCN", "CCN", "IK2", {"compound_id": "C1"})]
    result = identifier_leakage(train, test, ["compound_id"])
    assert len(result["compound_id"]) == 1
    assert result["compound_id"][0]["value"] == "C1"


def test_internal_duplicates_within_train_split():
    records = [
        make_record("train", "train.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "C1"}),
        make_record("train", "train.csv", 1, "CCO", "CCO", "IK1", {"compound_id": "C2"}),
    ]
    result = internal_duplicates(records, ["compound_id"])
    assert len(result["inchikey"]) == 1
    assert result["inchikey"][0]["value"] == "IK1"
    assert len(result["inchikey"][0]["occurrences"]) == 2
    assert result["identifiers"]["compound_id"] == []  # different ids, no id-level dup


def test_internal_duplicates_none_when_all_unique():
    records = [
        make_record("train", "train.csv", 0, "CCO", "CCO", "IK1"),
        make_record("train", "train.csv", 1, "CCN", "CCN", "IK2"),
    ]
    result = internal_duplicates(records, [])
    assert result["inchikey"] == []


def test_identifier_namespace_one_id_multiple_structures():
    records = [
        make_record("train", "f.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "SAME_ID"}),
        make_record("test", "f.csv", 0, "CCN", "CCN", "IK2", {"compound_id": "SAME_ID"}),
    ]
    result = identifier_namespace_issues(records, ["compound_id"])
    issues = result["compound_id"]["one_id_multiple_structures"]
    assert len(issues) == 1
    assert issues[0]["identifier"] == "SAME_ID"
    assert len(issues[0]["structures"]) == 2


def test_identifier_namespace_one_structure_multiple_ids():
    records = [
        make_record("train", "f.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "ID_A"}),
        make_record("test", "f.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "ID_B"}),
    ]
    result = identifier_namespace_issues(records, ["compound_id"])
    issues = result["compound_id"]["one_structure_multiple_ids"]
    assert len(issues) == 1
    assert {i["identifier"] for i in issues[0]["identifiers"]} == {"ID_A", "ID_B"}


def test_identifier_namespace_no_issues_when_consistent():
    records = [
        make_record("train", "f.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "ID_A"}),
        make_record("test", "f.csv", 0, "CCO", "CCO", "IK1", {"compound_id": "ID_A"}),
        make_record("train", "f.csv", 1, "CCN", "CCN", "IK2", {"compound_id": "ID_B"}),
    ]
    result = identifier_namespace_issues(records, ["compound_id"])
    assert result["compound_id"]["one_id_multiple_structures"] == []
    assert result["compound_id"]["one_structure_multiple_ids"] == []


def test_tanimoto_leakage_flags_near_duplicate_below_exact_match():
    train = [
        make_record("train", "f.csv", 0, "CCCCO", "CCCCO", "IK_BUTANOL", fingerprint=fp("CCCCO"))
    ]
    test = [
        make_record("test", "f.csv", 0, "CCCCC", "CCCCC", "IK_PENTANE", fingerprint=fp("CCCCC"))
    ]
    result = tanimoto_leakage(train, test, threshold=0.3)
    assert len(result) == 1
    assert result[0]["train"]["row_index"] == 0
    assert result[0]["test"]["row_index"] == 0
    assert 0.3 <= result[0]["similarity"] <= 1.0


def test_tanimoto_leakage_excludes_exact_inchikey_matches():
    train = [make_record("train", "f.csv", 0, "CCO", "CCO", "IK_SAME", fingerprint=fp("CCO"))]
    test = [make_record("test", "f.csv", 0, "CCO", "CCO", "IK_SAME", fingerprint=fp("CCO"))]
    result = tanimoto_leakage(train, test, threshold=1.0)
    assert result == []  # already covered by exact InChIKey leakage, not double-reported


def test_tanimoto_leakage_catches_stereo_mismatch_at_default_threshold():
    # Same flat structure, different stereo descriptor -> different InChIKey,
    # identical stereo-blind fingerprint.
    r_ala = fp("C[C@H](N)C(=O)O")
    s_ala = fp("C[C@@H](N)C(=O)O")
    train = [
        make_record(
            "train", "f.csv", 0, "C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O", "IK_R", fingerprint=r_ala
        )
    ]
    test = [
        make_record(
            "test", "f.csv", 0, "C[C@@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "IK_S", fingerprint=s_ala
        )
    ]
    result = tanimoto_leakage(train, test, threshold=1.0)
    assert len(result) == 1
    assert result[0]["similarity"] == 1.0


def test_tanimoto_leakage_below_threshold_not_flagged():
    train = [make_record("train", "f.csv", 0, "CCO", "CCO", "IK1", fingerprint=fp("CCO"))]
    test = [
        make_record("test", "f.csv", 0, "c1ccccc1", "c1ccccc1", "IK2", fingerprint=fp("c1ccccc1"))
    ]
    result = tanimoto_leakage(train, test, threshold=0.9)
    assert result == []
