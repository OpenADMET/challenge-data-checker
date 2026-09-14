"""Leakage, duplicate, and identifier-namespace consistency checks."""

from collections import defaultdict
from typing import Any, Callable

from rdkit import DataStructs

from challenge_data_checker.models import MoleculeRecord

KeyFunc = Callable[[MoleculeRecord], str | None]


def unparseable_entries(records: list[MoleculeRecord]) -> list[dict]:
    """Collect rows whose SMILES could not be parsed or processed.

    Parameters
    ----------
    records
        The records to inspect.

    Returns
    -------
    A list of location/error dicts, one per unparseable record.
    """
    return [{**r.location(), "error": r.parse_error} for r in records if not r.is_parsed]


def _index_by(records: list[MoleculeRecord], key_func: KeyFunc) -> dict[str, list[MoleculeRecord]]:
    """Group records by a derived key, dropping records with no key.

    Parameters
    ----------
    records
        The records to index.
    key_func
        A function mapping a record to its grouping key, or ``None`` to
        exclude the record.

    Returns
    -------
    A mapping of key to the list of records sharing that key.
    """
    index: dict[str, list[MoleculeRecord]] = defaultdict(list)
    for r in records:
        key = key_func(r)
        if key is not None:
            index[key].append(r)
    return index


def _representation_key_funcs() -> dict[str, KeyFunc]:
    """Build the key functions for each chemical representation.

    Returns
    -------
    A mapping of representation name (``raw_smiles``, ``canonical_smiles``,
    ``inchikey``) to a function extracting that representation's value
    from a record.
    """
    return {
        "raw_smiles": lambda r: r.raw_smiles if r.raw_smiles else None,
        "canonical_smiles": lambda r: r.canonical_smiles,
        "inchikey": lambda r: r.inchikey,
    }


def _identifier_key_func(col: str) -> KeyFunc:
    """Build a key function extracting a given identifier column's value.

    Parameters
    ----------
    col
        The identifier column name to extract.

    Returns
    -------
    A function mapping a record to its value for ``col``.
    """

    def key_func(r: MoleculeRecord) -> str | None:
        return r.identifiers.get(col)

    return key_func


def train_test_leakage(
    train_records: list[MoleculeRecord], test_records: list[MoleculeRecord]
) -> dict[str, list[dict]]:
    """Find raw SMILES / canonical SMILES / InChIKey values present in both pools.

    Parameters
    ----------
    train_records
        Processed records from the training pool.
    test_records
        Processed records from the test pool.

    Returns
    -------
    A mapping of representation name to a list of leakage findings, each
    with the shared value and its train/test occurrences.
    """
    result: dict[str, list[dict]] = {}
    for name, key_func in _representation_key_funcs().items():
        train_index = _index_by(train_records, key_func)
        test_index = _index_by(test_records, key_func)
        shared = sorted(set(train_index) & set(test_index))
        result[name] = [
            {
                "value": value,
                "train_occurrences": [r.location() for r in train_index[value]],
                "test_occurrences": [r.location() for r in test_index[value]],
            }
            for value in shared
        ]
    return result


def identifier_leakage(
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    identifier_columns: list[str],
) -> dict[str, list[dict]]:
    """Find identifier values (within the same column) present in both pools.

    Parameters
    ----------
    train_records
        Processed records from the training pool.
    test_records
        Processed records from the test pool.
    identifier_columns
        Identifier column names to check.

    Returns
    -------
    A mapping of identifier column name to a list of leakage findings,
    each with the shared identifier value and its train/test occurrences.
    """
    result: dict[str, list[dict]] = {}
    for col in identifier_columns:
        key_func = _identifier_key_func(col)
        train_index = _index_by(train_records, key_func)
        test_index = _index_by(test_records, key_func)
        shared = sorted(set(train_index) & set(test_index))
        result[col] = [
            {
                "value": value,
                "train_occurrences": [r.location() for r in train_index[value]],
                "test_occurrences": [r.location() for r in test_index[value]],
            }
            for value in shared
        ]
    return result


def internal_duplicates(
    records: list[MoleculeRecord], identifier_columns: list[str]
) -> dict[str, Any]:
    """Find values that appear more than once within a single pool.

    Parameters
    ----------
    records
        Processed records from a single pool (train-only or test-only).
    identifier_columns
        Identifier column names to check.

    Returns
    -------
    A mapping with one entry per representation (``raw_smiles``,
    ``canonical_smiles``, ``inchikey``) plus an ``"identifiers"`` entry
    mapping each identifier column to its own duplicate findings. Each
    finding lists the duplicated value and all of its occurrences.
    """
    result: dict[str, Any] = {}
    for name, key_func in _representation_key_funcs().items():
        index = _index_by(records, key_func)
        result[name] = [
            {"value": value, "occurrences": [r.location() for r in occs]}
            for value, occs in sorted(index.items())
            if len(occs) > 1
        ]

    identifiers_result: dict[str, list[dict]] = {}
    for col in identifier_columns:
        key_func = _identifier_key_func(col)
        index = _index_by(records, key_func)
        identifiers_result[col] = [
            {"value": value, "occurrences": [r.location() for r in occs]}
            for value, occs in sorted(index.items())
            if len(occs) > 1
        ]
    result["identifiers"] = identifiers_result
    return result


def identifier_namespace_issues(
    all_records: list[MoleculeRecord], identifier_columns: list[str]
) -> dict[str, dict[str, list[dict]]]:
    """Check identifier-to-structure consistency within each identifier column.

    For each identifier column, checked across the combined train+test pool:
        - ``one_id_multiple_structures``: a single identifier string maps to
          2+ structurally distinct molecules (differing canonical SMILES
          and/or InChIKey).
        - ``one_structure_multiple_ids``: a single structure (same canonical
          SMILES AND InChIKey) is given 2+ distinct identifier strings within
          that column.

    Parameters
    ----------
    all_records
        Processed records from both the training and test pools, combined.
    identifier_columns
        Identifier column names to check.

    Returns
    -------
    A mapping of identifier column name to a dict with
    ``one_id_multiple_structures`` and ``one_structure_multiple_ids``
    finding lists.
    """
    result: dict[str, dict[str, list[dict]]] = {}
    for col in identifier_columns:
        by_id: dict[str, list[MoleculeRecord]] = defaultdict(list)
        by_structure: dict[tuple[str | None, str | None], list[MoleculeRecord]] = defaultdict(list)
        for r in all_records:
            id_value = r.identifiers.get(col)
            if id_value is None or not r.is_parsed:
                continue
            by_id[id_value].append(r)
            by_structure[(r.canonical_smiles, r.inchikey)].append(r)

        one_id_multiple_structures = []
        for id_value, recs in sorted(by_id.items()):
            structures = {(r.canonical_smiles, r.inchikey) for r in recs}
            if len(structures) > 1:
                one_id_multiple_structures.append(
                    {
                        "identifier": id_value,
                        "structures": [
                            {
                                "canonical_smiles": cs,
                                "inchikey": ik,
                                "occurrences": [
                                    r.location()
                                    for r in recs
                                    if (r.canonical_smiles, r.inchikey) == (cs, ik)
                                ],
                            }
                            for cs, ik in sorted(structures, key=lambda t: (t[0] or "", t[1] or ""))
                        ],
                    }
                )

        one_structure_multiple_ids = []
        for (cs, ik), recs in sorted(
            by_structure.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")
        ):
            id_values = {r.identifiers.get(col) for r in recs}
            if len(id_values) > 1:
                one_structure_multiple_ids.append(
                    {
                        "canonical_smiles": cs,
                        "inchikey": ik,
                        "identifiers": [
                            {
                                "identifier": id_value,
                                "occurrences": [
                                    r.location() for r in recs if r.identifiers.get(col) == id_value
                                ],
                            }
                            for id_value in sorted(v for v in id_values if v is not None)
                        ],
                    }
                )

        result[col] = {
            "one_id_multiple_structures": one_id_multiple_structures,
            "one_structure_multiple_ids": one_structure_multiple_ids,
        }
    return result


def tanimoto_leakage(
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    threshold: float,
) -> list[dict]:
    """Compute bulk pairwise Tanimoto similarity between train and test fingerprints.

    Flags pairs with similarity >= threshold as possible data leakage,
    excluding pairs that already share an InChIKey (those are reported as
    exact leakage elsewhere). Uses RDKit's ``BulkTanimotoSimilarity`` to avoid
    nested Python loops.

    Parameters
    ----------
    train_records
        Processed records from the training pool.
    test_records
        Processed records from the test pool.
    threshold
        Minimum Tanimoto similarity (inclusive) to flag a pair.

    Returns
    -------
    A list of findings, each with the train/test locations and their
    similarity, sorted by descending similarity.
    """
    train_valid = [r for r in train_records if r.fingerprint is not None]
    test_valid = [r for r in test_records if r.fingerprint is not None]
    if not train_valid or not test_valid:
        return []

    test_fps = [r.fingerprint for r in test_valid]
    findings = []
    for train_rec in train_valid:
        similarities = DataStructs.BulkTanimotoSimilarity(train_rec.fingerprint, test_fps)
        for test_rec, similarity in zip(test_valid, similarities):
            if similarity < threshold:
                continue
            if train_rec.inchikey is not None and train_rec.inchikey == test_rec.inchikey:
                continue  # already reported as exact InChIKey leakage
            findings.append(
                {
                    "train": train_rec.location(),
                    "test": test_rec.location(),
                    "similarity": similarity,
                }
            )
    findings.sort(key=lambda f: f["similarity"], reverse=True)
    return findings
