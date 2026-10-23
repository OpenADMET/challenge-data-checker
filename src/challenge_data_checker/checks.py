"""Leakage, duplicate, and identifier-namespace consistency checks."""

import logging
from collections import defaultdict
from collections.abc import Callable

from rdkit import DataStructs

from challenge_data_checker.models import MoleculeRecord
from challenge_data_checker.report_types import (
    COARSE_REPRESENTATIONS,
    DuplicateFinding,
    IdentifierOccurrences,
    InternalDuplicates,
    LeakageFinding,
    NamespaceIssues,
    OneIdMultipleStructures,
    OneStructureMultipleIds,
    Representation,
    StructureOccurrences,
    TanimotoFinding,
    UnparseableEntry,
)

logger = logging.getLogger(__name__)

KeyFunc = Callable[[MoleculeRecord], str | None]

# Above this many train x test pairs the all-pairs Tanimoto check gets slow
# (roughly a 10k x 10k split); warn so a long run isn't mistaken for a hang.
TANIMOTO_PAIR_WARNING_THRESHOLD = 100_000_000


def unparseable_entries(records: list[MoleculeRecord]) -> list[UnparseableEntry]:
    """Collect rows whose SMILES could not be parsed or processed.

    Parameters
    ----------
    records : list[MoleculeRecord]
        The records to inspect.

    Returns
    -------
    list[UnparseableEntry]
        A list of location/error dicts, one per unparseable record.
    """
    return [{**r.location(), "error": r.parse_error} for r in records if r.parse_error is not None]


def _index_by(records: list[MoleculeRecord], key_func: KeyFunc) -> dict[str, list[MoleculeRecord]]:
    """Group records by a derived key, dropping records with no key.

    Parameters
    ----------
    records : list[MoleculeRecord]
        The records to index.
    key_func : KeyFunc
        A function mapping a record to its grouping key, or ``None`` to
        exclude the record.

    Returns
    -------
    dict[str, list[MoleculeRecord]]
        A mapping of key to the list of records sharing that key.
    """
    index: dict[str, list[MoleculeRecord]] = defaultdict(list)
    for r in records:
        key = key_func(r)
        if key is not None:
            index[key].append(r)
    return index


def _representation_key_funcs(strip_salts: bool = True) -> dict[Representation, KeyFunc]:
    """Build the key functions for each chemical representation.

    Parameters
    ----------
    strip_salts : bool
        Whether to compare salt-stripped parent compounds. If ``False``,
        the ``parent_inchikey`` key matches nothing.

    Returns
    -------
    dict[Representation, KeyFunc]
        A mapping of representation name (``raw_smiles``,
        ``canonical_smiles``, ``inchikey``, ``parent_inchikey``,
        ``inchikey_connectivity``) to a function extracting that
        representation's value from a record.
    """
    return {
        "raw_smiles": lambda r: r.raw_smiles if r.raw_smiles else None,
        "canonical_smiles": lambda r: r.canonical_smiles,
        "inchikey": lambda r: r.inchikey,
        "parent_inchikey": (lambda r: r.parent_inchikey) if strip_salts else (lambda r: None),
        "inchikey_connectivity": _connectivity_key,
    }


def _connectivity_key(record: MoleculeRecord) -> str | None:
    """Extract the first (connectivity) block of a record's InChIKey.

    Parameters
    ----------
    record : MoleculeRecord
        The record to extract the key from.

    Returns
    -------
    str | None
        The 14-character connectivity block, which ignores
        stereochemistry, or ``None`` if the record has no InChIKey.
    """
    if record.inchikey is None:
        return None
    return record.inchikey.split("-", 1)[0]


def _is_reportable(name: Representation, records: list[MoleculeRecord]) -> bool:
    """Decide whether a group of records sharing a representation is worth reporting.

    Exact representations always are. A coarse (salt-form- or
    stereo-insensitive) group is only reported when its records don't all
    share one full InChIKey; otherwise it would just repeat the exact
    InChIKey match.

    Parameters
    ----------
    name : Representation
        The representation the records were grouped by.
    records : list[MoleculeRecord]
        The records sharing that representation's value.

    Returns
    -------
    bool
        Whether the group should be reported under ``name``.
    """
    if name not in COARSE_REPRESENTATIONS:
        return True
    return len({r.inchikey for r in records}) > 1


def _identifier_key_func(col: str) -> KeyFunc:
    """Build a key function extracting a given identifier column's value.

    Parameters
    ----------
    col : str
        The identifier column name to extract.

    Returns
    -------
    KeyFunc
        A function mapping a record to its value for ``col``.
    """

    def key_func(r: MoleculeRecord) -> str | None:
        return r.identifiers.get(col)

    return key_func


def train_test_leakage(
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    strip_salts: bool = True,
) -> dict[str, list[LeakageFinding]]:
    """Find structures present in both pools, per representation.

    Exact representations (raw SMILES, canonical SMILES, InChIKey) report
    every shared value. The coarse representations only report values
    whose train and test records don't all share one full InChIKey:
    ``parent_inchikey`` finds a salt and its free base (or another salt
    form), and ``inchikey_connectivity`` finds stereoisomers and racemate /
    enantiomer pairs.

    Parameters
    ----------
    train_records : list[MoleculeRecord]
        Processed records from the training pool.
    test_records : list[MoleculeRecord]
        Processed records from the test pool.
    strip_salts : bool
        Whether to compare salt-stripped parent compounds.

    Returns
    -------
    dict[str, list[LeakageFinding]]
        A mapping of representation name to a list of leakage findings,
        each with the shared value and its train/test occurrences.
    """
    result: dict[str, list[LeakageFinding]] = {}
    for name, key_func in _representation_key_funcs(strip_salts).items():
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
            if _is_reportable(name, train_index[value] + test_index[value])
        ]
    return result


def identifier_leakage(
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    identifier_columns: list[str],
) -> dict[str, list[LeakageFinding]]:
    """Find identifier values (within the same column) present in both pools.

    Parameters
    ----------
    train_records : list[MoleculeRecord]
        Processed records from the training pool.
    test_records : list[MoleculeRecord]
        Processed records from the test pool.
    identifier_columns : list[str]
        Identifier column names to check.

    Returns
    -------
    dict[str, list[LeakageFinding]]
        A mapping of identifier column name to a list of leakage
        findings, each with the shared identifier value and its
        train/test occurrences.
    """
    result: dict[str, list[LeakageFinding]] = {}
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


def _duplicates_by(
    records: list[MoleculeRecord],
    key_func: KeyFunc,
    representation: Representation | None = None,
) -> list[DuplicateFinding]:
    """Find key values shared by more than one record.

    Parameters
    ----------
    records : list[MoleculeRecord]
        The records to inspect.
    key_func : KeyFunc
        A function mapping a record to the value to compare, or ``None``
        to exclude the record.
    representation : Representation | None
        The chemical representation ``key_func`` extracts, used to skip
        stereo-insensitive groups that only repeat an exact match (see
        ``_is_reportable``), or ``None`` for a non-chemical key such as
        an identifier column.

    Returns
    -------
    list[DuplicateFinding]
        One finding per duplicated value, sorted by value, listing all of
        its occurrences.
    """
    index = _index_by(records, key_func)
    return [
        {"value": value, "occurrences": [r.location() for r in occs]}
        for value, occs in sorted(index.items())
        if len(occs) > 1 and (representation is None or _is_reportable(representation, occs))
    ]


def internal_duplicates(
    records: list[MoleculeRecord], identifier_columns: list[str], strip_salts: bool = True
) -> InternalDuplicates:
    """Find values that appear more than once within a single pool.

    Parameters
    ----------
    records : list[MoleculeRecord]
        Processed records from a single pool (train-only or test-only).
    identifier_columns : list[str]
        Identifier column names to check.
    strip_salts : bool
        Whether to compare salt-stripped parent compounds.

    Returns
    -------
    InternalDuplicates
        A mapping with one entry per representation (``raw_smiles``,
        ``canonical_smiles``, ``inchikey``, ``parent_inchikey``,
        ``inchikey_connectivity``) plus an ``"identifiers"`` entry
        mapping each identifier column to its own duplicate findings.
        Each finding lists the duplicated value and all of its
        occurrences.
    """
    key_funcs = _representation_key_funcs(strip_salts)

    def dupes(name: Representation) -> list[DuplicateFinding]:
        return _duplicates_by(records, key_funcs[name], name)

    return {
        "raw_smiles": dupes("raw_smiles"),
        "canonical_smiles": dupes("canonical_smiles"),
        "inchikey": dupes("inchikey"),
        "parent_inchikey": dupes("parent_inchikey"),
        "inchikey_connectivity": dupes("inchikey_connectivity"),
        "identifiers": {
            col: _duplicates_by(records, _identifier_key_func(col)) for col in identifier_columns
        },
    }


def identifier_namespace_issues(
    all_records: list[MoleculeRecord], identifier_columns: list[str]
) -> dict[str, NamespaceIssues]:
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
    all_records : list[MoleculeRecord]
        Processed records from both the training and test pools, combined.
    identifier_columns : list[str]
        Identifier column names to check.

    Returns
    -------
    dict[str, NamespaceIssues]
        A mapping of identifier column name to a dict with
        ``one_id_multiple_structures`` and ``one_structure_multiple_ids``
        finding lists.
    """
    result: dict[str, NamespaceIssues] = {}
    for col in identifier_columns:
        by_id: dict[str, list[MoleculeRecord]] = defaultdict(list)
        by_structure: dict[tuple[str | None, str | None], list[MoleculeRecord]] = defaultdict(list)
        for r in all_records:
            id_value = r.identifiers.get(col)
            if id_value is None or not r.is_parsed:
                continue
            by_id[id_value].append(r)
            by_structure[(r.canonical_smiles, r.inchikey)].append(r)

        one_id_multiple_structures: list[OneIdMultipleStructures] = []
        for id_value, recs in sorted(by_id.items()):
            structures = {(r.canonical_smiles, r.inchikey) for r in recs}
            if len(structures) > 1:
                structure_entries: list[StructureOccurrences] = [
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
                ]
                one_id_multiple_structures.append(
                    {"identifier": id_value, "structures": structure_entries}
                )

        one_structure_multiple_ids: list[OneStructureMultipleIds] = []
        for (cs, ik), recs in sorted(
            by_structure.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")
        ):
            id_values = {r.identifiers.get(col) for r in recs}
            if len(id_values) > 1:
                identifier_entries: list[IdentifierOccurrences] = [
                    {
                        "identifier": id_value,
                        "occurrences": [
                            r.location() for r in recs if r.identifiers.get(col) == id_value
                        ],
                    }
                    for id_value in sorted(v for v in id_values if v is not None)
                ]
                one_structure_multiple_ids.append(
                    {"canonical_smiles": cs, "inchikey": ik, "identifiers": identifier_entries}
                )

        result[col] = {
            "one_id_multiple_structures": one_id_multiple_structures,
            "one_structure_multiple_ids": one_structure_multiple_ids,
        }
    return result


def _already_matched(a: MoleculeRecord, b: MoleculeRecord, strip_salts: bool) -> bool:
    """Whether two records already match on connectivity or (optionally) parent compound.

    Parameters
    ----------
    a : MoleculeRecord
        The first record.
    b : MoleculeRecord
        The second record.
    strip_salts : bool
        Whether a shared parent compound counts as a match.

    Returns
    -------
    bool
        ``True`` if the records share an InChIKey connectivity block, or
        ``strip_salts`` is on and they share a parent InChIKey.
    """
    connectivity = _connectivity_key(a)
    if connectivity is not None and connectivity == _connectivity_key(b):
        return True
    return strip_salts and a.parent_inchikey is not None and a.parent_inchikey == b.parent_inchikey


def tanimoto_leakage(
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    threshold: float,
    strip_salts: bool = True,
) -> list[TanimotoFinding]:
    """Compute bulk pairwise Tanimoto similarity between train and test fingerprints.

    Flags pairs with similarity >= threshold as possible data leakage,
    excluding pairs that ``train_test_leakage`` already reports: those
    sharing an InChIKey connectivity block (an exact InChIKey match, or
    only the stereochemistry differs) and, with ``strip_salts``, those
    sharing a parent compound. Uses RDKit's ``BulkTanimotoSimilarity`` to
    avoid nested Python loops.

    Parameters
    ----------
    train_records : list[MoleculeRecord]
        Processed records from the training pool.
    test_records : list[MoleculeRecord]
        Processed records from the test pool.
    threshold : float
        Minimum Tanimoto similarity (inclusive) to flag a pair.
    strip_salts : bool
        Whether parent-compound matches are reported elsewhere, and so
        skipped here.

    Returns
    -------
    list[TanimotoFinding]
        A list of findings, each with the train/test locations and their
        similarity, sorted by descending similarity.
    """
    train_valid = [r for r in train_records if r.fingerprint is not None]
    test_valid = [r for r in test_records if r.fingerprint is not None]
    if not train_valid or not test_valid:
        return []

    n_pairs = len(train_valid) * len(test_valid)
    if n_pairs > TANIMOTO_PAIR_WARNING_THRESHOLD:
        logger.warning(
            "Computing Tanimoto similarity for %d train x %d test pairs (%d total); "
            "this may take a while.",
            len(train_valid),
            len(test_valid),
            n_pairs,
        )

    test_fps = [r.fingerprint for r in test_valid]
    findings: list[TanimotoFinding] = []
    for train_rec in train_valid:
        similarities = DataStructs.BulkTanimotoSimilarity(train_rec.fingerprint, test_fps)
        for test_rec, similarity in zip(test_valid, similarities, strict=True):
            if similarity < threshold:
                continue
            if _already_matched(train_rec, test_rec, strip_salts):
                continue  # already reported by train_test_leakage
            findings.append(
                {
                    "train": train_rec.location(),
                    "test": test_rec.location(),
                    "similarity": similarity,
                }
            )
    findings.sort(key=lambda f: f["similarity"], reverse=True)
    return findings
