"""Leakage, duplicate, and identifier-namespace consistency checks."""

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TypedDict

from rdkit import Chem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize

from challenge_data_checker.models import Fingerprint, MoleculeRecord
from challenge_data_checker.report_types import (
    COARSE_REPRESENTATIONS,
    DuplicateFinding,
    IdConflictType,
    IdentifierOccurrences,
    IdentifierWhitespaceEntry,
    IdWithUnparseableAndValidStructures,
    InternalDuplicates,
    LeakageFinding,
    MissingStructureResolvableById,
    NamespaceIssues,
    NearDuplicateIdentifiers,
    OneIdMultipleStructures,
    OneStructureMultipleIds,
    Representation,
    SharedStructureType,
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
    return [
        {
            **r.location(),
            "error": r.parse_error,
            "category": r.parse_error_category or "other",
            "resolvable_via_identifier": None,
        }
        for r in records
        if r.parse_error is not None
    ]


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
    all_records: list[MoleculeRecord], identifier_columns: list[str], strip_salts: bool = True
) -> dict[str, NamespaceIssues]:
    """Check identifier-to-structure consistency within each identifier column.

    For each identifier column, checked across the combined train+test pool
    and on normalised identifier values (see
    ``MoleculeProcessor.normalise_identifier``):
        - ``one_id_multiple_structures``: a single identifier string maps to
          2+ structurally distinct molecules (differing canonical SMILES
          and/or InChIKey), with a ``conflict_type`` saying how they differ
          (see ``_id_conflict_type``).
        - ``one_structure_multiple_ids``: a single compound (same parent
          connectivity, see ``_compound_key``) is given 2+ distinct
          identifier strings within that column, with a ``conflict_type``
          saying how its structures relate.
        - ``identifier_whitespace``: rows whose raw value had surrounding
          whitespace (stripped before every comparison).
        - ``near_duplicates``: distinct values that only differ in case or
          ``-``/``_``.
        - ``id_with_unparseable_and_valid_structures``: an identifier with a
          parsed structure in some rows and a broken (non-empty) SMILES in
          others.
        - ``missing_structure_resolvable_by_id``: an identifier with an empty
          SMILES in some rows and a parsed structure in others, listing the
          candidate structures (ambiguous if more than one).

    The first two only consider rows whose SMILES parsed.

    Parameters
    ----------
    all_records : list[MoleculeRecord]
        Processed records from both the training and test pools, combined.
    identifier_columns : list[str]
        Identifier column names to check.
    strip_salts : bool
        Whether a shared salt-stripped parent counts as the same compound.

    Returns
    -------
    dict[str, NamespaceIssues]
        A mapping of identifier column name to its finding lists.
    """
    result: dict[str, NamespaceIssues] = {}
    for col in identifier_columns:
        by_id: dict[str, list[MoleculeRecord]] = defaultdict(list)
        by_compound: dict[str, list[MoleculeRecord]] = defaultdict(list)
        for r in all_records:
            id_value = r.identifiers.get(col)
            if id_value is None or not r.is_parsed:
                continue
            by_id[id_value].append(r)
            compound = _compound_key(r, strip_salts)
            if compound is not None:
                by_compound[compound].append(r)

        one_id_multiple_structures: list[OneIdMultipleStructures] = [
            {
                "identifier": id_value,
                "conflict_type": _id_conflict_type(recs, strip_salts),
                "max_similarity": _max_similarity(recs),
                "structures": _structures(recs),
            }
            for id_value, recs in sorted(by_id.items())
            if len({(r.canonical_smiles, r.inchikey) for r in recs}) > 1
        ]

        one_structure_multiple_ids: list[OneStructureMultipleIds] = []
        for _, recs in sorted(by_compound.items()):
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
                    {
                        "conflict_type": _shared_structure_type(recs, strip_salts),
                        "canonical_smiles": recs[0].canonical_smiles,
                        "inchikey": recs[0].inchikey,
                        "identifiers": identifier_entries,
                    }
                )

        result[col] = {
            "one_id_multiple_structures": one_id_multiple_structures,
            "one_structure_multiple_ids": one_structure_multiple_ids,
            "identifier_whitespace": _identifier_whitespace(all_records, col),
            "near_duplicates": _near_duplicate_identifiers(all_records, col),
            **_unparsed_rows_by_identifier(all_records, col, by_id),
        }
    return result


def _salt_level_key(record: MoleculeRecord, strip_salts: bool) -> str | None:
    """The InChIKey a record is compared on, salt form ignored if ``strip_salts``.

    Parameters
    ----------
    record : MoleculeRecord
        A parsed record.
    strip_salts : bool
        Whether to use the salt-stripped parent's InChIKey.

    Returns
    -------
    str | None
        The parent InChIKey (``strip_salts``, falling back to the record's
        own if no parent was computed) or the record's own.
    """
    if strip_salts and record.parent_inchikey is not None:
        return record.parent_inchikey
    return record.inchikey


def _compound_key(record: MoleculeRecord, strip_salts: bool) -> str | None:
    """Group records that are the same compound, ignoring salt form and stereo.

    Parameters
    ----------
    record : MoleculeRecord
        A parsed record.
    strip_salts : bool
        Whether to ignore salt form (use the parent compound).

    Returns
    -------
    str | None
        The connectivity block of the (parent) InChIKey.
    """
    key = _salt_level_key(record, strip_salts)
    return key.split("-", 1)[0] if key else None


_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


def _tautomer_key(record: MoleculeRecord, strip_salts: bool) -> str | None:
    """The connectivity of a record's tautomer-canonical (parent) structure.

    Only computed for records involved in an identifier conflict, so the
    tautomer canonicalisation cost stays small.

    Parameters
    ----------
    record : MoleculeRecord
        A parsed record.
    strip_salts : bool
        Whether to canonicalise the salt-stripped parent rather than the
        structure as drawn.

    Returns
    -------
    str | None
        The InChIKey connectivity block of the tautomer-canonicalised
        structure, or ``None`` if it can't be computed.
    """
    smiles = record.parent_smiles if strip_salts and record.parent_smiles else None
    mol = Chem.MolFromSmiles(smiles or record.canonical_smiles or "")
    if mol is None:
        return None
    key = Chem.MolToInchiKey(_TAUTOMER_ENUMERATOR.Canonicalize(mol))
    return key.split("-", 1)[0] if key else None


def _all_equal(values: Iterable[str | None]) -> bool:
    """Whether every value is the same and not ``None``.

    Parameters
    ----------
    values : Iterable[str | None]
        The values to compare.

    Returns
    -------
    bool
        ``True`` if there is exactly one distinct value and it isn't
        ``None``.
    """
    distinct = set(values)
    return len(distinct) == 1 and None not in distinct


def _id_conflict_type(records: list[MoleculeRecord], strip_salts: bool) -> IdConflictType:
    """Classify how the structures given one identifier differ.

    Uses the finest comparison under which all of them are the same:
    identical InChIKey (only the drawing differs, e.g. a mobile-H tautomer)
    or tautomer-canonical parent -> ``tautomer``; same salt-stripped parent
    -> ``salt_form``; same parent connectivity -> ``stereo`` (salt form may
    differ too); otherwise ``different_compound``.

    Parameters
    ----------
    records : list[MoleculeRecord]
        The parsed records sharing the identifier.
    strip_salts : bool
        Whether salt-form differences can be recognised.

    Returns
    -------
    IdConflictType
        The kind of conflict.
    """
    if _all_equal(r.inchikey for r in records):
        return "tautomer"
    if strip_salts and _all_equal(r.parent_inchikey for r in records):
        return "salt_form"
    if _all_equal(_compound_key(r, strip_salts) for r in records):
        return "stereo"
    if _all_equal(_tautomer_key(r, strip_salts) for r in records):
        return "tautomer"
    return "different_compound"


def _shared_structure_type(records: list[MoleculeRecord], strip_salts: bool) -> SharedStructureType:
    """Classify how the structures of one compound given several identifiers relate.

    Parameters
    ----------
    records : list[MoleculeRecord]
        Parsed records sharing a compound key (see ``_compound_key``).
    strip_salts : bool
        Whether salt-form differences can be recognised.

    Returns
    -------
    SharedStructureType
        ``same_structure`` if they're all identical, ``salt_form`` if they
        share a parent, otherwise ``stereo``.
    """
    if _all_equal(f"{r.canonical_smiles}|{r.inchikey}" for r in records):
        return "same_structure"
    if strip_salts and _all_equal(r.parent_inchikey for r in records):
        return "salt_form"
    return "stereo"


def _max_similarity(records: list[MoleculeRecord]) -> float | None:
    """Highest Tanimoto similarity between the distinct structures among records.

    Parameters
    ----------
    records : list[MoleculeRecord]
        Parsed records.

    Returns
    -------
    float | None
        The maximum pairwise similarity between one representative of each
        distinct structure, or ``None`` if fewer than two have
        fingerprints.
    """
    representatives: dict[tuple[str | None, str | None], Fingerprint] = {}
    for r in records:
        if r.fingerprint is not None:
            representatives.setdefault((r.canonical_smiles, r.inchikey), r.fingerprint)
    fps = list(representatives.values())
    if len(fps) < 2:
        return None
    return max(
        max(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])) for i, fp in enumerate(fps[:-1])
    )


def _structures(records: list[MoleculeRecord]) -> list[StructureOccurrences]:
    """Group parsed records by structure.

    Parameters
    ----------
    records : list[MoleculeRecord]
        Parsed records.

    Returns
    -------
    list[StructureOccurrences]
        One entry per distinct (canonical SMILES, InChIKey), sorted, with
        the rows where it appears.
    """
    structures = {(r.canonical_smiles, r.inchikey) for r in records}
    return [
        {
            "canonical_smiles": cs,
            "inchikey": ik,
            "occurrences": [
                r.location() for r in records if (r.canonical_smiles, r.inchikey) == (cs, ik)
            ],
        }
        for cs, ik in sorted(structures, key=lambda t: (t[0] or "", t[1] or ""))
    ]


class _UnparsedRowFindings(TypedDict):
    """The identifier findings about rows whose SMILES didn't parse."""

    id_with_unparseable_and_valid_structures: list[IdWithUnparseableAndValidStructures]
    missing_structure_resolvable_by_id: list[MissingStructureResolvableById]


def _unparsed_rows_by_identifier(
    all_records: list[MoleculeRecord], col: str, parsed_by_id: dict[str, list[MoleculeRecord]]
) -> _UnparsedRowFindings:
    """Relate rows whose SMILES didn't parse to parsed rows with the same identifier.

    An identifier with a broken SMILES in some rows and a parsed structure
    in others shows which structure was probably intended. An identifier
    with an empty SMILES in some rows and a parsed structure in others
    means the empty rows can be filled in.

    Parameters
    ----------
    all_records : list[MoleculeRecord]
        Records from both pools.
    col : str
        The identifier column to check.
    parsed_by_id : dict[str, list[MoleculeRecord]]
        Parsed records grouped by their (normalised) value for ``col``.

    Returns
    -------
    _UnparsedRowFindings
        The ``id_with_unparseable_and_valid_structures`` and
        ``missing_structure_resolvable_by_id`` findings for ``col``.
    """
    broken: dict[str, list[MoleculeRecord]] = defaultdict(list)
    empty: dict[str, list[MoleculeRecord]] = defaultdict(list)
    for r in all_records:
        id_value = r.identifiers.get(col)
        if id_value is None or r.is_parsed or id_value not in parsed_by_id:
            continue
        (empty if r.parse_error_category == "empty" else broken)[id_value].append(r)

    with_broken: list[IdWithUnparseableAndValidStructures] = [
        {
            "identifier": id_value,
            "structures": _structures(parsed_by_id[id_value]),
            "unparseable": [
                {
                    **r.location(),
                    "error": r.parse_error or "",
                    "category": r.parse_error_category or "other",
                }
                for r in recs
            ],
        }
        for id_value, recs in sorted(broken.items())
    ]
    resolvable: list[MissingStructureResolvableById] = []
    for id_value, recs in sorted(empty.items()):
        candidates = _structures(parsed_by_id[id_value])
        resolvable.append(
            {
                "identifier": id_value,
                "missing": [r.location() for r in recs],
                "candidates": candidates,
                "ambiguous": len(candidates) > 1,
            }
        )
    return {
        "id_with_unparseable_and_valid_structures": with_broken,
        "missing_structure_resolvable_by_id": resolvable,
    }


_NEAR_DUPLICATE_STRIP_RE = re.compile(r"[-_]")


def _near_duplicate_key(value: str) -> str:
    """Reduce an identifier to the form near-duplicates share.

    Parameters
    ----------
    value : str
        A (normalised) identifier value.

    Returns
    -------
    str
        The value casefolded, with ``-`` and ``_`` removed.
    """
    return _NEAR_DUPLICATE_STRIP_RE.sub("", value.casefold())


def _identifier_whitespace(
    all_records: list[MoleculeRecord], col: str
) -> list[IdentifierWhitespaceEntry]:
    """Find rows whose identifier value had surrounding whitespace stripped.

    Parameters
    ----------
    all_records : list[MoleculeRecord]
        Records from both pools, parsed or not.
    col : str
        The identifier column to check.

    Returns
    -------
    list[IdentifierWhitespaceEntry]
        One entry per affected row, with the raw and stripped values.
    """
    entries: list[IdentifierWhitespaceEntry] = []
    for r in all_records:
        raw = r.raw_identifiers.get(col)
        if raw is not None and raw != raw.strip():
            entries.append({**r.location(), "raw_value": raw, "value": raw.strip()})
    return entries


def _near_duplicate_identifiers(
    all_records: list[MoleculeRecord], col: str
) -> list[NearDuplicateIdentifiers]:
    """Group distinct identifier values that only differ in case or ``-``/``_``.

    Such values (e.g. ``Molport-1`` and ``MolPort_1``) are probably the same
    identifier written inconsistently, but might be distinct identifiers in a
    poorly designed scheme, so they're flagged rather than merged.

    Parameters
    ----------
    all_records : list[MoleculeRecord]
        Records from both pools, parsed or not.
    col : str
        The identifier column to check.

    Returns
    -------
    list[NearDuplicateIdentifiers]
        One group per near-duplicate key with two or more distinct
        (normalised) values, listing each value's occurrences.
    """
    by_key: dict[str, dict[str, list[MoleculeRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        value = r.identifiers.get(col)
        if value is not None:
            by_key[_near_duplicate_key(value)][value].append(r)
    return [
        {
            "key": key,
            "identifiers": [
                {"identifier": value, "occurrences": [r.location() for r in recs]}
                for value, recs in sorted(values.items())
            ],
        }
        for key, values in sorted(by_key.items())
        if len(values) > 1
    ]


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
