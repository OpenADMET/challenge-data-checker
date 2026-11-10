"""Match a blinded test set against a public database (``public-search``).

This is the audit's leakage check with a public database standing in for the
training pool: the same representations, the same rule 3 tiering, the same
``Location`` rows. What differs is the second half of the question. A
train/test overlap is leakage on its own, but a test compound merely
appearing in ChEMBL is not: most drug-like compounds appear somewhere. What
makes it leakage is a public record that measures *this challenge's
endpoint*, because then the answer can be read off.

So each match has two axes, and both set the tier (see
``severity.public_finding_type``):

**How the structure matched** - an exact InChIKey, the salt-stripped parent
(so a test free base matches a ChEMBL hydrochloride), or the InChIKey
connectivity block (so undefined or differing stereochemistry still
matches). Salt forms and undefined stereochemistry are the leakage modes
seen in practice, which is why the parent lookup goes through ChEMBL's
``molecule_hierarchy`` rather than comparing InChIKeys as drawn.

**How the endpoint matched** - either the configuration named the ChEMBL
assay or target outright (``explicit_id``), or the target was resolved from
a name or gene symbol and the measurement type is one of the configured
ones (``target_type``), which is weaker and never blocks.

Presence in the public source, with no endpoint match, is counted and
listed but never flagged.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from challenge_data_checker.chembl import CONNECTIVITY_BLOCK_LENGTH, StructureMatch
from challenge_data_checker.models import MoleculeRecord
from challenge_data_checker.report_types import (
    PUBLIC_SEARCH_REPRESENTATIONS,
    ChemblActivity,
    ChemblCompound,
    Location,
    PublicCompoundPresence,
    PublicDocumentSummary,
    PublicLeakageFinding,
    PublicMatchType,
    Representation,
)

logger = logging.getLogger(__name__)

# Strongest first. A structure with both kinds of evidence is reported with
# the stronger one, since the weaker record adds no information.
_MATCH_STRENGTH: dict[PublicMatchType, int] = {"explicit_id": 2, "target_type": 1}

type RowKey = tuple[str, str, int]


def _row_key(location: Location) -> RowKey:
    """Build the identity of a row from its report location.

    Parameters
    ----------
    location : Location
        A row location from the report.

    Returns
    -------
    RowKey
        The row's (pool, source_file, row_index).
    """
    return (location["pool"], location["source_file"], location["row_index"])


def connectivity_block(inchikey: str | None) -> str | None:
    """Take the connectivity (first) block of an InChIKey.

    Parameters
    ----------
    inchikey : str | None
        A full InChIKey, or ``None``.

    Returns
    -------
    str | None
        The 14-character connectivity block, or ``None`` if there is no
        InChIKey.
    """
    if not inchikey:
        return None
    return inchikey[:CONNECTIVITY_BLOCK_LENGTH]


def representation_value(record: MoleculeRecord, representation: Representation) -> str | None:
    """Extract one representation's lookup value from a test record.

    Parameters
    ----------
    record : MoleculeRecord
        A processed test row.
    representation : Representation
        One of ``PUBLIC_SEARCH_REPRESENTATIONS``.

    Returns
    -------
    str | None
        The value to look the record up by, or ``None`` if the record has
        no such value (an unparsed row, or no parent when salt stripping
        is off).
    """
    if representation == "inchikey":
        return record.inchikey
    if representation == "parent_inchikey":
        return record.parent_inchikey
    if representation == "inchikey_connectivity":
        return connectivity_block(record.inchikey)
    return None


def index_records(
    records: Sequence[MoleculeRecord], *, strip_salts: bool, match_connectivity: bool
) -> dict[Representation, dict[str, list[MoleculeRecord]]]:
    """Group the test records by each representation's lookup value.

    Parameters
    ----------
    records : Sequence[MoleculeRecord]
        The processed test records.
    strip_salts : bool
        Whether to look compounds up by their salt-stripped parent as
        well. Mirrors the audit's ``strip_salts`` setting.
    match_connectivity : bool
        Whether to also look compounds up by InChIKey connectivity block,
        which catches differing stereochemistry but is the slowest lookup.

    Returns
    -------
    dict[Representation, dict[str, list[MoleculeRecord]]]
        Per representation, the records behind each distinct value.
    """
    enabled: list[Representation] = ["inchikey"]
    if strip_salts:
        enabled.append("parent_inchikey")
    if match_connectivity:
        enabled.append("inchikey_connectivity")

    index: dict[Representation, dict[str, list[MoleculeRecord]]] = {
        representation: defaultdict(list) for representation in enabled
    }
    for record in records:
        if not record.is_parsed:
            continue
        for representation in enabled:
            value = representation_value(record, representation)
            if value:
                index[representation][value].append(record)
    return {representation: dict(values) for representation, values in index.items()}


def lookup_keys(
    index: Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]],
) -> dict[Representation, list[str]]:
    """Reduce a record index to the key lists the source is queried with.

    Parameters
    ----------
    index : Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]]
        The index built by ``index_records``.

    Returns
    -------
    dict[Representation, list[str]]
        The sorted distinct keys to look up, per representation.
    """
    return {representation: sorted(values) for representation, values in index.items() if values}


def classify_activity(
    activity: ChemblActivity,
    *,
    assay_chembl_ids: frozenset[str],
    target_chembl_ids: frozenset[str],
    resolved_target_chembl_ids: frozenset[str],
    standard_types: frozenset[str],
) -> PublicMatchType | None:
    """Decide whether one activity record measures the challenge endpoint.

    Parameters
    ----------
    activity : ChemblActivity
        The activity to classify.
    assay_chembl_ids : frozenset[str]
        ChEMBL assay IDs named in the configuration.
    target_chembl_ids : frozenset[str]
        ChEMBL target IDs named in the configuration.
    resolved_target_chembl_ids : frozenset[str]
        ChEMBL target IDs resolved from the configured target names or
        gene symbols.
    standard_types : frozenset[str]
        Allowed ``standard_type`` values, compared case-insensitively. An
        empty set matches any type.

    Returns
    -------
    PublicMatchType | None
        ``"explicit_id"`` if the configuration named this assay or target,
        ``"target_type"`` if a resolved target and an allowed measurement
        type both match, otherwise ``None``.
    """
    assay_id = activity["assay_chembl_id"]
    target_id = activity["target_chembl_id"]
    if assay_id is not None and assay_id in assay_chembl_ids:
        return "explicit_id"
    if target_id is not None and target_id in target_chembl_ids:
        return "explicit_id"

    if target_id is None or target_id not in resolved_target_chembl_ids:
        return None
    if not standard_types:
        return "target_type"
    standard_type = activity["standard_type"]
    if standard_type is not None and standard_type.lower() in standard_types:
        return "target_type"
    return None


def _locations(records: Iterable[MoleculeRecord]) -> list[Location]:
    """Build the report locations of some test records, in file order.

    Parameters
    ----------
    records : Iterable[MoleculeRecord]
        The records to locate.

    Returns
    -------
    list[Location]
        Their locations, sorted by source file and row index.
    """
    return [
        record.location() for record in sorted(records, key=lambda r: (r.source_file, r.row_index))
    ]


def group_matches(
    matches: Iterable[StructureMatch],
) -> dict[tuple[Representation, str], list[ChemblCompound]]:
    """Group structure matches by the representation and value that matched.

    Parameters
    ----------
    matches : Iterable[StructureMatch]
        Matches as returned by ``chembl.find_structures``.

    Returns
    -------
    dict[tuple[Representation, str], list[ChemblCompound]]
        The distinct ChEMBL compounds behind each matched key, ordered by
        ``molregno``.
    """
    grouped: dict[tuple[Representation, str], dict[int, ChemblCompound]] = defaultdict(dict)
    for match in matches:
        grouped[(match.representation, match.value)][match.compound["molregno"]] = match.compound
    return {
        key: [compounds[molregno] for molregno in sorted(compounds)]
        for key, compounds in grouped.items()
    }


def build_presence(
    index: Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]],
    grouped: Mapping[tuple[Representation, str], Sequence[ChemblCompound]],
) -> list[PublicCompoundPresence]:
    """List the test structures present in the public source at all.

    Reported as a count and a list, never as a finding: presence without an
    endpoint match is not leakage, and flagging it would drown the real
    findings in false positives.

    Parameters
    ----------
    index : Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]]
        The index built by ``index_records``.
    grouped : Mapping[tuple[Representation, str], Sequence[ChemblCompound]]
        The compounds behind each matched key, from ``group_matches``.

    Returns
    -------
    list[PublicCompoundPresence]
        One entry per matched key, in representation then value order.
    """
    presence: list[PublicCompoundPresence] = []
    for representation in PUBLIC_SEARCH_REPRESENTATIONS:
        values = index.get(representation) or {}
        for value in sorted(values):
            compounds = grouped.get((representation, value))
            if not compounds:
                continue
            presence.append(
                {
                    "value": value,
                    "representation": representation,
                    "test_occurrences": _locations(values[value]),
                    "chembl_ids": sorted(
                        compound["chembl_id"]
                        for compound in compounds
                        if compound["chembl_id"] is not None
                    ),
                }
            )
    return presence


def build_findings(
    index: Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]],
    grouped: Mapping[tuple[Representation, str], Sequence[ChemblCompound]],
    activities_by_molregno: Mapping[int, Sequence[ChemblActivity]],
    *,
    assay_chembl_ids: Iterable[str],
    target_chembl_ids: Iterable[str],
    resolved_target_chembl_ids: Iterable[str],
    standard_types: Iterable[str],
) -> list[PublicLeakageFinding]:
    """Build the endpoint-match findings for every matched test structure.

    One finding per matched key, carrying its strongest evidence: a
    structure with both an explicitly named assay and a weaker
    target-plus-type record is reported once, as the explicit match.

    Representations are walked exact-first, and a coarser representation is
    skipped when its test rows are already covered by an equal-or-stronger
    finding, so one leak seen through the InChIKey, the parent and the
    connectivity block counts once rather than three times.

    Parameters
    ----------
    index : Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]]
        The index built by ``index_records``.
    grouped : Mapping[tuple[Representation, str], Sequence[ChemblCompound]]
        The compounds behind each matched key, from ``group_matches``.
    activities_by_molregno : Mapping[int, Sequence[ChemblActivity]]
        Activities per ChEMBL compound, from ``chembl.find_activities``.
    assay_chembl_ids : Iterable[str]
        ChEMBL assay IDs named in the configuration.
    target_chembl_ids : Iterable[str]
        ChEMBL target IDs named in the configuration.
    resolved_target_chembl_ids : Iterable[str]
        ChEMBL target IDs resolved from the configured target names.
    standard_types : Iterable[str]
        Allowed ``standard_type`` values; empty matches any.

    Returns
    -------
    list[PublicLeakageFinding]
        The findings, in representation then value order.
    """
    assays = frozenset(assay_chembl_ids)
    targets = frozenset(target_chembl_ids)
    resolved = frozenset(resolved_target_chembl_ids)
    types = frozenset(t.lower() for t in standard_types)

    findings: list[PublicLeakageFinding] = []
    # The strongest finding already reported for each test row, so a coarse
    # representation can tell whether it would be repeating one.
    covered: dict[RowKey, int] = {}

    for representation in PUBLIC_SEARCH_REPRESENTATIONS:
        values = index.get(representation) or {}
        for value in sorted(values):
            compounds = grouped.get((representation, value))
            if not compounds:
                continue
            matched: dict[PublicMatchType, list[ChemblActivity]] = defaultdict(list)
            for compound in compounds:
                for activity in activities_by_molregno.get(compound["molregno"], ()):
                    # Named apart from the chosen match_type below, which is
                    # never None.
                    activity_match = classify_activity(
                        activity,
                        assay_chembl_ids=assays,
                        target_chembl_ids=targets,
                        resolved_target_chembl_ids=resolved,
                        standard_types=types,
                    )
                    if activity_match is not None:
                        matched[activity_match].append(activity)
            if not matched:
                continue
            match_type = max(matched, key=lambda m: _MATCH_STRENGTH[m])
            strength = _MATCH_STRENGTH[match_type]
            records = values[value]
            rows = {_row_key(record.location()) for record in records}
            if all(covered.get(row, 0) >= strength for row in rows):
                continue
            findings.append(
                {
                    "value": value,
                    "representation": representation,
                    "match_type": match_type,
                    "test_occurrences": _locations(records),
                    "compounds": list(compounds),
                    "activities": matched[match_type],
                }
            )
            for row in rows:
                covered[row] = max(covered.get(row, 0), strength)
    return findings


def summarise_documents(
    findings: Sequence[PublicLeakageFinding], total_test_rows: int
) -> list[PublicDocumentSummary]:
    """Summarise how much of the test set each source document covers.

    One publication or deposition covering a large share of the test set is
    the strongest unblinding signal available: it means a single paper holds
    the challenge's answers.

    Parameters
    ----------
    findings : Sequence[PublicLeakageFinding]
        The findings to group.
    total_test_rows : int
        Total rows in the test set, used for the fraction.

    Returns
    -------
    list[PublicDocumentSummary]
        One entry per document, largest share first, then by document ID.
    """
    rows_by_document: dict[str, set[RowKey]] = defaultdict(set)
    reference_by_document: dict[str, str | None] = {}
    for finding in findings:
        rows = {_row_key(location) for location in finding["test_occurrences"]}
        for activity in finding["activities"]:
            document = activity["document_chembl_id"]
            if document is None:
                continue
            rows_by_document[document] |= rows
            reference_by_document.setdefault(document, activity["document_reference"])

    summaries: list[PublicDocumentSummary] = [
        {
            "document_chembl_id": document,
            "reference": reference_by_document.get(document),
            "test_rows": len(rows),
            "fraction_of_test_set": (len(rows) / total_test_rows if total_test_rows else 0.0),
        }
        for document, rows in rows_by_document.items()
    ]
    summaries.sort(key=lambda s: (-s["test_rows"], s["document_chembl_id"]))
    return summaries


def count_rows_found(
    index: Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]],
    grouped: Mapping[tuple[Representation, str], Sequence[ChemblCompound]],
) -> int:
    """Count the distinct test rows present in the public source.

    This is the ``N`` of the "N of M test compounds found in ChEMBL"
    statistic: a context number, not a finding.

    Parameters
    ----------
    index : Mapping[Representation, Mapping[str, Sequence[MoleculeRecord]]]
        The index built by ``index_records``.
    grouped : Mapping[tuple[Representation, str], Sequence[ChemblCompound]]
        The compounds behind each matched key, from ``group_matches``.

    Returns
    -------
    int
        How many distinct test rows matched by any representation.
    """
    rows: set[RowKey] = set()
    for (representation, value), compounds in grouped.items():
        if not compounds:
            continue
        for record in (index.get(representation) or {}).get(value, ()):
            rows.add(_row_key(record.location()))
    return len(rows)
