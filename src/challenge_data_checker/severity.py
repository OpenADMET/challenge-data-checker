"""Severity tiers for audit findings, and counting them by distinct affected rows.

Every finding has a type (``exact_overlap``, ``duplicate_structure``, ...)
and a tier: **FAIL** (blocking: the split can't be released as is) or
**WARN** (flagged: needs a human to check). The tier follows four rules:

1. Chemistry errors always FAIL, in both pools.
2. Data-curation issues depend on the pool: WARN in train (a training set may
   combine sources or hold replicates the user has to resolve), FAIL in test
   (the test set must be scoreable ground truth).
3. Exact leakage is FAIL; non-exact leakage (only the stereochemistry
   differs, or Tanimoto similarity at the default threshold) is WARN.
4. Heuristic quality flags are WARN.

Findings are counted by distinct rows, not by how many checks fired (#4):
one duplicate seen through raw SMILES, canonical SMILES and InChIKey is one
finding, not three.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from challenge_data_checker.report_types import (
    QUALITY_FLAG_CATEGORIES,
    REPRESENTATIONS,
    SALT_FORM_INSENSITIVE_REPRESENTATIONS,
    STEREO_INSENSITIVE_REPRESENTATIONS,
    InternalDuplicates,
    LeakageFinding,
    Location,
    NamespaceIssues,
    PerPool,
    QualityFlags,
    Status,
    TanimotoFinding,
    TierSummary,
    UnparseableEntry,
)

type Pool = Literal["train", "test"]
type RowKey = tuple[str, str, int]


class Severity(StrEnum):
    """A finding's tier: blocking (``FAIL``) or flagged for review (``WARN``)."""

    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class TierContext:
    """The settings a finding's tier can depend on.

    Attributes
    ----------
    max_train_test_similarity : float
        The Tanimoto threshold in use. At the default ``1.0`` the Tanimoto
        check is only a fallback for stereo misses (WARN); a lower value
        is the user asking for near-duplicates to block (FAIL).
    """

    max_train_test_similarity: float = 1.0


type TierRule = Severity | PerPool[Severity] | Callable[[TierContext], Severity]

_TRAIN_WARN_TEST_FAIL: PerPool[Severity] = {"train": Severity.WARN, "test": Severity.FAIL}

# The tier of every finding type. A plain Severity applies everywhere; a
# PerPool rule depends on which pool a within-pool finding is in; a callable
# depends on the run's settings.
TIERS: dict[str, TierRule] = {
    # Chemistry errors (rule 1)
    "unparseable_smiles": Severity.FAIL,
    # Train/test leakage (rule 3)
    "exact_overlap": Severity.FAIL,
    "salt_form_overlap": Severity.FAIL,
    "identifier_overlap": Severity.FAIL,
    "stereo_overlap": Severity.WARN,
    "tanimoto_similarity": lambda ctx: (
        Severity.WARN if ctx.max_train_test_similarity >= 1.0 else Severity.FAIL
    ),
    # Data curation within a pool (rule 2)
    "duplicate_structure": _TRAIN_WARN_TEST_FAIL,
    "duplicate_salt_form": _TRAIN_WARN_TEST_FAIL,
    "duplicate_identifier": _TRAIN_WARN_TEST_FAIL,
    "duplicate_stereoisomer": Severity.WARN,
    # Identifier namespace
    "one_id_multiple_structures": Severity.FAIL,
    "one_structure_multiple_ids": Severity.WARN,
    "identifier_whitespace": Severity.WARN,
    "identifier_near_duplicate": Severity.WARN,
    "id_with_unparseable_and_valid_structures": Severity.WARN,
    "missing_structure_resolvable_by_id": Severity.WARN,
    # Heuristic quality flags (rule 4)
    "mixture": Severity.WARN,
    "salt_or_metal_complex": Severity.WARN,
    "suspicious_fragment": Severity.WARN,
    "smiles_whitespace": Severity.WARN,
    "rdkit_warning": Severity.WARN,
}

# Finding type for each per-pool quality-flag category in the report.
QUALITY_FLAG_FINDING_TYPES: dict[str, str] = {
    "mixtures": "mixture",
    "salts_or_metal_complexes": "salt_or_metal_complex",
    "suspicious_fragments": "suspicious_fragment",
    "smiles_whitespace": "smiles_whitespace",
    "rdkit_warnings": "rdkit_warning",
}


def severity_of(finding_type: str, pool: Pool | None, context: TierContext) -> Severity:
    """Look up the tier of a finding.

    Parameters
    ----------
    finding_type : str
        A key of ``TIERS``.
    pool : Pool | None
        The pool a within-pool finding is in, or ``None`` for a finding
        spanning both pools.
    context : TierContext
        The settings the tier may depend on.

    Returns
    -------
    Severity
        The finding's tier.

    Raises
    ------
    ValueError
        If a per-pool rule is looked up without a pool.
    """
    rule = TIERS[finding_type]
    if isinstance(rule, Severity):
        return rule
    if isinstance(rule, dict):
        if pool is None:
            raise ValueError(f"Finding type '{finding_type}' needs a pool to determine its tier.")
        return rule[pool]
    return rule(context)


@dataclass(frozen=True)
class Finding:
    """One finding, reduced to what's needed to tier and count it.

    Attributes
    ----------
    finding_type : str
        A key of ``TIERS``.
    pool : Pool | None
        The pool of a within-pool finding, or ``None`` if it spans both.
    rows : frozenset[RowKey]
        The distinct (pool, source_file, row_index) rows it involves.
    """

    finding_type: str
    pool: Pool | None
    rows: frozenset[RowKey]


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


def _rows(locations: Iterable[Location]) -> frozenset[RowKey]:
    """Build the set of distinct rows behind some locations.

    Parameters
    ----------
    locations : Iterable[Location]
        Row locations from the report.

    Returns
    -------
    frozenset[RowKey]
        The distinct rows.
    """
    return frozenset(_row_key(loc) for loc in locations)


def leakage_finding_type(representation: str) -> str:
    """Map a cross-pool representation to its finding type.

    Parameters
    ----------
    representation : str
        A representation name from ``train_test_leakage["exact"]``.

    Returns
    -------
    str
        ``salt_form_overlap``, ``stereo_overlap`` or ``exact_overlap``.
    """
    if representation in SALT_FORM_INSENSITIVE_REPRESENTATIONS:
        return "salt_form_overlap"
    if representation in STEREO_INSENSITIVE_REPRESENTATIONS:
        return "stereo_overlap"
    return "exact_overlap"


def duplicate_finding_type(representation: str) -> str:
    """Map a within-pool representation to its finding type.

    Parameters
    ----------
    representation : str
        A representation name from ``internal_duplicates[pool]``.

    Returns
    -------
    str
        ``duplicate_salt_form``, ``duplicate_stereoisomer`` or
        ``duplicate_structure``.
    """
    if representation in SALT_FORM_INSENSITIVE_REPRESENTATIONS:
        return "duplicate_salt_form"
    if representation in STEREO_INSENSITIVE_REPRESENTATIONS:
        return "duplicate_stereoisomer"
    return "duplicate_structure"


def collect_findings(
    unparseable: list[UnparseableEntry],
    exact_leakage: dict[str, list[LeakageFinding]],
    identifier_leakage: dict[str, list[LeakageFinding]],
    tanimoto: list[TanimotoFinding],
    internal_duplicates: PerPool[InternalDuplicates],
    namespace_issues: dict[str, NamespaceIssues],
    quality_flags: PerPool[QualityFlags],
) -> Iterator[Finding]:
    """Reduce every check's results to typed findings with their rows.

    Parameters
    ----------
    unparseable : list[UnparseableEntry]
        The report's ``unparseable_smiles``.
    exact_leakage : dict[str, list[LeakageFinding]]
        The report's ``train_test_leakage["exact"]``.
    identifier_leakage : dict[str, list[LeakageFinding]]
        The report's ``train_test_leakage["identifiers"]``.
    tanimoto : list[TanimotoFinding]
        The report's ``train_test_leakage["tanimoto_similarity"]``.
    internal_duplicates : PerPool[InternalDuplicates]
        The report's ``internal_duplicates``.
    namespace_issues : dict[str, NamespaceIssues]
        The report's ``identifier_namespace_issues``.
    quality_flags : PerPool[QualityFlags]
        The report's ``quality_flags``.

    Yields
    ------
    Finding
        One finding per reported item, before any merging.
    """
    for entry in unparseable:
        pool: Pool = "train" if entry["pool"] == "train" else "test"
        yield Finding("unparseable_smiles", pool, _rows([entry]))

    for representation, leaks in exact_leakage.items():
        for leak in leaks:
            rows = _rows(leak["train_occurrences"]) | _rows(leak["test_occurrences"])
            yield Finding(leakage_finding_type(representation), None, rows)
    for leaks in identifier_leakage.values():
        for leak in leaks:
            rows = _rows(leak["train_occurrences"]) | _rows(leak["test_occurrences"])
            yield Finding("identifier_overlap", None, rows)
    for pair in tanimoto:
        yield Finding("tanimoto_similarity", None, _rows([pair["train"], pair["test"]]))

    for pool in ("train", "test"):
        dupes = internal_duplicates[pool]
        for representation in REPRESENTATIONS:
            for dupe in dupes[representation]:
                yield Finding(
                    duplicate_finding_type(representation), pool, _rows(dupe["occurrences"])
                )
        for id_dupes in dupes["identifiers"].values():
            for dupe in id_dupes:
                yield Finding("duplicate_identifier", pool, _rows(dupe["occurrences"]))

        flags = quality_flags[pool]
        for category in QUALITY_FLAG_CATEGORIES:
            for location in flags[category]:
                yield Finding(QUALITY_FLAG_FINDING_TYPES[category], pool, _rows([location]))

    for issues in namespace_issues.values():
        for conflict in issues["one_id_multiple_structures"]:
            rows = frozenset(
                _row_key(loc)
                for structure in conflict["structures"]
                for loc in structure["occurrences"]
            )
            yield Finding("one_id_multiple_structures", None, rows)
        for id_conflict in issues["one_structure_multiple_ids"]:
            rows = frozenset(
                _row_key(loc)
                for identifier in id_conflict["identifiers"]
                for loc in identifier["occurrences"]
            )
            yield Finding("one_structure_multiple_ids", None, rows)
        for whitespace_entry in issues["identifier_whitespace"]:
            yield Finding("identifier_whitespace", None, _rows([whitespace_entry]))
        for group in issues["near_duplicates"]:
            rows = frozenset(
                _row_key(loc)
                for identifier in group["identifiers"]
                for loc in identifier["occurrences"]
            )
            yield Finding("identifier_near_duplicate", None, rows)
        for broken in issues["id_with_unparseable_and_valid_structures"]:
            rows = _rows(broken["unparseable"]) | frozenset(
                _row_key(loc)
                for structure in broken["structures"]
                for loc in structure["occurrences"]
            )
            yield Finding("id_with_unparseable_and_valid_structures", None, rows)
        for missing in issues["missing_structure_resolvable_by_id"]:
            yield Finding("missing_structure_resolvable_by_id", None, _rows(missing["missing"]))


def _drop_redundant(row_sets: Iterable[frozenset[RowKey]]) -> list[frozenset[RowKey]]:
    """Drop row sets that equal, or are contained in, another one.

    The same problem is often reported by several representations (raw
    SMILES, canonical SMILES, InChIKey) with the same or a subset of rows;
    it should count once.

    Parameters
    ----------
    row_sets : Iterable[frozenset[RowKey]]
        The row sets of findings of one type.

    Returns
    -------
    list[frozenset[RowKey]]
        The maximal row sets, each kept once.
    """
    kept: list[frozenset[RowKey]] = []
    containing: dict[RowKey, list[frozenset[RowKey]]] = defaultdict(list)
    for rows in sorted(set(row_sets), key=len, reverse=True):
        if not rows:
            continue
        first = min(rows)
        if any(rows <= other for other in containing[first]):
            continue
        kept.append(rows)
        for row in rows:
            containing[row].append(rows)
    return kept


def summarise_tiers(
    findings: Iterable[Finding], context: TierContext
) -> tuple[Status, TierSummary, TierSummary]:
    """Tier the findings and count them by distinct affected rows.

    Parameters
    ----------
    findings : Iterable[Finding]
        Every finding, as produced by ``collect_findings``.
    context : TierContext
        The settings tiers may depend on.

    Returns
    -------
    Status
        ``"FAIL"`` if anything is blocking, ``"WARN"`` if anything is
        flagged, otherwise ``"PASS"``.
    TierSummary
        Counts for the blocking (FAIL) tier.
    TierSummary
        Counts for the flagged (WARN) tier.
    """
    grouped: dict[tuple[Severity, str], list[frozenset[RowKey]]] = defaultdict(list)
    for finding in findings:
        severity = severity_of(finding.finding_type, finding.pool, context)
        grouped[(severity, finding.finding_type)].append(finding.rows)

    tiers: dict[Severity, TierSummary] = {
        severity: {"findings": 0, "rows": 0, "by_type": {}} for severity in Severity
    }
    tier_rows: dict[Severity, set[RowKey]] = {severity: set() for severity in Severity}
    for (severity, finding_type), row_sets in sorted(grouped.items()):
        kept = _drop_redundant(row_sets)
        rows = frozenset().union(*kept)
        tiers[severity]["by_type"][finding_type] = {"findings": len(kept), "rows": len(rows)}
        tiers[severity]["findings"] += len(kept)
        tier_rows[severity] |= rows
    for severity in Severity:
        tiers[severity]["rows"] = len(tier_rows[severity])

    blocking, flagged = tiers[Severity.FAIL], tiers[Severity.WARN]
    status: Status = "FAIL" if blocking["findings"] else "WARN" if flagged["findings"] else "PASS"
    return status, blocking, flagged
