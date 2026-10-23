"""Typed schema of the audit report dict returned by ``audit()`` and the CLI.

The report is a plain JSON-serialisable dict; these ``TypedDict`` classes
only describe its shape so type checkers can verify the code that builds and
renders it. They add no runtime behaviour.
"""

from typing import Literal, TypedDict

# Chemical representations compared by the leakage and duplicate checks (see
# ``checks._representation_key_funcs``). The exact ones match only the same
# structure, stereochemistry and salt form included; the coarse ones
# deliberately ignore part of the structure.
type Representation = Literal[
    "raw_smiles", "canonical_smiles", "inchikey", "parent_inchikey", "inchikey_connectivity"
]
REPRESENTATIONS: tuple[Representation, ...] = (
    "raw_smiles",
    "canonical_smiles",
    "inchikey",
    "parent_inchikey",
    "inchikey_connectivity",
)
# Representations that ignore the salt form: the InChIKey of the salt- and
# solvent-stripped, neutralised parent (only compared when the
# ``strip_salts`` setting is on). A match groups a salt with its free base
# or another salt form of the same compound, stereochemistry included.
SALT_FORM_INSENSITIVE_REPRESENTATIONS: frozenset[Representation] = frozenset({"parent_inchikey"})
# Representations that ignore stereochemistry: the first (connectivity)
# block of the InChIKey. A match groups stereoisomers and racemate /
# enantiomer pairs.
STEREO_INSENSITIVE_REPRESENTATIONS: frozenset[Representation] = frozenset({"inchikey_connectivity"})
# A group under a coarse representation is only reported when its members
# don't all share one full InChIKey; otherwise it would just repeat an
# exact match.
COARSE_REPRESENTATIONS: frozenset[Representation] = (
    SALT_FORM_INSENSITIVE_REPRESENTATIONS | STEREO_INSENSITIVE_REPRESENTATIONS
)
EXACT_REPRESENTATIONS: tuple[Representation, ...] = tuple(
    r for r in REPRESENTATIONS if r not in COARSE_REPRESENTATIONS
)

# Per-pool quality-filter categories (see ``report.quality_flag_entries``).
type QualityFlagCategory = Literal[
    "salt_or_solvate",
    "contains_metal",
    "multi_component",
    "homodimer_without_counterion",
    "unclassified_mixture",
    "net_charge",
    "undefined_stereo",
    "suspicious_fragments",
    "smiles_whitespace",
    "rdkit_warnings",
    "mixtures",
    "salts_or_metal_complexes",
]
QUALITY_FLAG_CATEGORIES: tuple[QualityFlagCategory, ...] = (
    "salt_or_solvate",
    "contains_metal",
    "multi_component",
    "homodimer_without_counterion",
    "unclassified_mixture",
    "net_charge",
    "undefined_stereo",
    "suspicious_fragments",
    "smiles_whitespace",
    "rdkit_warnings",
    "mixtures",
    "salts_or_metal_complexes",
)
# Kept for backwards compatibility, as totals: every multi-fragment entry,
# and every salt/solvate or metal-containing entry. Each such entry is also
# under one of the specific categories above, which are the ones tiered.
DEPRECATED_QUALITY_FLAG_CATEGORIES: frozenset[QualityFlagCategory] = frozenset(
    {"mixtures", "salts_or_metal_complexes"}
)


class PerPool[T](TypedDict):
    """A value held separately for the train and test pools."""

    train: T
    test: T


class Location(TypedDict):
    """Where a finding's row came from, as built by ``MoleculeRecord.location``."""

    pool: str
    source_file: str
    row_index: int
    raw_smiles: str


class UnparseableOccurrence(Location):
    """A row whose SMILES could not be parsed, with why."""

    error: str
    category: str


class UnparseableEntry(UnparseableOccurrence):
    """A row whose SMILES could not be parsed or processed.

    ``resolvable_via_identifier`` names an identifier column through which an
    empty SMILES can be filled in: the row's identifier has exactly one
    parsed structure elsewhere. ``None`` otherwise.
    """

    resolvable_via_identifier: str | None


class RdkitWarningEntry(Location):
    """A row whose SMILES parsed, but with a warning from RDKit."""

    warning: str


class NetChargeEntry(Location):
    """A row whose formal charges don't sum to zero."""

    charge: int


class UndefinedStereoEntry(Location):
    """A row whose SMILES leaves some stereochemistry unspecified."""

    stereocentres: int
    stereo_bonds: int


class LeakageFinding(TypedDict):
    """A value shared between the train and test pools."""

    value: str
    train_occurrences: list[Location]
    test_occurrences: list[Location]


class TanimotoFinding(TypedDict):
    """A train/test pair at or above the Tanimoto similarity threshold."""

    train: Location
    test: Location
    similarity: float


class TrainTestLeakage(TypedDict):
    """All train/test leakage findings."""

    exact: dict[str, list[LeakageFinding]]
    identifiers: dict[str, list[LeakageFinding]]
    tanimoto_similarity: list[TanimotoFinding]


class DuplicateFinding(TypedDict):
    """A value that appears more than once within a single pool."""

    value: str
    occurrences: list[Location]


class InternalDuplicates(TypedDict):
    """Within-pool duplicates, per representation and per identifier column."""

    raw_smiles: list[DuplicateFinding]
    canonical_smiles: list[DuplicateFinding]
    inchikey: list[DuplicateFinding]
    parent_inchikey: list[DuplicateFinding]
    inchikey_connectivity: list[DuplicateFinding]
    identifiers: dict[str, list[DuplicateFinding]]


class StructureOccurrences(TypedDict):
    """One distinct structure, and the rows where it appears."""

    canonical_smiles: str | None
    inchikey: str | None
    occurrences: list[Location]


# How the structures behind one identifier differ, from the finest level
# that unifies them all: tautomer/representation, salt form, stereochemistry,
# or not at all (genuinely different compounds).
type IdConflictType = Literal["tautomer", "salt_form", "stereo", "different_compound"]

# How the structures given several identifiers relate: identical, the same
# parent in different salt forms, or the same connectivity with different
# stereochemistry.
type SharedStructureType = Literal["same_structure", "salt_form", "stereo"]


class OneIdMultipleStructures(TypedDict):
    """A single identifier value mapped to two or more distinct structures.

    ``max_similarity`` is the highest Tanimoto similarity between the
    structures, which helps tell a near-analogue mix-up from an unrelated
    one; ``None`` if fingerprints weren't available.
    """

    identifier: str
    conflict_type: IdConflictType
    max_similarity: float | None
    structures: list[StructureOccurrences]


class IdentifierOccurrences(TypedDict):
    """One identifier value, and the rows where it appears."""

    identifier: str
    occurrences: list[Location]


class OneStructureMultipleIds(TypedDict):
    """A single compound given two or more distinct identifier values.

    Rows are grouped by the connectivity of their parent compound, so the
    same compound in different salt forms or with different stereo
    annotation is caught too; ``conflict_type`` says how the structures
    relate. ``canonical_smiles`` and ``inchikey`` are those of the first
    row, as a representative.
    """

    conflict_type: SharedStructureType
    canonical_smiles: str | None
    inchikey: str | None
    identifiers: list[IdentifierOccurrences]


class IdentifierWhitespaceEntry(Location):
    """A row whose identifier value had surrounding whitespace stripped."""

    raw_value: str
    value: str


class NearDuplicateIdentifiers(TypedDict):
    """Distinct identifier values that only differ in case or ``-``/``_``."""

    key: str
    identifiers: list[IdentifierOccurrences]


class IdWithUnparseableAndValidStructures(TypedDict):
    """An identifier with parsed structures in some rows and broken SMILES in others."""

    identifier: str
    structures: list[StructureOccurrences]
    unparseable: list[UnparseableOccurrence]


class MissingStructureResolvableById(TypedDict):
    """An identifier with empty SMILES in some rows and parsed structures in others.

    ``ambiguous`` is ``True`` when the identifier has more than one parsed
    structure, so the empty rows can't be filled in unambiguously.
    """

    identifier: str
    missing: list[Location]
    candidates: list[StructureOccurrences]
    ambiguous: bool


class NamespaceIssues(TypedDict):
    """Identifier consistency findings for one identifier column."""

    one_id_multiple_structures: list[OneIdMultipleStructures]
    one_structure_multiple_ids: list[OneStructureMultipleIds]
    identifier_whitespace: list[IdentifierWhitespaceEntry]
    near_duplicates: list[NearDuplicateIdentifiers]
    id_with_unparseable_and_valid_structures: list[IdWithUnparseableAndValidStructures]
    missing_structure_resolvable_by_id: list[MissingStructureResolvableById]


class QualityFlags(TypedDict):
    """Quality-filter findings for one pool."""

    mixtures: list[Location]
    salts_or_metal_complexes: list[Location]
    suspicious_fragments: list[Location]
    salt_or_solvate: list[Location]
    contains_metal: list[Location]
    multi_component: list[Location]
    homodimer_without_counterion: list[Location]
    unclassified_mixture: list[Location]
    net_charge: list[NetChargeEntry]
    undefined_stereo: list[UndefinedStereoEntry]
    smiles_whitespace: list[Location]
    rdkit_warnings: list[RdkitWarningEntry]


class NamespaceIssueCounts(TypedDict):
    """Finding counts for one identifier column's namespace checks."""

    one_id_multiple_structures: int
    one_id_multiple_structures_by_type: dict[str, int]
    one_structure_multiple_ids: int
    identifier_whitespace: int
    near_duplicates: int
    id_with_unparseable_and_valid_structures: int
    missing_structure_resolvable_by_id: int


# Overall audit result: FAIL if any blocking finding, WARN if only flagged
# findings, PASS if nothing was found.
type Status = Literal["PASS", "WARN", "FAIL"]


class TypeCounts(TypedDict):
    """Counts for one finding type within a severity tier."""

    findings: int
    rows: int


class TierSummary(TypedDict):
    """Counts for one severity tier (blocking or flagged).

    ``findings`` counts distinct problems (one duplicate seen through several
    representations counts once); ``rows`` counts distinct affected rows.
    """

    findings: int
    rows: int
    by_type: dict[str, TypeCounts]


class Summary(TypedDict):
    """Headline counts for every check."""

    status: Status
    blocking: TierSummary
    flagged: TierSummary
    total_train_rows: int
    total_test_rows: int
    total_unparseable: int
    total_unparseable_train: int
    total_unparseable_test: int
    total_unparseable_by_category: dict[str, int]
    total_unparseable_resolvable_via_identifier: int
    total_exact_leaked_train_molecules: int
    total_exact_leaked_test_molecules: int
    total_salt_form_leaked_train_molecules: int
    total_salt_form_leaked_test_molecules: int
    total_stereo_leaked_train_molecules: int
    total_stereo_leaked_test_molecules: int
    total_identifier_leakage_events: int
    total_tanimoto_leakage_pairs: int
    total_internal_duplicates_train: dict[str, int]
    total_internal_duplicates_test: dict[str, int]
    total_internal_duplicates_train_identifiers: dict[str, int]
    total_internal_duplicates_test_identifiers: dict[str, int]
    total_identifier_namespace_issues: dict[str, NamespaceIssueCounts]
    total_mixtures: PerPool[int]
    total_salts_or_metal_complexes: PerPool[int]
    total_suspicious_fragments: PerPool[int]
    total_quality_flags: dict[str, PerPool[int]]


class ReportConfig(TypedDict):
    """The configuration echoed back into the report."""

    train_files: list[str]
    test_files: list[str]
    smiles_column: str
    resolved_smiles_columns: PerPool[dict[str, str]]
    identifier_columns: list[str]
    tautomer_standardisation: bool
    strip_salts: bool
    identifier_case_insensitive: bool
    multi_component_min_heavy_atoms: int
    max_train_test_similarity: float
    fp_radius: int
    fp_n_bits: int
    report_format: str


class Report(TypedDict):
    """The full audit report."""

    generated_at: str
    mode: Literal["audit"]
    config: ReportConfig
    summary: Summary
    unparseable_smiles: list[UnparseableEntry]
    train_test_leakage: TrainTestLeakage
    internal_duplicates: PerPool[InternalDuplicates]
    identifier_namespace_issues: dict[str, NamespaceIssues]
    quality_flags: PerPool[QualityFlags]


class CheckSummary(TypedDict):
    """Headline counts for a single-dataset check."""

    status: Status
    blocking: TierSummary
    flagged: TierSummary
    total_rows: int
    total_unparseable: int
    total_unparseable_by_category: dict[str, int]
    total_unparseable_resolvable_via_identifier: int
    total_internal_duplicates: dict[str, int]
    total_internal_duplicates_identifiers: dict[str, int]
    total_identifier_namespace_issues: dict[str, NamespaceIssueCounts]
    total_quality_flags: dict[str, int]


class CheckReportConfig(TypedDict):
    """The configuration echoed back into a check report."""

    files: list[str]
    smiles_column: str
    resolved_smiles_columns: dict[str, str]
    identifier_columns: list[str]
    tautomer_standardisation: bool
    strip_salts: bool
    identifier_case_insensitive: bool
    multi_component_min_heavy_atoms: int
    report_format: str


class CheckReport(TypedDict):
    """The report of a single-dataset check: the audit's single-pool sections.

    Rows are in the ``"dataset"`` pool. Duplicates within the dataset are
    flagged (WARN), as for a training set, since the checker can't know
    whether the dataset will be a test set.
    """

    generated_at: str
    mode: Literal["check"]
    config: CheckReportConfig
    summary: CheckSummary
    unparseable_smiles: list[UnparseableEntry]
    internal_duplicates: InternalDuplicates
    identifier_namespace_issues: dict[str, NamespaceIssues]
    quality_flags: QualityFlags
