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
type QualityFlagCategory = Literal["mixtures", "salts_or_metal_complexes", "suspicious_fragments"]
QUALITY_FLAG_CATEGORIES: tuple[QualityFlagCategory, ...] = (
    "mixtures",
    "salts_or_metal_complexes",
    "suspicious_fragments",
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


class UnparseableEntry(Location):
    """A row whose SMILES could not be parsed or processed."""

    error: str


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


class OneIdMultipleStructures(TypedDict):
    """A single identifier value mapped to two or more distinct structures."""

    identifier: str
    structures: list[StructureOccurrences]


class IdentifierOccurrences(TypedDict):
    """One identifier value, and the rows where it appears."""

    identifier: str
    occurrences: list[Location]


class OneStructureMultipleIds(TypedDict):
    """A single structure given two or more distinct identifier values."""

    canonical_smiles: str | None
    inchikey: str | None
    identifiers: list[IdentifierOccurrences]


class NamespaceIssues(TypedDict):
    """Identifier-to-structure consistency findings for one identifier column."""

    one_id_multiple_structures: list[OneIdMultipleStructures]
    one_structure_multiple_ids: list[OneStructureMultipleIds]


class QualityFlags(TypedDict):
    """Quality-filter findings for one pool."""

    mixtures: list[Location]
    salts_or_metal_complexes: list[Location]
    suspicious_fragments: list[Location]


class NamespaceIssueCounts(TypedDict):
    """Finding counts for one identifier column's namespace checks."""

    one_id_multiple_structures: int
    one_structure_multiple_ids: int


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


class ReportConfig(TypedDict):
    """The configuration echoed back into the report."""

    train_files: list[str]
    test_files: list[str]
    smiles_column: str
    resolved_smiles_columns: PerPool[dict[str, str]]
    identifier_columns: list[str]
    tautomer_standardisation: bool
    strip_salts: bool
    max_train_test_similarity: float
    fp_radius: int
    fp_n_bits: int
    report_format: str


class Report(TypedDict):
    """The full audit report."""

    generated_at: str
    config: ReportConfig
    summary: Summary
    unparseable_smiles: list[UnparseableEntry]
    train_test_leakage: TrainTestLeakage
    internal_duplicates: PerPool[InternalDuplicates]
    identifier_namespace_issues: dict[str, NamespaceIssues]
    quality_flags: PerPool[QualityFlags]
