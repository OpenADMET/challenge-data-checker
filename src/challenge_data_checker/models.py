"""Shared data model for a single processed molecule row."""

from dataclasses import dataclass, field
from typing import Any

Fingerprint = Any  # rdkit.DataStructs.cDataStructs.ExplicitBitVect


@dataclass
class MoleculeRecord:
    """The result of processing a single row of a train/test data file.

    Attributes
    ----------
    pool
        Which pool the row belongs to, ``"train"`` or ``"test"``.
    source_file
        Path of the file the row was loaded from.
    row_index
        Original row index within ``source_file``.
    raw_smiles
        The raw SMILES string as read from the input file.
    identifiers
        Mapping of identifier column name to its value for this row
        (``None`` if the column was absent or the cell was empty).
    parse_error
        Description of why parsing/processing failed, or ``None`` if the
        row was processed successfully.
    canonical_smiles
        RDKit canonical SMILES, or ``None`` if unparsed.
    inchikey
        InChIKey, or ``None`` if unparsed or InChIKey generation failed.
    fingerprint
        Stereo-blind Morgan fingerprint, or ``None`` if unparsed.
    heavy_atom_count
        Number of heavy atoms, or ``None`` if unparsed.
    num_fragments
        Number of disconnected fragments, or ``None`` if unparsed.
    has_dot
        Whether the raw SMILES contains a ``.`` fragment separator.
    contains_metal
        Whether any atom is a transition/heavy/alkali(-earth) metal.
    is_mixture
        Whether the molecule has multiple disconnected fragments.
    is_salt_or_metal
        Whether the molecule is a salt or metal complex.
    is_suspicious_fragment
        Whether the molecule looks like a stripped degradation artefact
        (e.g. a bare counterion).
    """

    pool: str
    source_file: str
    row_index: int
    raw_smiles: str
    identifiers: dict[str, str | None] = field(default_factory=dict)

    parse_error: str | None = None
    canonical_smiles: str | None = None
    inchikey: str | None = None
    fingerprint: Fingerprint | None = None

    heavy_atom_count: int | None = None
    num_fragments: int | None = None
    has_dot: bool = False
    contains_metal: bool = False
    is_mixture: bool = False
    is_salt_or_metal: bool = False
    is_suspicious_fragment: bool = False

    @property
    def is_parsed(self) -> bool:
        """Whether the row's SMILES was parsed and processed successfully.

        Returns
        -------
        ``True`` if ``parse_error`` is ``None``, ``False`` otherwise.
        """
        return self.parse_error is None

    def location(self) -> dict[str, object]:
        """Build a compact reference to where this row came from, for reports.

        Returns
        -------
        A dict with the pool, source file, row index, and raw SMILES.
        """
        return {
            "pool": self.pool,
            "source_file": self.source_file,
            "row_index": self.row_index,
            "raw_smiles": self.raw_smiles,
        }
