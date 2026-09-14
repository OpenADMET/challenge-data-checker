"""Core RDKit molecule processing: parsing, standardisation, canonicalisation,
InChIKey and fingerprint generation.
"""

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi as rdinchi
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from challenge_data_checker.config import SettingsConfig
from challenge_data_checker.io_utils import RESOLVED_SMILES_COL, ROW_INDEX_COL, SOURCE_FILE_COL
from challenge_data_checker.models import Fingerprint, MoleculeRecord
from challenge_data_checker.quality_filters import (
    contains_metal,
    has_multiple_fragments,
    is_salt_or_metal_complex,
    is_suspicious_small_fragment,
)

# Silence RDKit's C++ warning/error logging; parse failures are surfaced explicitly
# per-row instead via MoleculeRecord.parse_error.
RDLogger.DisableLog("rdApp.*")


def parse_smiles(smiles: str) -> tuple[Chem.Mol | None, str | None]:
    """Parse a SMILES string.

    Parameters
    ----------
    smiles
        The SMILES string to parse.

    Returns
    -------
    A tuple of the parsed molecule and an error message. Exactly one of
    the two is ``None``: the molecule if parsing succeeded, or the error
    message if it failed.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "Empty or non-string SMILES value"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "RDKit failed to parse SMILES"
    return mol, None


class MoleculeProcessor:
    """Reusable RDKit helper for standardisation, canonicalisation, and fingerprinting.

    Parameters
    ----------
    settings
        Chemistry settings controlling tautomer standardisation and
        Morgan fingerprint generation.
    """

    def __init__(self, settings: SettingsConfig) -> None:
        self.settings = settings
        self._tautomer_enumerator = (
            rdMolStandardize.TautomerEnumerator() if settings.tautomer_standardisation else None
        )
        self._fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=settings.fp_radius,
            fpSize=settings.fp_n_bits,
            includeChirality=False,
        )

    def standardise(self, mol: Chem.Mol) -> Chem.Mol:
        """Canonicalise the molecule's tautomeric form, if enabled.

        Parameters
        ----------
        mol
            The molecule to standardise.

        Returns
        -------
        The tautomer-canonicalised molecule if
        ``settings.tautomer_standardisation`` is enabled, otherwise ``mol``
        unchanged.
        """
        if self._tautomer_enumerator is not None:
            return self._tautomer_enumerator.Canonicalize(mol)
        return mol

    def canonical_smiles(self, mol: Chem.Mol) -> str:
        """Generate the RDKit canonical SMILES for a molecule.

        Parameters
        ----------
        mol
            The molecule to canonicalise.

        Returns
        -------
        The canonical SMILES string.
        """
        return Chem.MolToSmiles(mol, canonical=True)

    def inchikey(self, mol: Chem.Mol) -> str | None:
        """Generate the InChIKey for a molecule.

        Parameters
        ----------
        mol
            The molecule to generate an InChIKey for.

        Returns
        -------
        The InChIKey, or ``None`` if InChIKey generation failed.
        """
        key = rdinchi.MolToInchiKey(mol)
        return key or None

    def fingerprint(self, mol: Chem.Mol) -> Fingerprint:
        """Generate a stereo-blind Morgan fingerprint for a molecule.

        Parameters
        ----------
        mol
            The molecule to fingerprint.

        Returns
        -------
        The Morgan fingerprint bit vector, using the radius and bit
        length from ``settings`` and ignoring stereochemistry.
        """
        return self._fp_generator.GetFingerprint(mol)

    def process_row(
        self,
        pool: str,
        source_file: str,
        row_index: int,
        raw_smiles: str,
        identifiers: dict[str, str | None],
    ) -> MoleculeRecord:
        """Process a single data row into a MoleculeRecord.

        Parses the SMILES, then (on success) standardises, canonicalises,
        and fingerprints the molecule and evaluates the quality-filter
        heuristics. Any failure at any stage is captured on the returned
        record's ``parse_error`` rather than raised.

        Parameters
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
            Mapping of identifier column name to its value for this row.

        Returns
        -------
        The fully populated MoleculeRecord for this row.
        """
        record = MoleculeRecord(
            pool=pool,
            source_file=source_file,
            row_index=row_index,
            raw_smiles=raw_smiles,
            identifiers=identifiers,
        )

        mol, error = parse_smiles(raw_smiles)
        if mol is None:
            record.parse_error = error
            return record

        try:
            mol = self.standardise(mol)
            record.canonical_smiles = self.canonical_smiles(mol)
            record.inchikey = self.inchikey(mol)
            record.fingerprint = self.fingerprint(mol)
            record.heavy_atom_count = mol.GetNumHeavyAtoms()
            record.num_fragments = len(Chem.GetMolFrags(mol))
            record.has_dot = "." in raw_smiles
            record.contains_metal = contains_metal(mol)
            record.is_mixture = has_multiple_fragments(mol, raw_smiles)
            record.is_salt_or_metal = is_salt_or_metal_complex(mol)
            record.is_suspicious_fragment = is_suspicious_small_fragment(
                mol, record.heavy_atom_count
            )
        except Exception as exc:  # noqa: BLE001 - surface any RDKit failure per-row
            record.parse_error = f"Processing failed: {exc}"

        return record


def process_pool(
    df: pd.DataFrame,
    pool: str,
    identifier_columns: list[str],
    processor: MoleculeProcessor,
) -> list[MoleculeRecord]:
    """Process every row of a pooled DataFrame into MoleculeRecords.

    Parameters
    ----------
    df
        A pooled DataFrame, as returned by ``io_utils.load_pool``.
    pool
        Which pool ``df`` represents, ``"train"`` or ``"test"``.
    identifier_columns
        Identifier column names to extract per row.
    processor
        The MoleculeProcessor to use for parsing/standardisation/
        fingerprinting.

    Returns
    -------
    One MoleculeRecord per row of ``df``.
    """
    records = []
    for _, row in df.iterrows():
        identifiers = {
            col: (None if pd.isna(row[col]) else str(row[col])) for col in identifier_columns
        }
        raw_smiles = row[RESOLVED_SMILES_COL]
        raw_smiles = "" if pd.isna(raw_smiles) else str(raw_smiles)
        record = processor.process_row(
            pool=pool,
            source_file=row[SOURCE_FILE_COL],
            row_index=int(row[ROW_INDEX_COL]),
            raw_smiles=raw_smiles,
            identifiers=identifiers,
        )
        records.append(record)
    return records
