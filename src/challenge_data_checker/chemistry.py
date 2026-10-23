"""Core RDKit molecule processing: parsing, standardisation, canonicalisation,
InChIKey and fingerprint generation.
"""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import inchi as rdinchi
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from challenge_data_checker.config import SettingsConfig
from challenge_data_checker.io_utils import RESOLVED_SMILES_COL, ROW_INDEX_COL, SOURCE_FILE_COL
from challenge_data_checker.models import Fingerprint, MoleculeRecord
from challenge_data_checker.parse_errors import (
    ParseErrorCategory,
    classify_parse_failure,
    has_embedded_control_characters,
)
from challenge_data_checker.quality_filters import (
    classify_fragments,
    contains_metal,
    count_undefined_stereo,
    has_multiple_fragments,
    is_suspicious_small_fragment,
)

logger = logging.getLogger(__name__)

# Route RDKit's C++ logging through Python's "rdkit" logger and keep it silent
# by default. Parse failures and warnings are captured per row instead (see
# capture_rdkit_log) and surfaced via MoleculeRecord.
rdBase.LogToPythonLogger()
RDLogger.DisableLog("rdApp.*")

_RDKIT_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s?")
_SMILES_PARSE_ERROR_PREFIX = "SMILES Parse Error: "


class _RecordingHandler(logging.Handler):
    """A logging handler that keeps every record's level and message."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store a log record's level and timestamp-free message.

        Parameters
        ----------
        record : logging.LogRecord
            The record RDKit logged.

        Returns
        -------
        None
        """
        message = _RDKIT_TIMESTAMP_RE.sub("", record.getMessage())
        self.records.extend((record.levelno, line) for line in message.splitlines())


@contextmanager
def capture_rdkit_log() -> Iterator[list[tuple[int, str]]]:
    """Capture RDKit's error and warning messages for the duration of a block.

    RDKit's logging is process-global, so this temporarily replaces the
    handlers on the ``rdkit`` Python logger and re-enables RDKit's error and
    warning channels, restoring both on exit.

    Yields
    ------
    list[tuple[int, str]]
        The ``(logging level, message line)`` pairs logged inside the
        block, filled in as RDKit logs them.
    """
    rdkit_logger = logging.getLogger("rdkit")
    saved = (rdkit_logger.handlers[:], rdkit_logger.level, rdkit_logger.propagate)
    handler = _RecordingHandler()
    rdkit_logger.handlers = [handler]
    rdkit_logger.setLevel(logging.DEBUG)
    rdkit_logger.propagate = False
    RDLogger.EnableLog("rdApp.error")
    RDLogger.EnableLog("rdApp.warning")
    try:
        yield handler.records
    finally:
        RDLogger.DisableLog("rdApp.*")
        rdkit_logger.handlers, level, rdkit_logger.propagate = saved
        rdkit_logger.setLevel(level)


def _error_message(lines: list[tuple[int, str]]) -> str:
    """Condense RDKit's parse-error log lines into one readable message.

    Drops the echoed input and the ``~~~^`` pointer lines, and the final
    "Failed parsing SMILES" summary, keeping the actual reason.

    Parameters
    ----------
    lines : list[tuple[int, str]]
        Captured ``(level, line)`` pairs.

    Returns
    -------
    str
        The error lines joined with ``"; "``, or an empty string if RDKit
        logged no error.
    """
    kept: list[str] = []
    for level, line in lines:
        if level < logging.ERROR:
            continue
        text = line.removeprefix(_SMILES_PARSE_ERROR_PREFIX).strip()
        if set(text) <= {"~", "^"}:
            if kept:
                kept.pop()  # the echoed input the pointer refers to
            continue
        if not text or text.startswith("Failed parsing SMILES"):
            continue
        if text.startswith("syntax error while parsing"):
            text = "syntax error"
        kept.append(text.rstrip(":"))
    return "; ".join(kept)


def _warning_message(lines: list[tuple[int, str]]) -> str | None:
    """Join RDKit's warning lines into one message.

    Parameters
    ----------
    lines : list[tuple[int, str]]
        Captured ``(level, line)`` pairs.

    Returns
    -------
    str | None
        The warning lines joined with a space, or ``None`` if RDKit
        logged no warning.
    """
    warnings = [line.strip() for level, line in lines if level == logging.WARNING]
    return " ".join(warnings) or None


@dataclass
class ParseResult:
    """The outcome of parsing one SMILES string.

    Attributes
    ----------
    mol : Chem.Mol | None
        The parsed molecule, or ``None`` if parsing failed.
    error : str | None
        RDKit's explanation of the failure, or ``None`` on success.
    error_category : ParseErrorCategory | None
        The failure's likely root cause (see
        ``parse_errors.classify_parse_failure``), or ``None`` on success.
    warning : str | None
        Any warning RDKit logged while parsing a molecule that did parse
        (e.g. conflicting double-bond directions, so that bond's stereo was
        dropped), or ``None``.
    whitespace_recovered : bool
        Whether the raw value failed to parse but parsed once leading /
        trailing whitespace was stripped.
    """

    mol: Chem.Mol | None
    error: str | None = None
    error_category: ParseErrorCategory | None = None
    warning: str | None = None
    whitespace_recovered: bool = field(default=False)


def _parse(smiles: str) -> tuple[Chem.Mol | None, list[tuple[int, str]]]:
    """Parse a SMILES string, capturing what RDKit logs.

    Parameters
    ----------
    smiles : str
        The SMILES string to parse.

    Returns
    -------
    Chem.Mol | None
        The parsed molecule, or ``None`` if parsing failed.
    list[tuple[int, str]]
        The ``(level, line)`` pairs RDKit logged.
    """
    with capture_rdkit_log() as lines:
        mol = Chem.MolFromSmiles(smiles)
    return mol, lines


def parse_smiles(smiles: str) -> ParseResult:
    """Parse a SMILES string, explaining any failure.

    A value that only parses once leading/trailing whitespace is stripped
    (RDKit tolerates that for plain SMILES, but not for CXSMILES) is kept,
    with ``whitespace_recovered`` set so it can be flagged. A value with
    control characters inside it is rejected even though RDKit would parse
    the part before them.

    Parameters
    ----------
    smiles : str
        The SMILES string to parse.

    Returns
    -------
    ParseResult
        The molecule on success; otherwise RDKit's error message and the
        failure's category.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ParseResult(None, "Empty or non-string SMILES value", "empty")
    if has_embedded_control_characters(smiles):
        return ParseResult(
            None,
            "Contains control characters (e.g. an embedded newline); RDKit would "
            "silently read only the part before them",
            "control_characters",
        )
    mol, lines = _parse(smiles)
    if mol is not None:
        return ParseResult(mol, warning=_warning_message(lines))
    stripped = smiles.strip()
    if stripped != smiles:
        retry, retry_lines = _parse(stripped)
        if retry is not None:
            return ParseResult(
                retry, warning=_warning_message(retry_lines), whitespace_recovered=True
            )
    error = _error_message(lines) or "RDKit failed to parse SMILES"
    return ParseResult(None, error, classify_parse_failure(smiles, error))


class MoleculeProcessor:
    """Reusable RDKit helper for standardisation, canonicalisation, and fingerprinting.

    Parameters
    ----------
    settings : SettingsConfig
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
        mol : Chem.Mol
            The molecule to standardise.

        Returns
        -------
        Chem.Mol
            The tautomer-canonicalised molecule if
            ``settings.tautomer_standardisation`` is enabled, otherwise
            ``mol`` unchanged.
        """
        if self._tautomer_enumerator is not None:
            return self._tautomer_enumerator.Canonicalize(mol)
        return mol

    def canonical_smiles(self, mol: Chem.Mol) -> str:
        """Generate the RDKit canonical SMILES for a molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to canonicalise.

        Returns
        -------
        str
            The canonical SMILES string.
        """
        return Chem.MolToSmiles(mol, canonical=True)

    def inchikey(self, mol: Chem.Mol) -> str | None:
        """Generate the InChIKey for a molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to generate an InChIKey for.

        Returns
        -------
        str | None
            The InChIKey, or ``None`` if InChIKey generation failed.
        """
        key = rdinchi.MolToInchiKey(mol)
        return key or None

    def fingerprint(self, mol: Chem.Mol) -> Fingerprint:
        """Generate a stereo-blind Morgan fingerprint for a molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to fingerprint.

        Returns
        -------
        Fingerprint
            The Morgan fingerprint bit vector, using the radius and bit
            length from ``settings`` and ignoring stereochemistry.
        """
        return self._fp_generator.GetFingerprint(mol)

    def normalise_identifier(self, value: str | None) -> str | None:
        """Normalise an identifier value for comparison.

        Surrounding whitespace is never meaningful in an identifier, so it is
        always stripped; letter case is folded only when the
        ``identifier_case_insensitive`` setting is on.

        Parameters
        ----------
        value : str | None
            The raw identifier value.

        Returns
        -------
        str | None
            The normalised value, or ``None`` if ``value`` is ``None`` or
            only whitespace.
        """
        if value is None:
            return None
        normalised = value.strip()
        if self.settings.identifier_case_insensitive:
            normalised = normalised.casefold()
        return normalised or None

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
        pool : str
            Which pool the row belongs to, ``"train"`` or ``"test"``.
        source_file : str
            Path of the file the row was loaded from.
        row_index : int
            Original row index within ``source_file``.
        raw_smiles : str
            The raw SMILES string as read from the input file.
        identifiers : dict[str, str | None]
            Mapping of identifier column name to its raw value for this row.

        Returns
        -------
        MoleculeRecord
            The fully populated MoleculeRecord for this row.
        """
        record = MoleculeRecord(
            pool=pool,
            source_file=source_file,
            row_index=row_index,
            raw_smiles=raw_smiles,
            identifiers={col: self.normalise_identifier(v) for col, v in identifiers.items()},
            raw_identifiers=dict(identifiers),
        )

        parsed = parse_smiles(raw_smiles)
        if parsed.mol is None:
            record.parse_error = parsed.error
            record.parse_error_category = parsed.error_category
            return record
        mol = parsed.mol
        record.net_charge = Chem.GetFormalCharge(mol)
        # Before standardisation, which could add or remove stereo elements.
        record.undefined_stereocentres, record.undefined_stereo_bonds = count_undefined_stereo(mol)
        record.rdkit_warning = parsed.warning
        record.smiles_whitespace_recovered = parsed.whitespace_recovered

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

            fragments = classify_fragments(mol)
            record.stripped_fragments = list(fragments.stripped)
            record.is_salt_or_solvate = fragments.is_salt_or_solvate
            record.is_salt_or_metal = record.contains_metal or fragments.is_salt_or_solvate
            min_heavy_atoms = self.settings.multi_component_min_heavy_atoms
            record.is_multi_component = fragments.is_multi_component(min_heavy_atoms)
            record.is_homodimer_without_counterion = fragments.is_homodimer_without_counterion
            record.is_unclassified_mixture = fragments.is_unclassified_mixture(min_heavy_atoms)
            parent = fragments.parent
            if Chem.MolToSmiles(parent) == record.canonical_smiles:
                # Already a single neutral fragment: reuse the work done above.
                record.parent_smiles = record.canonical_smiles
                record.parent_inchikey = record.inchikey
            else:
                # Neutralising can change the tautomer, so re-standardise.
                parent = self.standardise(parent)
                record.parent_smiles = self.canonical_smiles(parent)
                record.parent_inchikey = self.inchikey(parent)
            record.is_suspicious_fragment = is_suspicious_small_fragment(
                mol, record.heavy_atom_count
            )
        except Exception as exc:  # noqa: BLE001 - surface any RDKit failure per-row
            logger.debug("Processing failed for %s row %d", source_file, row_index, exc_info=True)
            record.parse_error = f"Processing failed: {type(exc).__name__}: {exc}"
            record.parse_error_category = "processing"

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
    df : pd.DataFrame
        A pooled DataFrame, as returned by ``io_utils.load_pool``.
    pool : str
        Which pool ``df`` represents, ``"train"`` or ``"test"``.
    identifier_columns : list[str]
        Identifier column names to extract per row.
    processor : MoleculeProcessor
        The MoleculeProcessor to use for parsing/standardisation/
        fingerprinting.

    Returns
    -------
    list[MoleculeRecord]
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
