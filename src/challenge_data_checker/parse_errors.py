"""Heuristic classification of SMILES parse failures by likely root cause.

A raw dataset's parse failures usually have a handful of distinct causes,
each with a different fix (re-export untruncated, resolve an R-group, strip
stoichiometry text, ...). Each failure gets one category, decided from the
raw string first and then from RDKit's own error message.
"""

import re
from typing import Literal

type ParseErrorCategory = Literal[
    "empty",
    "control_characters",
    "truncated",
    "placeholder",
    "markush",
    "stoichiometry_text",
    "valence",
    "kekulisation",
    "ring_closure",
    "processing",
    "other",
]

# String lengths at which spreadsheet, database and export tools commonly
# cut a field off.
TRUNCATION_LENGTHS = frozenset({255, 256, 1024, 4000, 32767})

# Values used to mean "no structure here".
_PLACEHOLDERS = frozenset(
    {"0", "na", "n/a", "nan", "none", "null", "nil", "-", "--", "?", ".", "not available"}
)

_CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x1f\x7f]|<NL>|<CR>|<TAB>", re.IGNORECASE)
# A bracketed word too long to be an atom, e.g. "[Gramicidin]".
_BRACKETED_NAME_RE = re.compile(r"^\[[A-Za-z][A-Za-z0-9 ,'\-]{4,}\]$")
_ONLY_FILLER_RE = re.compile(r"^[#\-?.\s*]+$")
# Characters of the SMILES organic subset; a string without any is not a SMILES.
_SMILES_ATOM_RE = re.compile(r"[BCNOPSFIbcnops]")
# R-groups ("[R]", "[R1]", a bare "R" that isn't the start of Rb/Ru/...),
# variable attachment ("[n=]"), repeat units ("(CH2)n") and generic atoms.
_MARKUSH_RE = re.compile(r"\[R\d*'?\]|R(?![a-z])|\[n=\]|\(CH2\)n|\[[XYQAG]\]")
# A bracket holding a fraction or decimal, e.g. "[0.5H2O]", "[1/2]", "[2.5]".
_STOICHIOMETRY_RE = re.compile(r"\[\s*\d+\s*[./]\s*\d*[^\]]*\]")
# Characters a complete SMILES can't end with.
_DANGLING_END = ("(", "[", "=", "#", "/", "\\", "-", "@", ".", "%")


def has_embedded_control_characters(smiles: str) -> bool:
    """Check for control characters inside a SMILES (not just around it).

    RDKit stops reading a SMILES at the first whitespace, so ``"CC\\nO"``
    silently parses as ethane: these must be treated as failures, not
    trusted.

    Parameters
    ----------
    smiles : str
        The raw SMILES value.

    Returns
    -------
    bool
        ``True`` if, after stripping surrounding whitespace, the value
        contains a control character or a ``<NL>``-style escape.
    """
    return bool(_CONTROL_CHARACTERS_RE.search(smiles.strip()))


def _looks_truncated(smiles: str) -> bool:
    """Check whether a SMILES looks cut off mid-structure.

    Parameters
    ----------
    smiles : str
        The stripped SMILES string.

    Returns
    -------
    bool
        ``True`` if it ends on a character that can't end a SMILES, or has
        more opening than closing branches or brackets.
    """
    return (
        smiles.endswith(_DANGLING_END)
        or smiles.count("(") > smiles.count(")")
        or smiles.count("[") > smiles.count("]")
    )


def _is_placeholder(smiles: str) -> bool:
    """Check whether a value is a placeholder rather than an attempted structure.

    Parameters
    ----------
    smiles : str
        The stripped SMILES string.

    Returns
    -------
    bool
        ``True`` for known "no value" markers, filler characters, a
        bracketed name, or a string with no SMILES atom characters at all.
    """
    return (
        smiles.lower() in _PLACEHOLDERS
        or bool(_ONLY_FILLER_RE.match(smiles))
        or bool(_BRACKETED_NAME_RE.match(smiles))
        or not _SMILES_ATOM_RE.search(smiles)
    )


def classify_parse_failure(raw_smiles: str, rdkit_message: str) -> ParseErrorCategory:
    """Assign a SMILES parse failure its most likely root cause.

    Checks run in order, so the most specific explanation wins: problems
    visible in the raw string first (control characters, truncation,
    placeholders, Markush notation, stoichiometry text), then RDKit's
    message (valence, kekulisation, ring closure).

    Parameters
    ----------
    raw_smiles : str
        The SMILES value exactly as read from the input file.
    rdkit_message : str
        RDKit's explanation of the failure (may be empty).

    Returns
    -------
    ParseErrorCategory
        The failure's category, ``"other"`` if nothing more specific fits.
    """
    smiles = raw_smiles.strip()
    message = rdkit_message.lower()
    if not smiles:
        return "empty"
    if has_embedded_control_characters(raw_smiles):
        return "control_characters"
    if len(raw_smiles) in TRUNCATION_LENGTHS:
        return "truncated"
    if _is_placeholder(smiles):
        return "placeholder"
    if _MARKUSH_RE.search(smiles):
        return "markush"
    if _STOICHIOMETRY_RE.search(smiles):
        return "stoichiometry_text"
    if "valence" in message:
        return "valence"
    if "kekulize" in message:
        return "kekulisation"
    if "ring" in message and ("unclosed" in message or "closure" in message):
        return "ring_closure"
    if _looks_truncated(smiles):
        return "truncated"
    return "other"
