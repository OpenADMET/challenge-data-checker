"""Cheminformatics quality-filter heuristics: mixtures, salts/metals, suspicious fragments."""

from rdkit import Chem

# Transition metals, lanthanides/actinides, and other heavy metals commonly seen
# either as the metal centre of a complex or stripped away from one.
METAL_SYMBOLS = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Fr",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Ra",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Al",
    "Ga",
    "In",
    "Sn",
    "Tl",
    "Pb",
    "Bi",
    "Sb",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Ac",
    "Th",
    "U",
    "Np",
    "Pu",
}

# Canonical SMILES of common small counterions/fragments seen in salt forms,
# including ones that look highly suspicious as a *standalone* entry (i.e. as
# though the metal or parent structure was stripped away by a broken pipeline).
_COUNTERION_SMILES_RAW = {
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[F-]",
    "[Na+]",
    "[K+]",
    "[Li+]",
    "[Ca+2]",
    "[Mg+2]",
    "[NH4+]",
    "[OH-]",
    "O=S(=O)([O-])[O-]",  # sulfate
    "O=P([O-])([O-])[O-]",  # phosphate
    "CC(=O)[O-]",  # acetate
    "CS(=O)(=O)[O-]",  # mesylate
    "OC(=O)C(=O)O",  # oxalic acid
    "[O-]C(=O)C(=O)[O-]",  # oxalate
    "OC(=O)/C=C/C(=O)O",  # fumaric acid
    "C#N",  # bare cyanide/nitrile fragment
    "[C-]#N",  # cyanide ion
    "N",  # ammonia
    "C",  # bare methyl/methane fragment
}


def _canonical(smiles: str) -> str | None:
    """Canonicalise a SMILES string, tolerating unparseable input.

    Parameters
    ----------
    smiles : str
        The SMILES string to canonicalise.

    Returns
    -------
    str | None
        The RDKit canonical SMILES, or ``None`` if parsing failed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


COUNTERION_CANONICAL_SMILES = {
    c for smi in _COUNTERION_SMILES_RAW if (c := _canonical(smi)) is not None
}


def has_multiple_fragments(mol: Chem.Mol, raw_smiles: str) -> bool:
    """Check whether a molecule is a mixture of multiple disconnected fragments.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.
    raw_smiles : str
        The original SMILES string the molecule was parsed from.

    Returns
    -------
    bool
        ``True`` if the SMILES contains a ``.`` or the parsed molecule has
        more than one fragment, ``False`` otherwise.
    """
    if "." in raw_smiles:
        return True
    return len(Chem.GetMolFrags(mol)) > 1


def contains_metal(mol: Chem.Mol) -> bool:
    """Check whether a molecule contains a metal atom.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.

    Returns
    -------
    bool
        ``True`` if any atom is a transition/heavy/alkali(-earth) metal,
        ``False`` otherwise.
    """
    return any(atom.GetSymbol() in METAL_SYMBOLS for atom in mol.GetAtoms())


def is_salt_or_metal_complex(mol: Chem.Mol) -> bool:
    """Check whether a molecule is a salt or metal complex.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.

    Returns
    -------
    bool
        ``True`` if the molecule contains a metal, or any of its fragments
        is a known counterion, ``False`` otherwise.
    """
    if contains_metal(mol):
        return True
    for frag in Chem.GetMolFrags(mol, asMols=True):
        frag_smiles = Chem.MolToSmiles(frag, canonical=True)
        if frag_smiles in COUNTERION_CANONICAL_SMILES:
            return True
    return False


def is_suspicious_small_fragment(mol: Chem.Mol, heavy_atom_count: int) -> bool:
    """Check whether a molecule looks like a stripped-degradation artefact.

    Flags small entries (<=3 heavy atoms) such as a bare cyanide, isolated
    methyl group, or lone counterion left behind by a naive/failed
    preprocessing pipeline.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.
    heavy_atom_count : int
        The molecule's heavy atom count.

    Returns
    -------
    bool
        ``True`` if the molecule is a suspiciously small fragment,
        ``False`` otherwise.
    """
    if heavy_atom_count > 3:
        return False
    has_carbon = any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())
    return has_carbon or heavy_atom_count <= 1
