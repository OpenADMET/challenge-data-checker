"""Cheminformatics quality-filter heuristics: mixtures, salts/metals, suspicious fragments."""

from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

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

# Counterions seen in salt forms. Each entry may be drawn ionic or neutral:
# fragments are compared after neutralisation and ignoring stereochemistry
# (see _fragment_key), so "[Cl-]" also matches "Cl", sulfate matches sulfuric
# acid, and maleate matches fumarate. Single metal ions (Na+, K+, ...) are
# recognised separately via METAL_SYMBOLS.
_COUNTERION_SMILES_RAW = {
    # Inorganic acids / anions
    "Cl",  # hydrochloride
    "Br",  # hydrobromide
    "I",  # hydroiodide
    "F",  # hydrofluoride
    "OS(=O)(=O)O",  # sulfate / hydrogen sulfate
    "OP(=O)(O)O",  # phosphate
    "O=[N+]([O-])O",  # nitrate
    "OCl(=O)(=O)=O",  # perchlorate
    "F[B-](F)(F)F",  # tetrafluoroborate
    "F[P-](F)(F)(F)(F)F",  # hexafluorophosphate
    "[C-]#N",  # cyanide
    # Organic acids / anions
    "OC=O",  # formate
    "CC(=O)O",  # acetate
    "OC(=O)C(F)(F)F",  # trifluoroacetate (TFA)
    "CS(=O)(=O)O",  # mesylate
    "Cc1ccc(S(=O)(=O)O)cc1",  # tosylate
    "OS(=O)(=O)c1ccccc1",  # besylate
    "OS(=O)(=O)c1ccc2ccccc2c1",  # napsylate
    "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2",  # camsylate
    "OC(=O)C(=O)O",  # oxalate
    "OC(=O)C=CC(=O)O",  # maleate / fumarate
    "OC(=O)CCC(=O)O",  # succinate
    "OC(=O)CC(O)C(=O)O",  # malate
    "OC(=O)C(O)C(O)C(=O)O",  # tartrate
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # citrate
    "CC(O)C(=O)O",  # lactate
    "OC(=O)c1ccccc1",  # benzoate
    "OC(=O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O",  # pamoate
    "Oc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",  # picrate
    # Bases / cations
    "N",  # ammonium
    "OCCN",  # ethanolamine
    "NC(CO)(CO)CO",  # tromethamine
    "CNCC(O)C(O)C(O)C(O)CO",  # meglumine
    "C[N+](C)(C)CCO",  # choline
}

# Solvents seen as solvates or co-crystallised fragments.
_SOLVENT_SMILES_RAW = {
    "O",  # water (hydrate)
    "CO",  # methanol
    "CCO",  # ethanol
    "CC(C)O",  # isopropanol
    "CC(C)=O",  # acetone
    "CC#N",  # acetonitrile
    "CS(C)=O",  # DMSO
    "CN(C)C=O",  # DMF
    "CCOC(C)=O",  # ethyl acetate
    "CCOCC",  # diethyl ether
    "C1CCOC1",  # THF
    "C1COCCO1",  # 1,4-dioxane
    "ClCCl",  # dichloromethane
    "ClC(Cl)Cl",  # chloroform
    "Cc1ccccc1",  # toluene
    "CCCCCC",  # hexane
}

_UNCHARGER = rdMolStandardize.Uncharger()


def _fragment_key(frag: Chem.Mol) -> str:
    """Build the lookup key for a fragment: neutralised, stereo-blind SMILES.

    Parameters
    ----------
    frag : Chem.Mol
        A single connected fragment.

    Returns
    -------
    str
        Canonical SMILES of the neutralised fragment, without
        stereochemistry.
    """
    return Chem.MolToSmiles(_UNCHARGER.uncharge(frag), isomericSmiles=False)


def _fragment_keys(smiles_set: set[str]) -> set[str]:
    """Build the lookup keys for a set of fragment SMILES.

    Parameters
    ----------
    smiles_set : set[str]
        SMILES strings, each a single fragment.

    Returns
    -------
    set[str]
        The ``_fragment_key`` of each parseable entry.
    """
    return {_fragment_key(m) for smi in smiles_set if (m := Chem.MolFromSmiles(smi)) is not None}


COUNTERION_KEYS = _fragment_keys(_COUNTERION_SMILES_RAW)
SOLVENT_KEYS = _fragment_keys(_SOLVENT_SMILES_RAW)

type FragmentKind = Literal["counterion", "solvent", "parent_copy", "unknown"]

# Default minimum heavy-atom count for an unrecognised fragment to count as a
# separate component in the multi_component check.
DEFAULT_MULTI_COMPONENT_MIN_HEAVY_ATOMS = 6


@dataclass(frozen=True)
class Fragment:
    """One connected fragment of a molecule.

    Attributes
    ----------
    smiles : str
        The fragment's canonical SMILES (as drawn, not neutralised).
    kind : FragmentKind
        ``"counterion"`` or ``"solvent"`` if recognised, ``"parent_copy"``
        for a further copy of the parent (e.g. in a 2:1 salt), otherwise
        ``"unknown"``.
    heavy_atoms : int
        The fragment's heavy-atom count.
    """

    smiles: str
    kind: FragmentKind
    heavy_atoms: int


@dataclass(frozen=True)
class FragmentClassification:
    """A molecule split into its parent and the fragments stripped from it.

    Attributes
    ----------
    parent : Chem.Mol
        The neutralised parent fragment: the largest fragment that isn't
        a known counterion or solvent, or the largest fragment overall if
        every fragment is known.
    parent_fragment : Fragment
        The parent as drawn (before neutralisation).
    parent_is_known : bool
        Whether the parent is itself a known counterion or solvent, i.e.
        the molecule has no other fragment to be the "real" compound.
    others : tuple[Fragment, ...]
        Every non-parent fragment.
    """

    parent: Chem.Mol
    parent_fragment: Fragment
    parent_is_known: bool
    others: tuple[Fragment, ...]

    @property
    def stripped(self) -> tuple[tuple[str, FragmentKind], ...]:
        """Canonical SMILES and kind of every non-parent fragment.

        Returns
        -------
        tuple[tuple[str, FragmentKind], ...]
            One ``(smiles, kind)`` pair per non-parent fragment.
        """
        return tuple((f.smiles, f.kind) for f in self.others)

    @property
    def is_salt_or_solvate(self) -> bool:
        """Whether the molecule is a parent compound plus only known counterions/solvents.

        Further copies of the parent are allowed (a 2:1 salt), but at least
        one fragment must be a known counterion or solvent.

        Returns
        -------
        bool
            ``True`` if the parent isn't itself a known counterion/solvent,
            every other fragment is a counterion, solvent or parent copy,
            and at least one is a counterion or solvent.
        """
        kinds = {f.kind for f in self.others}
        return (
            not self.parent_is_known
            and kinds <= {"counterion", "solvent", "parent_copy"}
            and bool(kinds & {"counterion", "solvent"})
        )

    @property
    def is_homodimer_without_counterion(self) -> bool:
        """Whether the molecule is two or more copies of one compound and nothing else.

        Typically a 2:1 salt whose counterion was stripped, leaving the
        parent doubled: the molecular weight and stoichiometry are wrong.

        Returns
        -------
        bool
            ``True`` if there are other fragments and all are copies of
            an unrecognised parent.
        """
        return bool(self.others) and all(f.kind == "parent_copy" for f in self.others)

    def is_multi_component(self, min_heavy_atoms: int) -> bool:
        """Whether the molecule holds two or more distinct unrecognised compounds.

        For example two unrelated drugs concatenated into one SMILES.

        Parameters
        ----------
        min_heavy_atoms : int
            Minimum heavy atoms for an unrecognised fragment to count as a
            compound in its own right.

        Returns
        -------
        bool
            ``True`` if at least two distinct unrecognised fragments
            (counting the parent) each have ``min_heavy_atoms`` or more.
        """
        candidates = [self.parent_fragment, *self.others]
        large = {
            f.smiles for f in candidates if f.kind == "unknown" and f.heavy_atoms >= min_heavy_atoms
        }
        return len(large) >= 2

    def is_unclassified_mixture(self, min_heavy_atoms: int) -> bool:
        """Whether the molecule has several fragments that no other flag explains.

        Parameters
        ----------
        min_heavy_atoms : int
            The ``multi_component`` threshold (see ``is_multi_component``).

        Returns
        -------
        bool
            ``True`` for a multi-fragment molecule that isn't a salt or
            solvate, a homodimer, or multi-component, e.g. a compound with
            a small unrecognised fragment, or only counterion-like
            fragments.
        """
        return (
            bool(self.others)
            and not self.is_salt_or_solvate
            and not self.is_homodimer_without_counterion
            and not self.is_multi_component(min_heavy_atoms)
        )


def fragment_kind(frag: Chem.Mol) -> FragmentKind:
    """Classify a single fragment as a counterion, solvent, or unknown.

    Parameters
    ----------
    frag : Chem.Mol
        A single connected fragment.

    Returns
    -------
    FragmentKind
        ``"counterion"`` for a lone metal ion or a known counterion,
        ``"solvent"`` for a known solvent, ``"unknown"`` otherwise.
    """
    if frag.GetNumAtoms() == 1 and frag.GetAtomWithIdx(0).GetSymbol() in METAL_SYMBOLS:
        return "counterion"
    key = _fragment_key(frag)
    if key in COUNTERION_KEYS:
        return "counterion"
    if key in SOLVENT_KEYS:
        return "solvent"
    return "unknown"


def classify_fragments(mol: Chem.Mol) -> FragmentClassification:
    """Split a molecule into its parent fragment and stripped counterions/solvents.

    The parent is the largest fragment (by heavy atoms, then exact mass,
    then SMILES, for determinism) that isn't a known counterion or solvent.
    Choosing among unknown fragments first means a large counterion such as
    pamoate is never picked over a smaller drug. The parent is neutralised,
    so a protonated amine hydrochloride and its free base share a parent.
    Other fragments identical to an unrecognised parent are marked as
    parent copies.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.

    Returns
    -------
    FragmentClassification
        The parent and the classified other fragments.
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    smiles = [Chem.MolToSmiles(f) for f in frags]
    kinds = [fragment_kind(f) for f in frags]
    unknown = [i for i, kind in enumerate(kinds) if kind == "unknown"]
    candidates = unknown or list(range(len(frags)))
    parent_idx = max(
        candidates,
        key=lambda i: (
            frags[i].GetNumHeavyAtoms(),
            rdMolDescriptors.CalcExactMolWt(frags[i]),
            smiles[i],
        ),
    )
    parent_is_known = kinds[parent_idx] != "unknown"

    def fragment(i: int) -> Fragment:
        kind = kinds[i]
        if i != parent_idx and not parent_is_known and smiles[i] == smiles[parent_idx]:
            kind = "parent_copy"
        return Fragment(smiles[i], kind, frags[i].GetNumHeavyAtoms())

    return FragmentClassification(
        parent=_UNCHARGER.uncharge(frags[parent_idx]),
        parent_fragment=fragment(parent_idx),
        parent_is_known=parent_is_known,
        others=tuple(fragment(i) for i in range(len(frags)) if i != parent_idx),
    )


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


# Legitimate small molecules (<=3 heavy atoms, containing carbon) that would
# otherwise trip the suspicious-fragment size check: common solvents, simple
# reagents, and small entries in solubility or fragment-screening datasets.
# Bare "C" and "C#N" are deliberately absent: as standalone entries they are
# more likely stripped-pipeline artefacts.
_SMALL_MOLECULE_ALLOWLIST_RAW = {
    "CO",  # methanol
    "CCO",  # ethanol
    "CC#N",  # acetonitrile
    "C=O",  # formaldehyde
    "CC=O",  # acetaldehyde
    "O=CO",  # formic acid
    "NC=O",  # formamide
    "CN",  # methylamine
    "CCN",  # ethylamine
    "CNC",  # dimethylamine
    "COC",  # dimethyl ether
    "C1CO1",  # ethylene oxide
    "CC",  # ethane
    "CCC",  # propane
    "C=C",  # ethylene
    "C=CC",  # propene
    "C#C",  # acetylene
    "CF",  # fluoromethane
    "CCl",  # chloromethane
    "CBr",  # bromomethane
    "CI",  # iodomethane
    "ClCCl",  # dichloromethane
    "CS",  # methanethiol
    "O=C=O",  # carbon dioxide
    "S=C=S",  # carbon disulfide
}

SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES = {
    c for smi in _SMALL_MOLECULE_ALLOWLIST_RAW if (c := _canonical(smi)) is not None
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
    """Check whether a molecule is a salt, solvate or metal complex.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.

    Returns
    -------
    bool
        ``True`` if the molecule contains a metal, or is a parent compound
        whose other fragments are all known counterions or solvents (see
        ``classify_fragments``), ``False`` otherwise.
    """
    return contains_metal(mol) or classify_fragments(mol).is_salt_or_solvate


def is_suspicious_small_fragment(mol: Chem.Mol, heavy_atom_count: int) -> bool:
    """Check whether a molecule looks like a stripped-degradation artefact.

    Flags small entries (<=3 heavy atoms) such as a bare cyanide, isolated
    methyl group, or lone counterion left behind by a naive/failed
    preprocessing pipeline. Legitimate small molecules on the allowlist
    (methanol, ethanol, acetonitrile, ...; see
    ``SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES``) are not flagged.

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
    if Chem.MolToSmiles(mol, canonical=True) in SMALL_MOLECULE_ALLOWLIST_CANONICAL_SMILES:
        return False
    has_carbon = any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())
    return has_carbon or heavy_atom_count <= 1
