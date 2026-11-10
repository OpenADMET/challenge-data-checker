"""Build a tiny stand-in for a ChEMBL SQLite release, for the public-search tests.

Only the tables and columns ``chembl.py`` queries are created, with a handful
of compounds chosen to exercise one matching path each. The InChIKeys are
computed with the checker's own ``MoleculeProcessor`` rather than written in
by hand, so the fixture is consistent with whatever RDKit version is
installed and the tests exercise the matching logic instead of InChI
generation.

A generated fixture is used rather than a committed ``.sqlite`` binary so
that what the fixture contains is reviewable in the diff.
"""

import sqlite3
from pathlib import Path

from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import SettingsConfig

_PROCESSOR = MoleculeProcessor(SettingsConfig())

RELEASE_NAME = "CHEMBL_99"
RELEASE_DATE = "2026-01-15"

# The CYP3A4 target, which the challenge endpoint in these tests is about.
CYP3A4_TARGET = "CHEMBL340"
CYP3A4_TARGET_NAME = "Cytochrome P450 3A4"
# An unrelated target, so a compound can be in ChEMBL with measurements that
# are not the challenge endpoint.
KINASE_TARGET = "CHEMBL999"
KINASE_TARGET_NAME = "Serine/threonine-protein kinase PLK1"

CYP3A4_ASSAY = "CHEMBL_ASSAY_CYP"
KINASE_ASSAY = "CHEMBL_ASSAY_KIN"
CYP3A4_DOCUMENT = "CHEMBL_DOC_1"

# Compounds, as (molregno, chembl_id, smiles, parent_molregno).
#
#   1 caffeine                 - exact InChIKey match, CYP3A4 data -> blocking
#   2 propranolol              - the parent of 3
#   3 propranolol hydrochloride- a salt whose parent is 2, CYP3A4 data, so a
#                                test set holding the free base matches it
#                                through molecule_hierarchy
#   4 racemic ibuprofen        - connectivity-only match for (S)-ibuprofen,
#                                with CYP3A4 data
#   5 aspirin                  - present in ChEMBL but only measured against
#                                an unrelated target, so presence without a
#                                finding
COMPOUNDS: tuple[tuple[int, str, str, int], ...] = (
    (1, "CHEMBL113", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", 1),
    (2, "CHEMBL27", "CC(C)NCC(O)COc1cccc2ccccc12", 2),
    (3, "CHEMBL1200640", "CC(C)NCC(O)COc1cccc2ccccc12.Cl", 2),
    (4, "CHEMBL521", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", 4),
    (5, "CHEMBL25", "CC(=O)Oc1ccccc1C(=O)O", 5),
)

# Activities, as (molregno, assay, target, target name, standard_type, value,
# units, document).
ACTIVITIES: tuple[tuple[int, str, str, str, str, float, str, str | None], ...] = (
    (1, CYP3A4_ASSAY, CYP3A4_TARGET, CYP3A4_TARGET_NAME, "IC50", 45.0, "nM", CYP3A4_DOCUMENT),
    (3, CYP3A4_ASSAY, CYP3A4_TARGET, CYP3A4_TARGET_NAME, "IC50", 120.0, "nM", CYP3A4_DOCUMENT),
    (4, CYP3A4_ASSAY, CYP3A4_TARGET, CYP3A4_TARGET_NAME, "Ki", 3100.0, "nM", CYP3A4_DOCUMENT),
    (5, KINASE_ASSAY, KINASE_TARGET, KINASE_TARGET_NAME, "IC50", 900.0, "nM", None),
)

_SCHEMA = """
CREATE TABLE version (name TEXT, creation_date TEXT, comments TEXT);
CREATE TABLE molecule_dictionary (molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT);
CREATE TABLE compound_structures (
    molregno INTEGER PRIMARY KEY,
    standard_inchi TEXT,
    standard_inchi_key TEXT,
    canonical_smiles TEXT
);
CREATE INDEX idx_cs_inchikey ON compound_structures (standard_inchi_key);
CREATE TABLE molecule_hierarchy (
    molregno INTEGER PRIMARY KEY,
    parent_molregno INTEGER,
    active_molregno INTEGER
);
CREATE TABLE target_dictionary (
    tid INTEGER PRIMARY KEY,
    chembl_id TEXT,
    pref_name TEXT,
    target_type TEXT,
    organism TEXT
);
CREATE TABLE component_synonyms (
    compsyn_id INTEGER PRIMARY KEY,
    component_id INTEGER,
    component_synonym TEXT,
    syn_type TEXT
);
CREATE TABLE target_components (
    tid INTEGER,
    component_id INTEGER
);
CREATE TABLE docs (
    doc_id INTEGER PRIMARY KEY,
    chembl_id TEXT,
    journal TEXT,
    year INTEGER,
    doi TEXT,
    title TEXT
);
CREATE TABLE assays (
    assay_id INTEGER PRIMARY KEY,
    chembl_id TEXT,
    description TEXT,
    assay_type TEXT,
    tid INTEGER,
    doc_id INTEGER
);
CREATE TABLE activities (
    activity_id INTEGER PRIMARY KEY,
    assay_id INTEGER,
    molregno INTEGER,
    standard_type TEXT,
    standard_relation TEXT,
    standard_value REAL,
    standard_units TEXT,
    pchembl_value REAL,
    doc_id INTEGER
);
CREATE INDEX idx_act_molregno ON activities (molregno);
"""


def inchikey_of(smiles: str) -> str:
    """Compute a SMILES' InChIKey the way the checker computes it.

    Parameters
    ----------
    smiles : str
        The SMILES to key.

    Returns
    -------
    str
        Its InChIKey.

    Raises
    ------
    ValueError
        If the SMILES does not parse, which would make the fixture
        meaningless.
    """
    record = _PROCESSOR.process_row("chembl", "<fixture>", 0, smiles, {})
    if record.inchikey is None:
        raise ValueError(f"fixture SMILES did not yield an InChIKey: {smiles}")
    return record.inchikey


def build_fixture(path: str | Path) -> Path:
    """Write a minimal ChEMBL-shaped SQLite release to ``path``.

    Parameters
    ----------
    path : str | Path
        Where to create the database file.

    Returns
    -------
    Path
        The path written to.
    """
    path = Path(path)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO version (name, creation_date, comments) VALUES (?, ?, ?)",
            (RELEASE_NAME, RELEASE_DATE, "test fixture"),
        )
        for molregno, chembl_id, smiles, parent in COMPOUNDS:
            conn.execute(
                "INSERT INTO molecule_dictionary (molregno, chembl_id, pref_name) VALUES (?, ?, ?)",
                (molregno, chembl_id, None),
            )
            conn.execute(
                """INSERT INTO compound_structures
                   (molregno, standard_inchi, standard_inchi_key, canonical_smiles)
                   VALUES (?, ?, ?, ?)""",
                (molregno, None, inchikey_of(smiles), smiles),
            )
            conn.execute(
                """INSERT INTO molecule_hierarchy (molregno, parent_molregno, active_molregno)
                   VALUES (?, ?, ?)""",
                (molregno, parent, parent),
            )

        targets = {
            CYP3A4_TARGET: (1, CYP3A4_TARGET_NAME, "CYP3A4"),
            KINASE_TARGET: (2, KINASE_TARGET_NAME, "PLK1"),
        }
        for chembl_id, (tid, pref_name, synonym) in targets.items():
            conn.execute(
                """INSERT INTO target_dictionary
                   (tid, chembl_id, pref_name, target_type, organism)
                   VALUES (?, ?, ?, ?, ?)""",
                (tid, chembl_id, pref_name, "SINGLE PROTEIN", "Homo sapiens"),
            )
            conn.execute(
                """INSERT INTO component_synonyms
                   (compsyn_id, component_id, component_synonym, syn_type)
                   VALUES (?, ?, ?, ?)""",
                (tid, tid, synonym, "GENE_SYMBOL"),
            )
            conn.execute(
                "INSERT INTO target_components (tid, component_id) VALUES (?, ?)", (tid, tid)
            )

        conn.execute(
            """INSERT INTO docs (doc_id, chembl_id, journal, year, doi, title)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (1, CYP3A4_DOCUMENT, "J Med Chem", 2019, "10.0000/fixture", "A CYP3A4 panel"),
        )
        assays = {CYP3A4_ASSAY: (1, 1, 1), KINASE_ASSAY: (2, 2, None)}
        for chembl_id, (assay_id, tid, doc_id) in assays.items():
            conn.execute(
                """INSERT INTO assays (assay_id, chembl_id, description, assay_type, tid, doc_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (assay_id, chembl_id, f"{chembl_id} description", "B", tid, doc_id),
            )

        for activity_id, activity in enumerate(ACTIVITIES, start=1):
            molregno, assay, _target, _name, standard_type, value, units, document = activity
            assay_id = assays[assay][0]
            conn.execute(
                """INSERT INTO activities
                   (activity_id, assay_id, molregno, standard_type, standard_relation,
                    standard_value, standard_units, pchembl_value, doc_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    activity_id,
                    assay_id,
                    molregno,
                    standard_type,
                    "=",
                    value,
                    units,
                    None,
                    1 if document else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return path
