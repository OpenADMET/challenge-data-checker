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
# A second CYP3A4 assay, published in a different paper, so one compound can
# be measured in two documents.
CYP3A4_ASSAY_2 = "CHEMBL_ASSAY_CYP2"
# An assay published in document 1 that also carries an activity whose own
# doc_id points somewhere else, which ChEMBL permits.
CYP3A4_ASSAY_3 = "CHEMBL_ASSAY_CYP3"
CYP3A4_DOCUMENT = "CHEMBL_DOC_1"
CYP3A4_DOCUMENT_2 = "CHEMBL_DOC_2"
CYP3A4_DOCUMENT_3 = "CHEMBL_DOC_3"

# A release's version table is keyed by name and need not hold only the
# database release, so a non-ChEMBL row is included, first, to prove the
# release is chosen by name rather than by row order.
OTHER_VERSION_NAME = "aa_ontology"
# Real releases also list the structure pipeline they used, under a name that
# starts with "ChEMBL" too (ChEMBL 37: "ChEMBL_Structure_Pipeline 1.2.0"). It
# is inserted before the release row, so matching on a "ChEMBL" prefix picks
# it instead of the release.
PIPELINE_VERSION_NAME = "ChEMBL_Structure_Pipeline 1.2.0"

# A protein-family target that has CYP3A4 among its components, as the
# real "Cytochrome P450" family (CHEMBL4523986) does. Resolving the gene
# symbol CYP3A4 to it would sweep in every P450.
P450_FAMILY_TARGET = "CHEMBL4523986"
P450_FAMILY_ASSAY = "CHEMBL_ASSAY_P450"

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
    (6, "CHEMBL112", "CC(=O)Nc1ccc(O)cc1", 6),
    (7, "CHEMBL13", "CC(C)NCC(O)COc1ccc(CCOC)cc1", 7),
    (8, "CHEMBL1200619", "CC(C)NCC(O)COc1ccc(CCOC)cc1.Cl", 7),
    (9, "CHEMBL190", "Cn1c(=O)c2[nH]cnc2n(C)c1=O", 9),
    (10, "CHEMBL3", "CN1CCC[C@H]1c1cccnc1", 10),
)

# Activities, as (molregno, assay, standard_type, value, own doc_id or None to
# inherit the assay's). A ``None`` value is a record with no measured result,
# such as an endpoint logged as "Not Determined".
#
#   1 caffeine        - CYP3A4 IC50
#   3 propranolol HCl - CYP3A4 IC50, reached from the free base via the parent
#   4 ibuprofen       - CYP3A4 Ki, so an IC50-only config must not match it
#   5 aspirin         - an unrelated target, so presence without a finding
#   6 paracetamol     - CYP3A4 but no measured result, so no answer is exposed
#   7 metoprolol      - CYP3A4 in document 2
#   8 metoprolol HCl  - CYP3A4 in document 1, so the free base reaches two
#                       documents, one of them only through its parent
#   9 theophylline    - an activity whose own doc_id differs from its assay's
ACTIVITIES: tuple[tuple[int, str, str, float | None, int | None], ...] = (
    (1, CYP3A4_ASSAY, "IC50", 45.0, None),
    (3, CYP3A4_ASSAY, "IC50", 120.0, None),
    (4, CYP3A4_ASSAY, "Ki", 3100.0, None),
    (5, KINASE_ASSAY, "IC50", 900.0, None),
    (6, CYP3A4_ASSAY, "IC50", None, None),
    (7, CYP3A4_ASSAY_2, "IC50", 61.0, None),
    (8, CYP3A4_ASSAY, "IC50", 72.0, None),
    (9, CYP3A4_ASSAY_3, "IC50", 88.0, 3),
    # 10 nicotine - measured only against the whole P450 family target
    (10, P450_FAMILY_ASSAY, "IC50", 50.0, None),
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
            (OTHER_VERSION_NAME, "2020-01-01", "an associated ontology, not the release"),
        )
        conn.execute(
            "INSERT INTO version (name, creation_date, comments) VALUES (?, ?, ?)",
            (PIPELINE_VERSION_NAME, None, "structure pipeline version, not the release"),
        )
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

        # The family shares CYP3A4's component (component_id 1), so the gene
        # symbol resolves to it as well as to the single protein.
        conn.execute(
            """INSERT INTO target_dictionary (tid, chembl_id, pref_name, target_type, organism)
               VALUES (?, ?, ?, ?, ?)""",
            (3, P450_FAMILY_TARGET, "Cytochrome P450", "PROTEIN FAMILY", "Homo sapiens"),
        )
        conn.execute("INSERT INTO target_components (tid, component_id) VALUES (?, ?)", (3, 1))

        documents = {
            CYP3A4_DOCUMENT: (1, "J Med Chem", 2019, "10.0000/fixture", "A CYP3A4 panel"),
            CYP3A4_DOCUMENT_2: (2, "J Pharm Sci", 2021, "10.0000/fixture2", "Another panel"),
            CYP3A4_DOCUMENT_3: (3, "Drug Metab Dispos", 2022, "10.0000/fixture3", "A third"),
        }
        for chembl_id, (doc_id, journal, year, doi, title) in documents.items():
            conn.execute(
                """INSERT INTO docs (doc_id, chembl_id, journal, year, doi, title)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, chembl_id, journal, year, doi, title),
            )
        # assay chembl_id -> (assay_id, tid, the assay's own doc_id)
        assays = {
            CYP3A4_ASSAY: (1, 1, 1),
            KINASE_ASSAY: (2, 2, None),
            CYP3A4_ASSAY_2: (3, 1, 2),
            CYP3A4_ASSAY_3: (4, 1, 1),
            P450_FAMILY_ASSAY: (5, 3, 1),
        }
        for chembl_id, (assay_id, tid, doc_id) in assays.items():
            conn.execute(
                """INSERT INTO assays (assay_id, chembl_id, description, assay_type, tid, doc_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (assay_id, chembl_id, f"{chembl_id} description", "B", tid, doc_id),
            )

        for activity_id, activity in enumerate(ACTIVITIES, start=1):
            molregno, assay, standard_type, value, own_doc_id = activity
            assay_id, _tid, assay_doc_id = assays[assay]
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
                    "=" if value is not None else None,
                    value,
                    "nM" if value is not None else None,
                    None,
                    own_doc_id if own_doc_id is not None else assay_doc_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return path
