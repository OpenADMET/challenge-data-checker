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
from typing import NamedTuple

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
# The paper that first measured compound 13. Document 1 only cites it, which
# ChEMBL flags as a potential duplicate, so only this one counts as coverage.
CYP3A4_DOCUMENT_4 = "CHEMBL_DOC_4"

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

# Compounds, as (molregno, chembl_id, smiles, parent_molregno), where a
# parent_molregno of None means the compound has no molecule_hierarchy row.
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
#  11 warfarin                 - a CYP3A4 result recorded as text with no
#                                number, which is still an answer
#  12 diclofenac               - its CYP3A4 number carries a
#                                data_validity_comment, so the finding has to
#                                survive and say so
#  13 naproxen                 - measured in document 4 and cited again in
#                                document 1, the citation flagged as a
#                                potential duplicate
#  14 a quinoxaline free base  - the parent of 15 with no molecule_hierarchy
#                                row of its own, as ChEMBL registers a parent
#                                it only holds as a salt (93,019 parents in
#                                ChEMBL 37, this one among them)
#  15 its hydrochloride        - a salt whose parent is 14
COMPOUNDS: tuple[tuple[int, str, str, int | None], ...] = (
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
    (11, "CHEMBL1464", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", 11),
    (12, "CHEMBL139", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl", 12),
    (13, "CHEMBL154", "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1", 13),
    (14, "CHEMBL7022", "Clc1nc2ccccc2nc1N1CCNCC1", None),
    (15, "CHEMBL1743921", "Cl.Clc1nc2ccccc2nc1N1CCNCC1", 14),
)


class Activity(NamedTuple):
    """One activity row of the fixture release.

    The defaults are the ordinary case - a numeric result, no curation
    comment - so a row only states the column it is in the fixture to
    exercise.

    Attributes
    ----------
    molregno : int
        The compound measured.
    assay : str
        ChEMBL ID of the assay, a key of ``assays`` in ``build_fixture``.
    standard_type : str
        The measurement type, e.g. ``IC50``.
    value : float | None
        The numeric result, or ``None`` for a record with none.
    own_doc_id : int | None
        The activity's own ``doc_id``, or ``None`` to inherit its assay's.
    text_value : str | None
        A categorical result in ``standard_text_value``, which ChEMBL uses
        instead of a number rather than as well as one.
    activity_comment : str | None
        ChEMBL's note on the measurement, e.g. ``"Not Determined"`` for an
        endpoint that was looked at but never measured.
    data_validity_comment : str | None
        ChEMBL's curation flag on the number itself, e.g. ``"Potential
        transcription error"``.
    potential_duplicate : int
        ``1`` where ChEMBL judges the row a repeat citation of an earlier
        paper's value rather than a new measurement.
    """

    molregno: int
    assay: str
    standard_type: str
    value: float | None = None
    own_doc_id: int | None = None
    text_value: str | None = None
    activity_comment: str | None = None
    data_validity_comment: str | None = None
    potential_duplicate: int = 0


# Activities.
#
#   1 caffeine        - CYP3A4 IC50
#   3 propranolol HCl - CYP3A4 IC50, reached from the free base via the parent
#   4 ibuprofen       - CYP3A4 Ki, so an IC50-only config must not match it
#   5 aspirin         - an unrelated target, so presence without a finding
#   6 paracetamol     - a CYP3A4 endpoint ChEMBL marks "Not Determined", so it
#                       was looked at, not measured, and exposes no answer
#   7 metoprolol      - CYP3A4 in document 2
#   8 metoprolol HCl  - CYP3A4 in document 1, so the free base reaches two
#                       documents, one of them only through its parent
#   9 theophylline    - an activity whose own doc_id differs from its assay's
#  11 warfarin        - a CYP3A4 result held as text with no number, which is
#                       an answer all the same
#  12 diclofenac      - a CYP3A4 number ChEMBL flags as possibly transcribed
#                       wrong, which is still a published answer
#  13 naproxen        - measured once in document 4, and once more in document
#                       1 as a repeat citation. Only document 4 should count
#                       towards per-document coverage
ACTIVITIES: tuple[Activity, ...] = (
    Activity(1, CYP3A4_ASSAY, "IC50", 45.0),
    Activity(3, CYP3A4_ASSAY, "IC50", 120.0),
    Activity(4, CYP3A4_ASSAY, "Ki", 3100.0),
    Activity(5, KINASE_ASSAY, "IC50", 900.0),
    Activity(6, CYP3A4_ASSAY, "IC50", activity_comment="Not Determined"),
    Activity(7, CYP3A4_ASSAY_2, "IC50", 61.0),
    Activity(8, CYP3A4_ASSAY, "IC50", 72.0),
    Activity(9, CYP3A4_ASSAY_3, "IC50", 88.0, own_doc_id=3),
    # 10 nicotine - measured only against the whole P450 family target
    Activity(10, P450_FAMILY_ASSAY, "IC50", 50.0),
    Activity(11, CYP3A4_ASSAY, "Activity", text_value="Not active"),
    Activity(
        12,
        CYP3A4_ASSAY,
        "IC50",
        5.0,
        data_validity_comment="Potential transcription error",
    ),
    Activity(13, CYP3A4_ASSAY, "IC50", 155.0, own_doc_id=4),
    Activity(13, CYP3A4_ASSAY, "IC50", 150.0, potential_duplicate=1),
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
    standard_text_value TEXT,
    activity_comment TEXT,
    data_validity_comment TEXT,
    potential_duplicate INTEGER,
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
            if parent is None:
                continue
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
            CYP3A4_DOCUMENT_4: (4, "Xenobiotica", 2023, "10.0000/fixture4", "The original"),
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
            assay_id, _tid, assay_doc_id = assays[activity.assay]
            value = activity.value
            conn.execute(
                """INSERT INTO activities
                   (activity_id, assay_id, molregno, standard_type, standard_relation,
                    standard_value, standard_units, pchembl_value, standard_text_value,
                    activity_comment, data_validity_comment, potential_duplicate, doc_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    activity_id,
                    assay_id,
                    activity.molregno,
                    activity.standard_type,
                    "=" if value is not None else None,
                    value,
                    "nM" if value is not None else None,
                    None,
                    activity.text_value,
                    activity.activity_comment,
                    activity.data_validity_comment,
                    activity.potential_duplicate,
                    activity.own_doc_id if activity.own_doc_id is not None else assay_doc_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return path
