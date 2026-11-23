"""Read-only lookups against a local ChEMBL SQLite release.

The ``public-search`` mode asks a different question from the audit: not
"does train overlap test?" but "is the blinded test set already public?".
Answering it needs a public database, and this module is the ChEMBL half of
that, deliberately kept behind a narrow interface so another source
(PubChem, BindingDB) can be added without touching the matching logic.

A **local release** is used rather than the REST API, for two reasons. The
test set of a blind challenge is the one thing meant to stay secret until
the challenge closes, and sending its InChIKeys to a public endpoint
discloses exactly that. And a release number makes a report reproducible: a
live query against a database that changes underneath you can be repeated
but not reproduced.

Only ``sqlite3`` from the standard library is needed, so this adds no
dependency. The connection is opened read-only, and the test set's lookup
keys go into a TEMP table (which lives in the separate temp database, so a
read-only connection can still create one) rather than into a long ``IN``
list, which keeps every query to one statement regardless of test-set size.

The activity query reads ``activities.standard_text_value``, which ChEMBL
added in release 26, so release 26 or newer is needed. An older release
fails with the ``ChemblSourceError`` ``find_activities`` raises for any
schema mismatch, rather than silently finding nothing.
"""

import logging
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from challenge_data_checker.report_types import (
    ChemblActivity,
    ChemblCompound,
    ChemblSource,
    Representation,
)

logger = logging.getLogger(__name__)

# Tables the search needs. Checked up front so a wrong path or a partial
# dump fails with one clear message instead of a sqlite3 error deep inside
# the first query.
REQUIRED_TABLES: tuple[str, ...] = (
    "compound_structures",
    "molecule_hierarchy",
    "molecule_dictionary",
    "activities",
    "assays",
    "target_dictionary",
)

# How a release names itself in its version table, e.g. "ChEMBL_37", and
# how it describes itself, e.g. "ChEMBL Release 37 (https://...)".
_RELEASE_NAME = re.compile(r"chembl_?\d+", re.IGNORECASE)
_RELEASE_COMMENT = re.compile(r"chembl release \d+", re.IGNORECASE)

# Target types a gene symbol resolves to unless the configuration widens it.
# A symbol such as CYP3A4 is also a component of protein-family targets, the
# broadest being every cytochrome P450 at once, so resolving to those would
# flag a CYP2D6 measurement filed under the family as a CYP3A4 leak.
DEFAULT_TARGET_TYPES: tuple[str, ...] = ("SINGLE PROTEIN",)

# The length of an InChIKey's first block, which encodes connectivity only
# and so is shared by stereoisomers and racemate/enantiomer pairs.
CONNECTIVITY_BLOCK_LENGTH = 14


class ChemblSourceError(ValueError):
    """Raised when the ChEMBL SQLite release is missing or not usable."""


@dataclass(frozen=True)
class StructureMatch:
    """One ChEMBL compound matched by one of a test structure's keys.

    Attributes
    ----------
    representation : Representation
        Which representation matched (``inchikey``, ``parent_inchikey`` or
        ``inchikey_connectivity``).
    value : str
        The key's value, as computed from the test record.
    compound : ChemblCompound
        The matched ChEMBL compound.
    """

    representation: Representation
    value: str
    compound: ChemblCompound


def connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a ChEMBL SQLite release read-only and check it looks like ChEMBL.

    Parameters
    ----------
    database_path : str | Path
        Path to the ChEMBL SQLite file.

    Returns
    -------
    sqlite3.Connection
        A read-only connection with ``row_factory`` set to
        ``sqlite3.Row``.

    Raises
    ------
    ChemblSourceError
        If the file does not exist, is not a readable SQLite database, or
        is missing any of ``REQUIRED_TABLES``.
    """
    path = Path(database_path)
    if not path.is_file():
        raise ChemblSourceError(f"ChEMBL SQLite release not found: {path}")
    # The path goes into a URI, where '#', '?' and '%' are syntax: an
    # unescaped '#' truncates the path and drops ?mode=ro with it, so SQLite
    # would open a different file, writable. quote() leaves '/' alone.
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ChemblSourceError(f"Could not open {path} as a SQLite database: {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        present = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise ChemblSourceError(f"{path} is not a readable SQLite database: {exc}") from exc
    missing = [table for table in REQUIRED_TABLES if table not in present]
    if missing:
        conn.close()
        raise ChemblSourceError(
            f"{path} does not look like a ChEMBL release: missing table(s) {missing}."
        )
    return conn


def read_source(conn: sqlite3.Connection, database_path: str | Path) -> ChemblSource:
    """Read the release's identity from its own ``version`` table.

    Recording this is what makes a report reproducible: another person with
    the same release number gets the same answer.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open connection, as returned by ``connect``.
    database_path : str | Path
        The path the release was opened from, echoed into the report.

    Returns
    -------
    ChemblSource
        The release name, date and comments where the ``version`` table
        provides them, with ``None`` for anything it does not.
    """
    source: ChemblSource = {
        "database_path": str(database_path),
        "release": None,
        "release_date": None,
        "comments": None,
    }
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(version)")}
    if not columns:
        logger.warning(
            "ChEMBL release at %s has no 'version' table, so the report cannot record "
            "which release it used.",
            database_path,
        )
        return source
    wanted = [col for col in ("name", "creation_date", "comments") if col in columns]
    if not wanted:
        logger.warning(
            "ChEMBL release at %s has a 'version' table with none of the expected "
            "columns, so the report cannot record which release it used.",
            database_path,
        )
        return source
    # A real release's version table also lists the versions of the
    # ontologies and tools it was built with (ChEMBL 37 has 11 rows, with the
    # release third), and one of those, "ChEMBL_Structure_Pipeline 1.2.0",
    # also starts with "ChEMBL". So the release row is matched by its exact
    # name pattern, then by its comment, and only then by position.
    rows = conn.execute(f"SELECT {', '.join(wanted)} FROM version").fetchall()
    row = next(
        (r for r in rows if "name" in wanted and _RELEASE_NAME.fullmatch(str(r["name"] or ""))),
        None,
    )
    if row is None and "comments" in wanted:
        row = next(
            (r for r in rows if _RELEASE_COMMENT.match(str(r["comments"] or ""))),
            None,
        )
    if row is None and rows:
        row = rows[0]
    if row is None:
        return source
    if "name" in wanted and row["name"] is not None:
        source["release"] = str(row["name"])
    if "creation_date" in wanted and row["creation_date"] is not None:
        source["release_date"] = str(row["creation_date"])
    if "comments" in wanted and row["comments"] is not None:
        source["comments"] = str(row["comments"])
    return source


def _load_molregnos(conn: sqlite3.Connection, molregnos: Sequence[int]) -> None:
    """Load matched compound ids into an INTEGER TEMP table for joining.

    Kept integer-typed, and joined without a cast, so SQLite can use
    ChEMBL's index on ``activities.molregno``. Keying these as text instead
    turns the activity lookup into a full scan of a very large table.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open connection.
    molregnos : Sequence[int]
        The compound ids to load.

    Returns
    -------
    None
    """
    conn.execute("DROP TABLE IF EXISTS temp.cdc_molregnos")
    conn.execute("CREATE TEMP TABLE cdc_molregnos (key INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT OR IGNORE INTO temp.cdc_molregnos (key) VALUES (?)",
        [(molregno,) for molregno in molregnos],
    )


def _load_keys(conn: sqlite3.Connection, table: str, keys: Iterable[str]) -> int:
    """Load lookup keys into an indexed TEMP table for joining.

    ``table`` is interpolated into the statements because SQLite takes no
    parameter in a table name. Every caller passes one of this module's own
    constants, never anything derived from user input.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open connection.
    table : str
        Name of the TEMP table to create.
    keys : Iterable[str]
        The key values to load; duplicates are collapsed.

    Returns
    -------
    int
        How many distinct keys were loaded.
    """
    conn.execute(f"DROP TABLE IF EXISTS temp.{table}")
    conn.execute(f"CREATE TEMP TABLE {table} (key TEXT PRIMARY KEY)")
    distinct = sorted({key for key in keys if key})
    conn.executemany(
        f"INSERT OR IGNORE INTO temp.{table} (key) VALUES (?)",
        [(key,) for key in distinct],
    )
    return len(distinct)


def _compound(row: sqlite3.Row) -> ChemblCompound:
    """Build a report compound from a lookup row.

    Parameters
    ----------
    row : sqlite3.Row
        A row with ``molregno``, ``parent_molregno``, ``chembl_id`` and
        ``standard_inchi_key``.

    Returns
    -------
    ChemblCompound
        The compound, as it appears in the report.
    """
    return {
        "molregno": int(row["molregno"]),
        "parent_molregno": None if row["parent_molregno"] is None else int(row["parent_molregno"]),
        "chembl_id": None if row["chembl_id"] is None else str(row["chembl_id"]),
        "standard_inchi_key": (
            None if row["standard_inchi_key"] is None else str(row["standard_inchi_key"])
        ),
    }


# An exact InChIKey match: the test structure is in ChEMBL as drawn,
# stereochemistry and salt form included.
_EXACT_SQL = """
SELECT k.key AS key,
       cs.molregno AS molregno,
       COALESCE(mh.parent_molregno, cs.molregno) AS parent_molregno,
       md.chembl_id AS chembl_id,
       cs.standard_inchi_key AS standard_inchi_key
FROM temp.cdc_keys_exact k
JOIN compound_structures cs ON cs.standard_inchi_key = k.key
LEFT JOIN molecule_hierarchy mh ON mh.molregno = cs.molregno
LEFT JOIN molecule_dictionary md ON md.molregno = cs.molregno
"""

# A salt-form match, which is the leakage mode that salts and solvates
# actually cause. The key is the InChIKey of the checker's salt-stripped
# parent, so it is matched against ChEMBL's *parent* compound, and
# molecule_hierarchy then fans out to every salt form sharing that parent:
# a test set holding the free base matches a ChEMBL hydrochloride, and
# vice versa.
#
# The compound the key names is selected on its own, in the first half,
# rather than through its own molecule_hierarchy row, because a parent need
# not have one. In ChEMBL 37, 93,019 of the 2,798,678 distinct
# parent_molregno values have no row at all: typically a free base ChEMBL
# registered only as the parent of the salt it was tested as. Reached
# through the hierarchy alone, such a parent drops out of its own match, and
# a parent with no row and no salt forms would not match at all. A compound
# with no row is its own parent, as in ``_EXACT_SQL``. The second half adds
# every other compound filed under it, so the parent is never counted twice.
_PARENT_SQL = """
SELECT k.key AS key,
       parent.molregno AS molregno,
       COALESCE(own.parent_molregno, parent.molregno) AS parent_molregno,
       md.chembl_id AS chembl_id,
       parent.standard_inchi_key AS standard_inchi_key
FROM temp.cdc_keys_parent k
JOIN compound_structures parent ON parent.standard_inchi_key = k.key
LEFT JOIN molecule_hierarchy own ON own.molregno = parent.molregno
LEFT JOIN molecule_dictionary md ON md.molregno = parent.molregno
UNION ALL
SELECT k.key AS key,
       cs.molregno AS molregno,
       parent.molregno AS parent_molregno,
       md.chembl_id AS chembl_id,
       cs.standard_inchi_key AS standard_inchi_key
FROM temp.cdc_keys_parent k
JOIN compound_structures parent ON parent.standard_inchi_key = k.key
JOIN molecule_hierarchy mh ON mh.parent_molregno = parent.molregno
JOIN compound_structures cs ON cs.molregno = mh.molregno
LEFT JOIN molecule_dictionary md ON md.molregno = cs.molregno
WHERE mh.molregno <> parent.molregno
"""

# A connectivity-only match, which catches undefined or differing
# stereochemistry.
#
# Written as a range rather than as a prefix test, so ChEMBL's index on
# ``standard_inchi_key`` ("This is indexed", per the schema documentation)
# serves it. A standard InChIKey is 14 connectivity characters, '-', 10
# stereo/protonation characters, '-' and a final character, so every key
# sharing a connectivity block lies between ``<block>-`` and ``<block>.`` in
# byte order: '.' is the next code point after '-', so nothing else can fall
# in between. ``CROSS JOIN`` then forces the TEMP table to be the outer loop
# (as in ``_ACTIVITIES_SQL``) and each key becomes one index range scan.
# Measured over 200k synthetic structures, that runs ~40x faster than
# probing the TEMP table from ``substr(standard_inchi_key, 1, 14)``, which
# has to scan ``compound_structures`` whole.
#
# The two forms agree on every standard InChIKey. They differ only where
# ``standard_inchi_key`` is not one: a bare 14-character block, a first
# block of some other length, or a separator that is not '-' is matched by
# the ``substr`` form and not by this one. Those are not InChIKeys, so
# excluding them is right rather than merely cheaper.
#
# This does assume the index is present, as every official release has it.
# Without it the forced join order costs one table scan per key instead of
# one in total, so a hand-trimmed dump should be given the index back.
_CONNECTIVITY_SQL = """
SELECT k.key AS key,
       cs.molregno AS molregno,
       COALESCE(mh.parent_molregno, cs.molregno) AS parent_molregno,
       md.chembl_id AS chembl_id,
       cs.standard_inchi_key AS standard_inchi_key
FROM temp.cdc_keys_connectivity k
CROSS JOIN compound_structures cs
  ON cs.standard_inchi_key >= k.key || '-'
 AND cs.standard_inchi_key < k.key || '.'
LEFT JOIN molecule_hierarchy mh ON mh.molregno = cs.molregno
LEFT JOIN molecule_dictionary md ON md.molregno = cs.molregno
"""

_LOOKUPS: dict[Representation, tuple[str, str]] = {
    "inchikey": ("cdc_keys_exact", _EXACT_SQL),
    "parent_inchikey": ("cdc_keys_parent", _PARENT_SQL),
    "inchikey_connectivity": ("cdc_keys_connectivity", _CONNECTIVITY_SQL),
}


def find_structures(
    conn: sqlite3.Connection, keys: Mapping[Representation, Sequence[str]]
) -> list[StructureMatch]:
    """Look up test structures in ChEMBL by each representation's keys.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open connection, as returned by ``connect``.
    keys : Mapping[Representation, Sequence[str]]
        The lookup keys per representation. Only the representations in
        ``_LOOKUPS`` are searched; a representation with no keys is
        skipped.

    Returns
    -------
    list[StructureMatch]
        One match per (representation, key, ChEMBL compound) triple.

    Raises
    ------
    ChemblSourceError
        If a lookup query fails against this release's schema.
    """
    matches: list[StructureMatch] = []
    for representation, (table, sql) in _LOOKUPS.items():
        values = keys.get(representation) or []
        if not values:
            continue
        loaded = _load_keys(conn, table, values)
        logger.debug("Looking up %d %s key(s) in ChEMBL.", loaded, representation)
        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.DatabaseError as exc:
            raise ChemblSourceError(
                f"ChEMBL {representation} lookup failed against this release: {exc}"
            ) from exc
        matches.extend(
            StructureMatch(
                representation=representation, value=str(row["key"]), compound=_compound(row)
            )
            for row in rows
        )
    return matches


def resolve_target_names(
    conn: sqlite3.Connection,
    names: Sequence[str],
    target_types: Sequence[str] = DEFAULT_TARGET_TYPES,
) -> dict[str, list[str]]:
    """Resolve target names or gene symbols to ChEMBL target IDs.

    Matching is case-insensitive against ``target_dictionary.pref_name``
    and, where the release has it, ``component_synonyms.component_synonym``,
    which is where gene symbols such as ``CYP3A4`` live.

    Only targets of the given types are kept. In ChEMBL 37, ``CYP3A4``
    resolves to the single protein CHEMBL340 and also to three protein
    families, the broadest being every cytochrome P450, so the default is
    single proteins only.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open connection, as returned by ``connect``.
    names : Sequence[str]
        The target names or gene symbols from the configuration.
    target_types : Sequence[str]
        ``target_dictionary.target_type`` values to keep, compared
        case-insensitively. Defaults to ``DEFAULT_TARGET_TYPES``.

    Returns
    -------
    dict[str, list[str]]
        Each requested name mapped to the sorted ChEMBL target IDs it
        resolved to, empty where nothing matched.
    """
    resolved: dict[str, list[str]] = {name: [] for name in names}
    if not names:
        return resolved
    tables = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    has_synonyms = {"component_synonyms", "target_components"} <= tables
    types = sorted({t.upper() for t in target_types})
    placeholders = ", ".join("?" for _ in types)
    for name in names:
        found: set[str] = {
            str(row["chembl_id"])
            for row in conn.execute(
                "SELECT chembl_id FROM target_dictionary WHERE LOWER(pref_name) = LOWER(?) "
                f"AND UPPER(target_type) IN ({placeholders})",
                (name, *types),
            )
        }
        if has_synonyms:
            found |= {
                str(row["chembl_id"])
                for row in conn.execute(
                    f"""
                    SELECT DISTINCT td.chembl_id AS chembl_id
                    FROM component_synonyms csyn
                    JOIN target_components tc ON tc.component_id = csyn.component_id
                    JOIN target_dictionary td ON td.tid = tc.tid
                    WHERE LOWER(csyn.component_synonym) = LOWER(?)
                      AND UPPER(td.target_type) IN ({placeholders})
                    """,
                    (name, *types),
                )
            }
        resolved[name] = sorted(found)
        if not found:
            logger.warning("Target name %r did not resolve to any ChEMBL target.", name)
    return resolved


def _document_reference(row: sqlite3.Row, columns: frozenset[str]) -> str | None:
    """Format a citation for an activity's source document.

    Parameters
    ----------
    row : sqlite3.Row
        An activity row, which may carry document columns.
    columns : frozenset[str]
        The column names the row actually has.

    Returns
    -------
    str | None
        A ``"Journal Year, doi"``-style citation, or ``None`` if the row
        carries no document detail.
    """
    parts: list[str] = []
    if "journal" in columns and row["journal"]:
        parts.append(str(row["journal"]))
    if "year" in columns and row["year"]:
        parts.append(str(row["year"]))
    reference = " ".join(parts)
    if "doi" in columns and row["doi"]:
        reference = f"{reference}, doi:{row['doi']}" if reference else f"doi:{row['doi']}"
    return reference or None


# ``CROSS JOIN`` is SQLite's join-order hint, not a cartesian product: it
# forces the small TEMP table to be the outer loop so ``activities`` is probed
# through its ``molregno`` index. Written the other way round, SQLite scans the
# whole of ``activities`` (tens of millions of rows in a real release) and
# probes the TEMP table for each one, which measures over a hundred times
# slower.
_ACTIVITIES_SQL = """
SELECT a.molregno AS molregno,
       a.standard_type AS standard_type,
       a.standard_relation AS standard_relation,
       a.standard_value AS standard_value,
       a.standard_units AS standard_units,
       a.pchembl_value AS pchembl_value,
       -- A result can be text rather than a number ("Not active", "Toxic"),
       -- and activity_comment says whether the endpoint was measured at all.
       -- Both are needed to tell a qualitative answer from a non-answer.
       a.standard_text_value AS standard_text_value,
       a.activity_comment AS activity_comment,
       -- ChEMBL's own curation flags on the measurement. A validity comment
       -- ("Potential transcription error") says the number may be wrong, not
       -- that the endpoint was not measured; potential_duplicate says the row
       -- is likely a repeat citation of an earlier paper's value.
       a.data_validity_comment AS data_validity_comment,
       a.potential_duplicate AS potential_duplicate,
       ass.chembl_id AS assay_chembl_id,
       ass.description AS assay_description,
       td.chembl_id AS target_chembl_id,
       td.pref_name AS target_name,
       td.organism AS organism,
       -- An activity has its own doc_id, which ChEMBL permits to differ from
       -- its assay's, so the measurement is cited to the document it actually
       -- came from and only falls back to the assay's.
       d.chembl_id AS document_chembl_id,
       d.journal AS journal,
       d.year AS year,
       d.doi AS doi
FROM temp.cdc_molregnos m
CROSS JOIN activities a ON a.molregno = m.key
JOIN assays ass ON ass.assay_id = a.assay_id
LEFT JOIN target_dictionary td ON td.tid = ass.tid
LEFT JOIN docs d ON d.doc_id = COALESCE(a.doc_id, ass.doc_id)
"""

# The same query for a release with no docs table, so the search still runs
# (without per-document grouping) on a trimmed dump.
_ACTIVITIES_SQL_NO_DOCS = """
SELECT a.molregno AS molregno,
       a.standard_type AS standard_type,
       a.standard_relation AS standard_relation,
       a.standard_value AS standard_value,
       a.standard_units AS standard_units,
       a.pchembl_value AS pchembl_value,
       a.standard_text_value AS standard_text_value,
       a.activity_comment AS activity_comment,
       a.data_validity_comment AS data_validity_comment,
       a.potential_duplicate AS potential_duplicate,
       ass.chembl_id AS assay_chembl_id,
       ass.description AS assay_description,
       td.chembl_id AS target_chembl_id,
       td.pref_name AS target_name,
       td.organism AS organism
FROM temp.cdc_molregnos m
CROSS JOIN activities a ON a.molregno = m.key
JOIN assays ass ON ass.assay_id = a.assay_id
LEFT JOIN target_dictionary td ON td.tid = ass.tid
"""


def _as_float(value: object) -> float | None:
    """Coerce a SQLite numeric column to a float, tolerating junk.

    Parameters
    ----------
    value : object
        The raw column value.

    Returns
    -------
    float | None
        The value as a float, or ``None`` if it is null or not numeric.
    """
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_flag(value: object) -> bool:
    """Coerce a ChEMBL 0/1 flag column to a bool.

    The numeric value decides, not Python truthiness: ``potential_duplicate``
    is declared ``NUMBER(1)`` and arrives as an integer from an official
    SQLite dump, but a third-party conversion can store it as the text
    ``"0"``, which ``bool()`` would read as set.

    Parameters
    ----------
    value : object
        The raw column value.

    Returns
    -------
    bool
        ``True`` if the column holds a non-zero number, or a non-empty
        value that is not a number at all.
    """
    if value is None:
        return False
    try:
        return float(value) != 0.0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return bool(str(value).strip())


def find_activities(
    conn: sqlite3.Connection, molregnos: Iterable[int]
) -> dict[int, list[ChemblActivity]]:
    """Fetch every activity record for the matched ChEMBL compounds.

    Fetching all of them, rather than only those matching the endpoint,
    keeps the endpoint decision in one place (``public_search``) and lets
    the report show what else the compound was measured for.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open connection, as returned by ``connect``.
    molregnos : Iterable[int]
        The ``molregno`` of every matched compound.

    Returns
    -------
    dict[int, list[ChemblActivity]]
        Activities grouped by ``molregno``, empty for a compound with
        none.

    Raises
    ------
    ChemblSourceError
        If the activity query fails against this release's schema.
    """
    wanted = sorted({int(molregno) for molregno in molregnos})
    if not wanted:
        return {}
    _load_molregnos(conn, wanted)
    tables = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    sql = _ACTIVITIES_SQL if "docs" in tables else _ACTIVITIES_SQL_NO_DOCS
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ChemblSourceError(
            f"ChEMBL activity lookup failed against this release: {exc}"
        ) from exc

    grouped: dict[int, list[ChemblActivity]] = {molregno: [] for molregno in wanted}
    for row in rows:
        columns = frozenset(row.keys())
        activity: ChemblActivity = {
            "assay_chembl_id": (
                None if row["assay_chembl_id"] is None else str(row["assay_chembl_id"])
            ),
            "assay_description": (
                None if row["assay_description"] is None else str(row["assay_description"])
            ),
            "target_chembl_id": (
                None if row["target_chembl_id"] is None else str(row["target_chembl_id"])
            ),
            "target_name": None if row["target_name"] is None else str(row["target_name"]),
            "organism": None if row["organism"] is None else str(row["organism"]),
            "standard_type": None if row["standard_type"] is None else str(row["standard_type"]),
            "standard_relation": (
                None if row["standard_relation"] is None else str(row["standard_relation"])
            ),
            "standard_value": _as_float(row["standard_value"]),
            "standard_units": (
                None if row["standard_units"] is None else str(row["standard_units"])
            ),
            "pchembl_value": _as_float(row["pchembl_value"]),
            "standard_text_value": (
                None if row["standard_text_value"] is None else str(row["standard_text_value"])
            ),
            "activity_comment": (
                None if row["activity_comment"] is None else str(row["activity_comment"])
            ),
            "data_validity_comment": (
                None if row["data_validity_comment"] is None else str(row["data_validity_comment"])
            ),
            "potential_duplicate": _as_flag(row["potential_duplicate"]),
            "document_chembl_id": (
                str(row["document_chembl_id"])
                if "document_chembl_id" in columns and row["document_chembl_id"] is not None
                else None
            ),
            "document_reference": _document_reference(row, columns),
        }
        grouped[int(row["molregno"])].append(activity)
    return grouped
