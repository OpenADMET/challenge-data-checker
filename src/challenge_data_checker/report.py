"""Assembling the audit report (JSON or text) and the stdout summary dashboard."""

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rich import box
from rich.console import Console, Group, RenderableType
from rich.rule import Rule
from rich.table import Table

from challenge_data_checker.checks import (
    identifier_leakage,
    identifier_namespace_issues,
    internal_duplicates,
    tanimoto_leakage,
    train_test_leakage,
    unparseable_entries,
)
from challenge_data_checker.config import Config
from challenge_data_checker.io_utils import describe_source
from challenge_data_checker.models import MoleculeRecord
from challenge_data_checker.report_types import (
    EXACT_REPRESENTATIONS,
    QUALITY_FLAG_CATEGORIES,
    REPRESENTATIONS,
    SALT_FORM_INSENSITIVE_REPRESENTATIONS,
    STEREO_INSENSITIVE_REPRESENTATIONS,
    InternalDuplicates,
    LeakageFinding,
    Location,
    NamespaceIssues,
    PerPool,
    QualityFlags,
    Report,
    Summary,
    TanimotoFinding,
    UnparseableEntry,
)
from challenge_data_checker.severity import (
    QUALITY_FLAG_FINDING_TYPES,
    Severity,
    TierContext,
    collect_findings,
    duplicate_finding_type,
    leakage_finding_type,
    severity_of,
    summarise_tiers,
)


def quality_flag_entries(records: list[MoleculeRecord]) -> QualityFlags:
    """Collect quality-filter findings for one pool.

    Parameters
    ----------
    records : list[MoleculeRecord]
        Processed records from a single pool.

    Returns
    -------
    QualityFlags
        A mapping of each quality-flag category (mixtures,
        salts/metal complexes, suspicious fragments, SMILES only parsed
        after stripping whitespace, RDKit parse warnings) to the matching
        record locations.
    """
    parsed = [r for r in records if r.is_parsed]
    return {
        "salt_or_solvate": [r.location() for r in parsed if r.is_salt_or_solvate],
        "contains_metal": [r.location() for r in parsed if r.contains_metal],
        "multi_component": [r.location() for r in parsed if r.is_multi_component],
        "homodimer_without_counterion": [
            r.location() for r in parsed if r.is_homodimer_without_counterion
        ],
        "unclassified_mixture": [r.location() for r in parsed if r.is_unclassified_mixture],
        "net_charge": [
            {**r.location(), "charge": r.net_charge}
            for r in parsed
            if r.net_charge is not None and r.net_charge != 0
        ],
        "mixtures": [r.location() for r in parsed if r.is_mixture],
        "salts_or_metal_complexes": [r.location() for r in parsed if r.is_salt_or_metal],
        "suspicious_fragments": [
            r.location() for r in records if r.is_parsed and r.is_suspicious_fragment
        ],
        "smiles_whitespace": [
            r.location() for r in records if r.is_parsed and r.smiles_whitespace_recovered
        ],
        "rdkit_warnings": [
            {**r.location(), "warning": r.rdkit_warning}
            for r in records
            if r.is_parsed and r.rdkit_warning is not None
        ],
    }


def _leaked_rows(
    exact_leakage: dict[str, list[LeakageFinding]],
    side: Literal["train", "test"],
    representations: Iterable[str],
) -> set[tuple[str, str, int]]:
    """Collect the distinct rows involved in leakage findings on one side.

    Parameters
    ----------
    exact_leakage : dict[str, list[LeakageFinding]]
        The result of ``checks.train_test_leakage``.
    side : Literal["train", "test"]
        Which side to collect, ``"train"`` or ``"test"``.
    representations : Iterable[str]
        Which representations' findings to include.

    Returns
    -------
    set[tuple[str, str, int]]
        The distinct (pool, source_file, row_index) rows appearing as a
        ``{side}_occurrences`` entry under any of ``representations``.
    """
    seen = set()
    for name in representations:
        for entry in exact_leakage.get(name, []):
            occurrences = entry["train_occurrences" if side == "train" else "test_occurrences"]
            for occ in occurrences:
                seen.add((occ["pool"], occ["source_file"], occ["row_index"]))
    return seen


def _leaked_counts(
    exact_leakage: dict[str, list[LeakageFinding]], side: Literal["train", "test"]
) -> tuple[int, int, int]:
    """Count leaked rows by the most exact kind of match each row has.

    Parameters
    ----------
    exact_leakage : dict[str, list[LeakageFinding]]
        The result of ``checks.train_test_leakage``.
    side : Literal["train", "test"]
        Which side to count, ``"train"`` or ``"test"``.

    Returns
    -------
    int
        Distinct rows in an exact (raw SMILES, canonical SMILES or
        InChIKey) match.
    int
        Distinct rows in a parent-compound match but no exact match, e.g.
        a hydrochloride whose free base is in the other pool.
    int
        Distinct rows in a stereo-insensitive (connectivity) match but
        no exact or parent match, e.g. one enantiomer of a racemate in the
        other pool.
    """
    exact = _leaked_rows(exact_leakage, side, EXACT_REPRESENTATIONS)
    salt_form = _leaked_rows(exact_leakage, side, SALT_FORM_INSENSITIVE_REPRESENTATIONS)
    stereo = _leaked_rows(exact_leakage, side, STEREO_INSENSITIVE_REPRESENTATIONS)
    return len(exact), len(salt_form - exact), len(stereo - exact - salt_form)


def _mark_resolvable_rows(
    unparseable: list[UnparseableEntry], namespace_issues: dict[str, NamespaceIssues]
) -> None:
    """Record, on each empty-SMILES row, an identifier column that can fill it in.

    A row is resolvable through a column when its identifier there has
    exactly one parsed structure elsewhere. The first such column wins.

    Parameters
    ----------
    unparseable : list[UnparseableEntry]
        The unparseable rows, updated in place.
    namespace_issues : dict[str, NamespaceIssues]
        The result of ``checks.identifier_namespace_issues``.

    Returns
    -------
    None
    """
    resolvable: dict[tuple[str, str, int], str] = {}
    for col, issues in namespace_issues.items():
        for finding in issues["missing_structure_resolvable_by_id"]:
            if finding["ambiguous"]:
                continue
            for loc in finding["missing"]:
                resolvable.setdefault((loc["pool"], loc["source_file"], loc["row_index"]), col)
    for entry in unparseable:
        key = (entry["pool"], entry["source_file"], entry["row_index"])
        entry["resolvable_via_identifier"] = resolvable.get(key)


def build_report(
    config: Config,
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    identifier_columns: list[str],
    resolved_smiles_columns: PerPool[dict[str, str]] | None = None,
) -> Report:
    """Run all checks and assemble the full JSON-serialisable audit report.

    Parameters
    ----------
    config : Config
        The validated configuration used for this run.
    train_records : list[MoleculeRecord]
        Processed records from the training pool.
    test_records : list[MoleculeRecord]
        Processed records from the test pool.
    identifier_columns : list[str]
        Identifier columns present in at least one pool.
    resolved_smiles_columns : PerPool[dict[str, str]] | None
        The SMILES column actually resolved for each source, as
        ``{"train": {source_label: column, ...}, "test": {...}}`` (see
        ``io_utils.load_pool``). Recorded in the report so it's
        self-documenting even when the resolved column silently differs
        from ``config.columns.smiles_column``. Defaults to empty mappings.

    Returns
    -------
    Report
        The full report dict, containing the echoed config, a summary of
        counts, and the detailed findings from every check.
    """
    resolved_smiles_columns = resolved_smiles_columns or {"train": {}, "test": {}}
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    all_records = train_records + test_records

    unparseable = unparseable_entries(all_records)
    strip_salts = config.settings.strip_salts
    exact_leakage = train_test_leakage(train_records, test_records, strip_salts)
    id_leakage = identifier_leakage(train_records, test_records, identifier_columns)
    tanimoto = tanimoto_leakage(
        train_records, test_records, config.settings.max_train_test_similarity, strip_salts
    )
    train_dupes = internal_duplicates(train_records, identifier_columns, strip_salts)
    test_dupes = internal_duplicates(test_records, identifier_columns, strip_salts)
    namespace_issues = identifier_namespace_issues(all_records, identifier_columns)
    _mark_resolvable_rows(unparseable, namespace_issues)
    quality_train = quality_flag_entries(train_records)
    quality_test = quality_flag_entries(test_records)

    exact_train, salt_form_train, stereo_train = _leaked_counts(exact_leakage, "train")
    exact_test, salt_form_test, stereo_test = _leaked_counts(exact_leakage, "test")
    status, blocking, flagged = summarise_tiers(
        collect_findings(
            unparseable,
            exact_leakage,
            id_leakage,
            tanimoto,
            {"train": train_dupes, "test": test_dupes},
            namespace_issues,
            {"train": quality_train, "test": quality_test},
        ),
        TierContext(config.settings.max_train_test_similarity),
    )
    summary: Summary = {
        "status": status,
        "blocking": blocking,
        "flagged": flagged,
        "total_train_rows": len(train_records),
        "total_test_rows": len(test_records),
        "total_unparseable": len(unparseable),
        "total_unparseable_train": sum(1 for r in train_records if not r.is_parsed),
        "total_unparseable_test": sum(1 for r in test_records if not r.is_parsed),
        "total_unparseable_resolvable_via_identifier": sum(
            1 for entry in unparseable if entry["resolvable_via_identifier"] is not None
        ),
        "total_unparseable_by_category": dict(
            sorted(Counter(entry["category"] for entry in unparseable).items())
        ),
        "total_exact_leaked_train_molecules": exact_train,
        "total_exact_leaked_test_molecules": exact_test,
        "total_salt_form_leaked_train_molecules": salt_form_train,
        "total_salt_form_leaked_test_molecules": salt_form_test,
        "total_stereo_leaked_train_molecules": stereo_train,
        "total_stereo_leaked_test_molecules": stereo_test,
        "total_identifier_leakage_events": sum(len(v) for v in id_leakage.values()),
        "total_tanimoto_leakage_pairs": len(tanimoto),
        "total_internal_duplicates_train": {
            name: len(train_dupes[name]) for name in REPRESENTATIONS
        },
        "total_internal_duplicates_test": {name: len(test_dupes[name]) for name in REPRESENTATIONS},
        "total_internal_duplicates_train_identifiers": {
            k: len(v) for k, v in train_dupes["identifiers"].items()
        },
        "total_internal_duplicates_test_identifiers": {
            k: len(v) for k, v in test_dupes["identifiers"].items()
        },
        "total_identifier_namespace_issues": {
            col: {
                "one_id_multiple_structures": len(issues["one_id_multiple_structures"]),
                "one_structure_multiple_ids": len(issues["one_structure_multiple_ids"]),
                "identifier_whitespace": len(issues["identifier_whitespace"]),
                "near_duplicates": len(issues["near_duplicates"]),
                "id_with_unparseable_and_valid_structures": len(
                    issues["id_with_unparseable_and_valid_structures"]
                ),
                "missing_structure_resolvable_by_id": len(
                    issues["missing_structure_resolvable_by_id"]
                ),
            }
            for col, issues in namespace_issues.items()
        },
        "total_mixtures": {
            "train": len(quality_train["mixtures"]),
            "test": len(quality_test["mixtures"]),
        },
        "total_salts_or_metal_complexes": {
            "train": len(quality_train["salts_or_metal_complexes"]),
            "test": len(quality_test["salts_or_metal_complexes"]),
        },
        "total_suspicious_fragments": {
            "train": len(quality_train["suspicious_fragments"]),
            "test": len(quality_test["suspicious_fragments"]),
        },
        "total_quality_flags": {
            category: {
                "train": len(quality_train[category]),
                "test": len(quality_test[category]),
            }
            for category in QUALITY_FLAG_CATEGORIES
        },
    }

    return {
        "generated_at": generated_at,
        "config": {
            "train_files": [describe_source(s, i) for i, s in enumerate(config.paths.train_files)],
            "test_files": [describe_source(s, i) for i, s in enumerate(config.paths.test_files)],
            "smiles_column": config.columns.smiles_column,
            "resolved_smiles_columns": resolved_smiles_columns,
            "identifier_columns": identifier_columns,
            "tautomer_standardisation": config.settings.tautomer_standardisation,
            "strip_salts": config.settings.strip_salts,
            "identifier_case_insensitive": config.settings.identifier_case_insensitive,
            "multi_component_min_heavy_atoms": config.settings.multi_component_min_heavy_atoms,
            "max_train_test_similarity": config.settings.max_train_test_similarity,
            "fp_radius": config.settings.fp_radius,
            "fp_n_bits": config.settings.fp_n_bits,
            "report_format": config.settings.report_format,
        },
        "summary": summary,
        "unparseable_smiles": unparseable,
        "train_test_leakage": {
            "exact": exact_leakage,
            "identifiers": id_leakage,
            "tanimoto_similarity": tanimoto,
        },
        "internal_duplicates": {"train": train_dupes, "test": test_dupes},
        "identifier_namespace_issues": namespace_issues,
        "quality_flags": {"train": quality_train, "test": quality_test},
    }


def save_report(report: Report, output_path: str | Path, report_format: str = "json") -> None:
    """Write the report to disk as either indented JSON or human-readable text.

    Creates any missing parent directories of ``output_path``.

    Parameters
    ----------
    report : Report
        The report dict, as returned by ``build_report``.
    output_path : str | Path
        Path to write the report to.
    report_format : str
        Either ``"json"`` (machine-readable, full fidelity) or ``"txt"``
        (human-readable prose rendering of the same findings).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``report_format`` is not ``"json"`` or ``"txt"``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_format == "json":
        with output_path.open("w") as fh:
            json.dump(report, fh, indent=2)
    elif report_format == "txt":
        output_path.write_text(format_report_as_text(report))
    else:
        raise ValueError(f"Unsupported report_format '{report_format}'. Use 'json' or 'txt'.")


def _fmt_counts(counts: Mapping[str, object]) -> str:
    """Format a mapping of counts as a compact ``key=value, ...`` string.

    Parameters
    ----------
    counts : Mapping[str, object]
        The counts to format.

    Returns
    -------
    str
        A comma-separated ``key=value`` string.
    """
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def _tier_context(report: Report) -> TierContext:
    """Rebuild the settings that finding tiers depend on from a report.

    Parameters
    ----------
    report : Report
        The report dict, as returned by ``build_report``.

    Returns
    -------
    TierContext
        The tier context the report was built with.
    """
    return TierContext(report["config"]["max_train_test_similarity"])


_TIER_LABELS = {Severity.FAIL: "Blocking (FAIL)", Severity.WARN: "Flagged (WARN)"}


def _tier_breakdown_lines(s: Summary) -> list[str]:
    """Build the text lines summarising findings by severity tier.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.

    Returns
    -------
    list[str]
        The overall status, then one block per non-empty tier listing
        each finding type's finding and row counts.
    """
    lines = [f"Overall status        : {s['status']}"]
    for severity, tier in ((Severity.FAIL, s["blocking"]), (Severity.WARN, s["flagged"])):
        lines.append(
            f"{_TIER_LABELS[severity]:<22}: {tier['findings']} finding(s) in {tier['rows']} row(s)"
        )
        for finding_type, counts in tier["by_type"].items():
            lines.append(
                f"  {finding_type:<28}: {counts['findings']} finding(s), {counts['rows']} row(s)"
            )
    return lines


def _build_summary_lines(report: Report) -> list[str]:
    """Build the summary dashboard as a list of text lines.

    Parameters
    ----------
    report : Report
        The report dict, as returned by ``build_report``.

    Returns
    -------
    list[str]
        The dashboard lines, banner included but with no trailing
        "written to" line (callers add that themselves if relevant).
    """
    s = report["summary"]
    lines = [
        "=" * 60,
        "challenge-data-checker audit summary",
        "=" * 60,
        *_tier_breakdown_lines(s),
        "",
        f"Rows processed        : train={s['total_train_rows']}  test={s['total_test_rows']}",
        (
            f"Unparseable SMILES    : total={s['total_unparseable']}  "
            f"(train={s['total_unparseable_train']}, test={s['total_unparseable_test']})"
        ),
        *(
            [f"  by category        : {_fmt_counts(s['total_unparseable_by_category'])}"]
            if s["total_unparseable_by_category"]
            else []
        ),
        *(
            [
                f"  resolvable via id  : {s['total_unparseable_resolvable_via_identifier']} "
                "(empty SMILES whose identifier has one structure elsewhere)"
            ]
            if s["total_unparseable_resolvable_via_identifier"]
            else []
        ),
        "",
        "Train-test leakage",
        (
            f"  Train compounds in train/test overlap (raw/canonical/InChIKey): "
            f"{s['total_exact_leaked_train_molecules']}"
        ),
        (
            f"  Test compounds in train/test overlap (raw/canonical/InChIKey) : "
            f"{s['total_exact_leaked_test_molecules']}"
        ),
        (
            f"  Train compounds in salt-form-only overlap (same parent)       : "
            f"{s['total_salt_form_leaked_train_molecules']}"
        ),
        (
            f"  Test compounds in salt-form-only overlap (same parent)        : "
            f"{s['total_salt_form_leaked_test_molecules']}"
        ),
        (
            f"  Train compounds in stereo-only overlap (same connectivity)    : "
            f"{s['total_stereo_leaked_train_molecules']}"
        ),
        (
            f"  Test compounds in stereo-only overlap (same connectivity)     : "
            f"{s['total_stereo_leaked_test_molecules']}"
        ),
        (
            f"  Identifier overlap events                                     : "
            f"{s['total_identifier_leakage_events']}"
        ),
        (
            f"  Tanimoto similarity flags (>= threshold)                      : "
            f"{s['total_tanimoto_leakage_pairs']}"
        ),
        "",
        "Internal duplicates (within split)",
        f"  Train: {_fmt_counts(s['total_internal_duplicates_train'])}",
    ]
    if s["total_internal_duplicates_train_identifiers"]:
        lines.append(
            f"    identifiers: {_fmt_counts(s['total_internal_duplicates_train_identifiers'])}"
        )
    lines.append(f"  Test : {_fmt_counts(s['total_internal_duplicates_test'])}")
    if s["total_internal_duplicates_test_identifiers"]:
        lines.append(
            f"    identifiers: {_fmt_counts(s['total_internal_duplicates_test_identifiers'])}"
        )
    lines.append("")
    lines.append("Identifier namespace issues")
    for col, counts in s["total_identifier_namespace_issues"].items():
        lines.append(f"  {col}: {_fmt_counts(counts)}")
    if not s["total_identifier_namespace_issues"]:
        lines.append("  (no identifier columns configured/available)")
    lines.append("")
    lines.append("Quality filter flags")
    for category, label in _QUALITY_FLAG_LABELS.items():
        lines.append(f"  {label:<28}: {_fmt_counts(s['total_quality_flags'][category])}")
    lines.append("=" * 60)
    return lines


def _count_style(count: int, severity: Severity = Severity.FAIL) -> str:
    """Pick a rich style for a finding count: green if zero, else by tier.

    Parameters
    ----------
    count : int
        The finding count to style.
    severity : Severity
        The tier of the counted findings.

    Returns
    -------
    str
        ``"bold green"`` if ``count`` is zero, otherwise ``"bold red"``
        for blocking findings and ``"bold yellow"`` for flagged ones.
    """
    if count == 0:
        return "bold green"
    return "bold red" if severity is Severity.FAIL else "bold yellow"


def _status_cell(count: int, severity: Severity) -> str:
    """Render a count as a colored check-mark cell for a rich table.

    Parameters
    ----------
    count : int
        The finding count to render.
    severity : Severity
        The tier of the counted findings.

    Returns
    -------
    str
        A rich markup string: a green checkmark for zero, otherwise a red
        cross (blocking) or yellow warning sign (flagged) with the count.
    """
    if count == 0:
        return "[bold green]✓ 0[/]"
    if severity is Severity.FAIL:
        return f"[bold red]✗ {count}[/]"
    return f"[bold yellow]⚠ {count}[/]"


def _new_table(columns: list[str]) -> Table:
    """Create a bare rich table with the given column headers.

    The table has no title of its own; callers print a section heading
    separately so it isn't constrained (and wrapped) by the table's
    auto-sized column widths.

    Parameters
    ----------
    columns : list[str]
        Column header labels, in order.

    Returns
    -------
    Table
        An empty rich Table ready to have rows added.
    """
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold magenta",
        pad_edge=False,
    )
    for i, column in enumerate(columns):
        table.add_column(column, justify="right" if i > 0 else "left")
    return table


def _section_renderables(title: str, table: Table | None) -> list[RenderableType]:
    """Build the renderables for one dashboard section: heading, body, blank line.

    Kept separate from the table (or a placeholder) rather than printed
    directly, so the whole dashboard can be assembled into a single ``Group``
    and printed in one ``console.print`` call (see ``print_summary``) instead
    of one call per section — a Jupyter notebook renders every individual
    ``console.print`` as its own separate output block, which fragments the
    dashboard if it's built from many small print calls.

    Parameters
    ----------
    title : str
        The section heading text.
    table : Table | None
        The table to render, or ``None`` to render a "(none)" placeholder
        instead (e.g. no identifier columns configured).

    Returns
    -------
    list[RenderableType]
        The renderables for this section, in display order.
    """
    placeholder = "  (no identifier columns configured/available)"
    return [f"[bold cyan]{title}[/]", table if table is not None else placeholder, ""]


def _severity_table(s: Summary) -> Table | None:
    """Build the findings-by-severity table.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.

    Returns
    -------
    Table | None
        A rich table with one row per finding type that fired, blocking
        first, or ``None`` if there were no findings.
    """
    if not s["blocking"]["by_type"] and not s["flagged"]["by_type"]:
        return None
    table = _new_table(["Finding type", "Tier", "Findings", "Rows"])
    for severity, tier in ((Severity.FAIL, s["blocking"]), (Severity.WARN, s["flagged"])):
        style = _count_style(1, severity)
        for finding_type, counts in tier["by_type"].items():
            table.add_row(
                finding_type,
                f"[{style}]{severity}[/]",
                str(counts["findings"]),
                str(counts["rows"]),
            )
    return table


def _leakage_table(s: Summary, context: TierContext) -> Table:
    """Build the train-test leakage summary table.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.
    context : TierContext
        The settings finding tiers depend on.

    Returns
    -------
    Table
        A rich table with one row per leakage check.
    """
    fail, warn = Severity.FAIL, Severity.WARN
    tanimoto = severity_of("tanimoto_similarity", None, context)
    table = _new_table(["Check", "Result"])
    rows: list[tuple[str, int, Severity]] = [
        ("Train compounds in train/test overlap", s["total_exact_leaked_train_molecules"], fail),
        ("Test compounds in train/test overlap", s["total_exact_leaked_test_molecules"], fail),
        (
            "Train compounds in salt-form-only overlap",
            s["total_salt_form_leaked_train_molecules"],
            fail,
        ),
        (
            "Test compounds in salt-form-only overlap",
            s["total_salt_form_leaked_test_molecules"],
            fail,
        ),
        ("Train compounds in stereo-only overlap", s["total_stereo_leaked_train_molecules"], warn),
        ("Test compounds in stereo-only overlap", s["total_stereo_leaked_test_molecules"], warn),
        ("Identifier overlap events", s["total_identifier_leakage_events"], fail),
        ("Tanimoto similarity flags (>= threshold)", s["total_tanimoto_leakage_pairs"], tanimoto),
    ]
    for label, count, severity in rows:
        table.add_row(label, _status_cell(count, severity))
    return table


def _duplicates_table(s: Summary, context: TierContext) -> Table:
    """Build the internal (within-split) duplicates summary table.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.
    context : TierContext
        The settings finding tiers depend on.

    Returns
    -------
    Table
        A rich table with one row per representation/identifier column,
        showing train and test counts side by side.
    """
    table = _new_table(["Representation", "Train", "Test"])
    for representation in REPRESENTATIONS:
        finding_type = duplicate_finding_type(representation)
        table.add_row(
            representation,
            _status_cell(
                s["total_internal_duplicates_train"][representation],
                severity_of(finding_type, "train", context),
            ),
            _status_cell(
                s["total_internal_duplicates_test"][representation],
                severity_of(finding_type, "test", context),
            ),
        )
    for col in s["total_internal_duplicates_train_identifiers"]:
        table.add_row(
            f"identifier[{col}]",
            _status_cell(
                s["total_internal_duplicates_train_identifiers"][col],
                severity_of("duplicate_identifier", "train", context),
            ),
            _status_cell(
                s["total_internal_duplicates_test_identifiers"].get(col, 0),
                severity_of("duplicate_identifier", "test", context),
            ),
        )
    return table


type _NamespaceCountKey = Literal[
    "one_id_multiple_structures",
    "one_structure_multiple_ids",
    "id_with_unparseable_and_valid_structures",
    "missing_structure_resolvable_by_id",
    "near_duplicates",
    "identifier_whitespace",
]

# Rows of the dashboard's identifier table: label, NamespaceIssueCounts key,
# and finding type (for its tier).
_NAMESPACE_TABLE_ROWS: tuple[tuple[str, _NamespaceCountKey, str], ...] = (
    ("1 id -> many structures", "one_id_multiple_structures", "one_id_multiple_structures"),
    ("1 structure -> many ids", "one_structure_multiple_ids", "one_structure_multiple_ids"),
    (
        "id with broken + valid SMILES",
        "id_with_unparseable_and_valid_structures",
        "id_with_unparseable_and_valid_structures",
    ),
    (
        "empty SMILES resolvable via id",
        "missing_structure_resolvable_by_id",
        "missing_structure_resolvable_by_id",
    ),
    ("Near-duplicate ids", "near_duplicates", "identifier_near_duplicate"),
    ("Whitespace in ids", "identifier_whitespace", "identifier_whitespace"),
)


def _namespace_table(s: Summary, context: TierContext) -> Table | None:
    """Build the identifier namespace consistency summary table.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.
    context : TierContext
        The settings finding tiers depend on.

    Returns
    -------
    Table | None
        A rich table with one row per check and one column per identifier
        column, or ``None`` if no identifier columns were configured or
        available.
    """
    columns = s["total_identifier_namespace_issues"]
    if not columns:
        return None
    table = _new_table(["Check", *columns])
    for label, count_key, finding_type in _NAMESPACE_TABLE_ROWS:
        severity = severity_of(finding_type, None, context)
        table.add_row(
            label,
            *(_status_cell(counts[count_key], severity) for counts in columns.values()),
        )
    return table


# Dashboard labels for the tiered quality-flag categories, in display order.
# The deprecated totals ("mixtures", "salts_or_metal_complexes") stay in the
# JSON report but aren't shown.
_QUALITY_FLAG_LABELS: dict[str, str] = {
    "salt_or_solvate": "Salts/solvates",
    "contains_metal": "Contains metal",
    "multi_component": "Multi-component",
    "homodimer_without_counterion": "Homodimer, no counterion",
    "unclassified_mixture": "Other mixtures",
    "net_charge": "Net charge",
    "suspicious_fragments": "Suspicious fragments",
    "smiles_whitespace": "Whitespace in SMILES",
    "rdkit_warnings": "RDKit parse warnings",
}


def _quality_table(s: Summary, context: TierContext) -> Table:
    """Build the quality-filter flags summary table.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.
    context : TierContext
        The settings finding tiers depend on.

    Returns
    -------
    Table
        A rich table with one row per quality-filter category, showing
        train and test counts side by side.
    """
    table = _new_table(["Filter", "Train", "Test"])
    for category, label in _QUALITY_FLAG_LABELS.items():
        counts = s["total_quality_flags"][category]
        finding_type = QUALITY_FLAG_FINDING_TYPES[category]
        assert finding_type is not None  # deprecated categories aren't labelled
        table.add_row(
            label,
            _status_cell(counts["train"], severity_of(finding_type, "train", context)),
            _status_cell(counts["test"], severity_of(finding_type, "test", context)),
        )
    return table


def _status_rule(s: Summary) -> Rule:
    """Build the overall PASS / WARN / FAIL banner.

    Parameters
    ----------
    s : Summary
        The report's ``"summary"`` dict.

    Returns
    -------
    Rule
        A colored rich rule stating the overall status and the finding
        and row counts behind it.
    """
    blocking, flagged = s["blocking"], s["flagged"]
    if s["status"] == "PASS":
        return Rule("[bold green]✓ OVERALL PASS - no issues detected[/]", style="green")
    flagged_text = f"{flagged['findings']} flagged finding(s) in {flagged['rows']} row(s)"
    if s["status"] == "WARN":
        return Rule(f"[bold yellow]⚠ OVERALL WARN - {flagged_text} to review[/]", style="yellow")
    text = f"{blocking['findings']} blocking finding(s) in {blocking['rows']} row(s)"
    if flagged["findings"]:
        text += f"; {flagged_text}"
    return Rule(f"[bold red]✗ OVERALL FAIL - {text}[/]", style="red")


def print_summary(report: Report, report_output: str | Path | None) -> None:
    """Print the stdout summary dashboard for a completed audit report.

    Renders row counts, unparseable SMILES, a findings-by-severity table,
    and one colored table per check category (green checkmarks for clean
    results, red crosses for blocking findings, yellow warning signs for
    flagged ones), followed by an overall PASS/WARN/FAIL banner. The whole
    dashboard is assembled into a single rich ``Group`` and printed with one
    ``console.print`` call so it renders as one contiguous block rather than
    several (in a Jupyter notebook, each individual ``console.print`` call
    renders as its own separate output).

    Parameters
    ----------
    report : Report
        The report dict, as returned by ``build_report``.
    report_output : str | Path | None
        Path the full report was written to, shown in the final line of
        the dashboard, or ``None`` if it wasn't written to disk (that line
        is then omitted).

    Returns
    -------
    None
    """
    s = report["summary"]
    context = _tier_context(report)

    renderables: list[RenderableType] = [
        Rule("[bold cyan]challenge-data-checker audit summary[/]", style="cyan"),
        f"  Generated          : {report['generated_at']}",
        f"  Rows processed     : [bold]train[/]={s['total_train_rows']}  "
        f"[bold]test[/]={s['total_test_rows']}",
        f"  Unparseable SMILES : "
        f"[{_count_style(s['total_unparseable'])}]{s['total_unparseable']}[/] "
        f"(train={s['total_unparseable_train']}, test={s['total_unparseable_test']})"
        + (
            f" - {_fmt_counts(s['total_unparseable_by_category'])}"
            if s["total_unparseable_by_category"]
            else ""
        )
        + (
            f"; {s['total_unparseable_resolvable_via_identifier']} resolvable via identifier"
            if s["total_unparseable_resolvable_via_identifier"]
            else ""
        ),
        "  Legend             : [bold red]✗ blocking (FAIL)[/]  "
        "[bold yellow]⚠ flagged for review (WARN)[/]  [bold green]✓ clean[/]",
        "",
    ]
    severity_table = _severity_table(s)
    if severity_table is not None:
        renderables.extend(_section_renderables("Findings by severity", severity_table))
    renderables.extend(_section_renderables("Train-test leakage", _leakage_table(s, context)))
    renderables.extend(
        _section_renderables("Internal duplicates (within split)", _duplicates_table(s, context))
    )
    renderables.extend(
        _section_renderables("Identifier namespace issues", _namespace_table(s, context))
    )
    renderables.extend(_section_renderables("Quality filter flags", _quality_table(s, context)))
    renderables.append(_status_rule(s))
    if report_output is not None:
        renderables.append(f"Full detailed report written to: [bold]{report_output}[/]")

    Console(highlight=False).print(Group(*renderables))


def _tag(finding_type: str, pool: Literal["train", "test"] | None, context: TierContext) -> str:
    """Build the ``[FAIL]`` / ``[WARN]`` prefix for a finding line in the text report.

    Parameters
    ----------
    finding_type : str
        The finding's type (see ``severity.TIERS``).
    pool : Literal["train", "test"] | None
        The pool of a within-pool finding, or ``None`` if it spans both.
    context : TierContext
        The settings the finding's tier may depend on.

    Returns
    -------
    str
        The bracketed tier.
    """
    return f"[{severity_of(finding_type, pool, context)}]"


def _fmt_location(location: Location) -> str:
    """Format a record location dict as a single readable line.

    Parameters
    ----------
    location : Location
        A location dict, as returned by ``MoleculeRecord.location``.

    Returns
    -------
    str
        A string like ``[train] data/train.csv (row 3): 'CCO'``.
    """
    return (
        f"[{location['pool']}] {location['source_file']} "
        f"(row {location['row_index']}): {location['raw_smiles']!r}"
    )


def _fmt_locations(locations: Sequence[Location], indent: str = "      ") -> list[str]:
    """Format a list of record locations as indented lines.

    Parameters
    ----------
    locations : Sequence[Location]
        The location dicts to format.
    indent : str
        Prefix to indent each line with.

    Returns
    -------
    list[str]
        One formatted, indented line per location.
    """
    return [f"{indent}{_fmt_location(loc)}" for loc in locations]


def _section(title: str, body_lines: list[str]) -> list[str]:
    """Build a titled section of the text report, with a placeholder if empty.

    Parameters
    ----------
    title : str
        The section title.
    body_lines : list[str]
        The section's content lines.

    Returns
    -------
    list[str]
        The title, an underline, the body lines (or a "(none)"
        placeholder if ``body_lines`` is empty), and a trailing blank
        line.
    """
    lines = [title, "-" * len(title)]
    lines.extend(body_lines if body_lines else ["  (none)"])
    lines.append("")
    return lines


def _exact_leakage_lines(
    exact_leakage: dict[str, list[LeakageFinding]], representations: Iterable[str]
) -> list[str]:
    """Render train/test leakage findings for the given representations.

    Parameters
    ----------
    exact_leakage : dict[str, list[LeakageFinding]]
        The result of ``checks.train_test_leakage``.
    representations : Iterable[str]
        Which representations' findings to render, in order.

    Returns
    -------
    list[str]
        One block of lines per representation with a leaked value.
    """
    lines = []
    for representation in representations:
        tag = _tag(leakage_finding_type(representation), None, TierContext())
        for finding in exact_leakage.get(representation, []):
            lines.append(f"  {tag} {representation} = {finding['value']!r}")
            lines.extend(_fmt_locations(finding["train_occurrences"]))
            lines.extend(_fmt_locations(finding["test_occurrences"]))
    return lines


def _identifier_leakage_lines(id_leakage: dict[str, list[LeakageFinding]]) -> list[str]:
    """Render identifier-column overlap findings between train and test.

    Parameters
    ----------
    id_leakage : dict[str, list[LeakageFinding]]
        The result of ``checks.identifier_leakage``.

    Returns
    -------
    list[str]
        One block of lines per identifier column value shared across
        pools.
    """
    lines = []
    tag = _tag("identifier_overlap", None, TierContext())
    for column, findings in id_leakage.items():
        for finding in findings:
            lines.append(f"  {tag} {column} = {finding['value']!r}")
            lines.extend(_fmt_locations(finding["train_occurrences"]))
            lines.extend(_fmt_locations(finding["test_occurrences"]))
    return lines


def _tanimoto_lines(tanimoto: list[TanimotoFinding], context: TierContext) -> list[str]:
    """Render Tanimoto similarity leakage findings.

    Parameters
    ----------
    tanimoto : list[TanimotoFinding]
        The result of ``checks.tanimoto_leakage``.
    context : TierContext
        The settings the findings' tier depends on.

    Returns
    -------
    list[str]
        One block of lines per flagged train/test pair.
    """
    lines = []
    tag = _tag("tanimoto_similarity", None, context)
    for finding in tanimoto:
        lines.append(f"  {tag} similarity = {finding['similarity']:.3f}")
        lines.extend(_fmt_locations([finding["train"], finding["test"]]))
    return lines


def _internal_duplicates_lines(
    dupes: InternalDuplicates, pool: Literal["train", "test"]
) -> list[str]:
    """Render within-split internal duplicate findings.

    Parameters
    ----------
    dupes : InternalDuplicates
        The result of ``checks.internal_duplicates`` for one pool.
    pool : Literal["train", "test"]
        Which pool ``dupes`` is from, which the findings' tier depends on.

    Returns
    -------
    list[str]
        One block of lines per duplicated value, across representations
        and identifier columns.
    """
    lines = []
    for representation in REPRESENTATIONS:
        tag = _tag(duplicate_finding_type(representation), pool, TierContext())
        for finding in dupes[representation]:
            lines.append(f"  {tag} {representation} = {finding['value']!r}")
            lines.extend(_fmt_locations(finding["occurrences"]))
    tag = _tag("duplicate_identifier", pool, TierContext())
    for column, id_findings in dupes["identifiers"].items():
        for finding in id_findings:
            lines.append(f"  {tag} identifier[{column}] = {finding['value']!r}")
            lines.extend(_fmt_locations(finding["occurrences"]))
    return lines


def _identifier_namespace_lines(namespace_issues: dict[str, NamespaceIssues]) -> list[str]:
    """Render identifier-to-structure namespace consistency findings.

    Parameters
    ----------
    namespace_issues : dict[str, NamespaceIssues]
        The result of ``checks.identifier_namespace_issues``.

    Returns
    -------
    list[str]
        One block of lines per identifier column with namespace issues.
    """
    lines = []
    one_id_tag = _tag("one_id_multiple_structures", None, TierContext())
    one_structure_tag = _tag("one_structure_multiple_ids", None, TierContext())
    near_tag = _tag("identifier_near_duplicate", None, TierContext())
    whitespace_tag = _tag("identifier_whitespace", None, TierContext())
    broken_tag = _tag("id_with_unparseable_and_valid_structures", None, TierContext())
    missing_tag = _tag("missing_structure_resolvable_by_id", None, TierContext())
    for column, issues in namespace_issues.items():
        for finding in issues["one_id_multiple_structures"]:
            lines.append(
                f"  {one_id_tag} {column}: identifier {finding['identifier']!r} "
                "-> multiple structures"
            )
            for structure in finding["structures"]:
                lines.append(
                    f"    canonical_smiles={structure['canonical_smiles']!r} "
                    f"inchikey={structure['inchikey']!r}"
                )
                lines.extend(_fmt_locations(structure["occurrences"], indent="        "))
        for ids_finding in issues["one_structure_multiple_ids"]:
            lines.append(
                f"  {one_structure_tag} {column}: structure "
                f"(canonical_smiles={ids_finding['canonical_smiles']!r}, "
                f"inchikey={ids_finding['inchikey']!r}) -> multiple identifiers"
            )
            for identifier in ids_finding["identifiers"]:
                lines.append(f"    identifier={identifier['identifier']!r}")
                lines.extend(_fmt_locations(identifier["occurrences"], indent="        "))
        for group in issues["near_duplicates"]:
            values = ", ".join(repr(i["identifier"]) for i in group["identifiers"])
            lines.append(
                f"  {near_tag} {column}: near-duplicate identifiers {values} "
                "(differ only in case or -/_)"
            )
            for identifier in group["identifiers"]:
                lines.append(f"    identifier={identifier['identifier']!r}")
                lines.extend(_fmt_locations(identifier["occurrences"], indent="        "))
        for broken in issues["id_with_unparseable_and_valid_structures"]:
            lines.append(
                f"  {broken_tag} {column}: identifier {broken['identifier']!r} has valid "
                "and unparseable SMILES"
            )
            for structure in broken["structures"]:
                lines.append(f"    valid: canonical_smiles={structure['canonical_smiles']!r}")
                lines.extend(_fmt_locations(structure["occurrences"], indent="        "))
            for row in broken["unparseable"]:
                lines.append(
                    f"    unparseable ({row['category']}): {_fmt_location(row)} -- {row['error']}"
                )
        for missing in issues["missing_structure_resolvable_by_id"]:
            verdict = (
                "ambiguous: several candidate structures" if missing["ambiguous"] else "resolvable"
            )
            lines.append(
                f"  {missing_tag} {column}: identifier {missing['identifier']!r} has empty "
                f"SMILES ({verdict})"
            )
            lines.extend(_fmt_locations(missing["missing"], indent="    empty: "))
            for structure in missing["candidates"]:
                lines.append(f"    candidate: canonical_smiles={structure['canonical_smiles']!r}")
                lines.extend(_fmt_locations(structure["occurrences"], indent="        "))
        for entry in issues["identifier_whitespace"]:
            lines.append(
                f"  {whitespace_tag} {column}: {entry['raw_value']!r} has surrounding "
                f"whitespace, compared as {entry['value']!r}: {_fmt_location(entry)}"
            )
    return lines


def _quality_flag_lines(quality_flags: QualityFlags, pool: Literal["train", "test"]) -> list[str]:
    """Render quality-filter findings (mixtures, salts/metals, suspicious fragments).

    Parameters
    ----------
    quality_flags : QualityFlags
        The result of ``quality_flag_entries`` for one pool.
    pool : Literal["train", "test"]
        Which pool ``quality_flags`` is from.

    Returns
    -------
    list[str]
        One block of lines per flagged category with findings.
    """
    lines = []
    for category in QUALITY_FLAG_CATEGORIES:
        locations = quality_flags[category]
        finding_type = QUALITY_FLAG_FINDING_TYPES[category]
        if not locations or finding_type is None:
            continue  # nothing found, or a deprecated total repeating other categories
        tag = _tag(finding_type, pool, TierContext())
        lines.append(f"  {tag} {category}:")
        if category == "rdkit_warnings":
            lines.extend(
                f"    {_fmt_location(entry)} -- {entry['warning']}"
                for entry in quality_flags["rdkit_warnings"]
            )
        elif category == "net_charge":
            lines.extend(
                f"    {_fmt_location(entry)} -- net charge {entry['charge']:+d}"
                for entry in quality_flags["net_charge"]
            )
        else:
            lines.extend(_fmt_locations(locations, indent="    "))
    return lines


def format_report_as_text(report: Report) -> str:
    """Render the full audit report as human-readable, indented plain text.

    Contains the same findings as the JSON report (exact row indices, source
    files, values, and similarity scores) rendered as prose instead of raw
    JSON, for easier manual review.

    Parameters
    ----------
    report : Report
        The report dict, as returned by ``build_report``.

    Returns
    -------
    str
        The full text report as a single string.
    """
    cfg = report["config"]
    context = _tier_context(report)
    unparseable_tag = _tag("unparseable_smiles", "train", context)
    lines = [
        "=" * 72,
        "challenge-data-checker audit report",
        "=" * 72,
        f"Generated: {report['generated_at']}",
        "",
        "Configuration",
        "-------------",
        f"  train_files              : {cfg['train_files']}",
        f"  test_files               : {cfg['test_files']}",
        f"  smiles_column            : {cfg['smiles_column']}",
        f"  identifier_columns       : {cfg['identifier_columns']}",
        f"  tautomer_standardisation : {cfg['tautomer_standardisation']}",
        f"  strip_salts              : {cfg['strip_salts']}",
        f"  identifier_case_insens.  : {cfg['identifier_case_insensitive']}",
        f"  max_train_test_similarity: {cfg['max_train_test_similarity']}",
        f"  fp_radius                : {cfg['fp_radius']}",
        f"  fp_n_bits                : {cfg['fp_n_bits']}",
        "",
    ]
    lines.extend(_build_summary_lines(report))
    lines.append("")

    lines.extend(
        _section(
            "Unparseable SMILES",
            [
                f"  {unparseable_tag} {_fmt_location(entry)} -- "
                f"{entry['category']}: {entry['error']}"
                + (
                    f" (resolvable via identifier column {entry['resolvable_via_identifier']!r})"
                    if entry["resolvable_via_identifier"]
                    else ""
                )
                for entry in report["unparseable_smiles"]
            ],
        )
    )
    lines.extend(
        _section(
            "Train-Test Leakage: Exact Matches",
            _exact_leakage_lines(report["train_test_leakage"]["exact"], EXACT_REPRESENTATIONS),
        )
    )
    lines.extend(
        _section(
            "Train-Test Leakage: Same Parent Compound, Different Salt Form",
            _exact_leakage_lines(
                report["train_test_leakage"]["exact"],
                sorted(SALT_FORM_INSENSITIVE_REPRESENTATIONS),
            ),
        )
    )
    lines.extend(
        _section(
            "Train-Test Leakage: Same Connectivity, Different Stereochemistry",
            _exact_leakage_lines(
                report["train_test_leakage"]["exact"],
                sorted(STEREO_INSENSITIVE_REPRESENTATIONS),
            ),
        )
    )
    lines.extend(
        _section(
            "Train-Test Leakage: Identifier Overlap",
            _identifier_leakage_lines(report["train_test_leakage"]["identifiers"]),
        )
    )
    lines.extend(
        _section(
            "Train-Test Leakage: Tanimoto Similarity",
            _tanimoto_lines(report["train_test_leakage"]["tanimoto_similarity"], context),
        )
    )
    lines.extend(
        _section(
            "Internal Duplicates: Train",
            _internal_duplicates_lines(report["internal_duplicates"]["train"], "train"),
        )
    )
    lines.extend(
        _section(
            "Internal Duplicates: Test",
            _internal_duplicates_lines(report["internal_duplicates"]["test"], "test"),
        )
    )
    lines.extend(
        _section(
            "Identifier Namespace Issues",
            _identifier_namespace_lines(report["identifier_namespace_issues"]),
        )
    )
    lines.extend(
        _section(
            "Quality Flags: Train",
            _quality_flag_lines(report["quality_flags"]["train"], "train"),
        )
    )
    lines.extend(
        _section(
            "Quality Flags: Test",
            _quality_flag_lines(report["quality_flags"]["test"], "test"),
        )
    )

    return "\n".join(lines).rstrip() + "\n"
