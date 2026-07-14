"""Assembling the audit report (JSON or text) and the stdout summary dashboard."""

import json
from pathlib import Path

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


def quality_flag_entries(records: list[MoleculeRecord]) -> dict[str, list[dict]]:
    """Collect quality-filter findings (mixtures, salts/metals, suspicious fragments).

    Args:
        records: Processed records from a single pool.

    Returns:
        A mapping of ``"mixtures"``, ``"salts_or_metal_complexes"``, and
        ``"suspicious_fragments"`` to the list of matching record locations.

    """
    return {
        "mixtures": [r.location() for r in records if r.is_parsed and r.is_mixture],
        "salts_or_metal_complexes": [
            r.location() for r in records if r.is_parsed and r.is_salt_or_metal
        ],
        "suspicious_fragments": [
            r.location() for r in records if r.is_parsed and r.is_suspicious_fragment
        ],
    }


def _unique_leaked_locations(exact_leakage: dict[str, list[dict]], side: str) -> int:
    """Count the distinct rows involved in exact leakage findings on one side.

    Args:
        exact_leakage: The result of ``checks.train_test_leakage``.
        side: Which side to count, ``"train"`` or ``"test"``.

    Returns:
        The number of distinct (pool, source_file, row_index) rows appearing
        as a ``{side}_occurrences`` entry across all representations.

    """
    seen = set()
    for entries in exact_leakage.values():
        for entry in entries:
            for occ in entry[f"{side}_occurrences"]:
                seen.add((occ["pool"], occ["source_file"], occ["row_index"]))
    return len(seen)


def build_report(
    config: Config,
    train_records: list[MoleculeRecord],
    test_records: list[MoleculeRecord],
    identifier_columns: list[str],
) -> dict:
    """Run all checks and assemble the full JSON-serialisable audit report.

    Args:
        config: The validated configuration used for this run.
        train_records: Processed records from the training pool.
        test_records: Processed records from the test pool.
        identifier_columns: Identifier columns present in at least one pool.

    Returns:
        The full report dict, containing the echoed config, a summary of
        counts, and the detailed findings from every check.

    """
    all_records = train_records + test_records

    unparseable = unparseable_entries(all_records)
    exact_leakage = train_test_leakage(train_records, test_records)
    id_leakage = identifier_leakage(train_records, test_records, identifier_columns)
    tanimoto = tanimoto_leakage(
        train_records, test_records, config.settings.max_train_test_similarity
    )
    train_dupes = internal_duplicates(train_records, identifier_columns)
    test_dupes = internal_duplicates(test_records, identifier_columns)
    namespace_issues = identifier_namespace_issues(all_records, identifier_columns)
    quality_train = quality_flag_entries(train_records)
    quality_test = quality_flag_entries(test_records)

    summary = {
        "total_train_rows": len(train_records),
        "total_test_rows": len(test_records),
        "total_unparseable": len(unparseable),
        "total_unparseable_train": sum(1 for r in train_records if not r.is_parsed),
        "total_unparseable_test": sum(1 for r in test_records if not r.is_parsed),
        "total_exact_leaked_train_molecules": _unique_leaked_locations(exact_leakage, "train"),
        "total_exact_leaked_test_molecules": _unique_leaked_locations(exact_leakage, "test"),
        "total_identifier_leakage_events": sum(len(v) for v in id_leakage.values()),
        "total_tanimoto_leakage_pairs": len(tanimoto),
        "total_internal_duplicates_train": {
            k: len(v) for k, v in train_dupes.items() if k != "identifiers"
        },
        "total_internal_duplicates_test": {
            k: len(v) for k, v in test_dupes.items() if k != "identifiers"
        },
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
    }

    return {
        "config": {
            "train_files": [describe_source(s, i) for i, s in enumerate(config.paths.train_files)],
            "test_files": [describe_source(s, i) for i, s in enumerate(config.paths.test_files)],
            "smiles_column": config.columns.smiles_column,
            "identifier_columns": identifier_columns,
            "tautomer_standardisation": config.settings.tautomer_standardisation,
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


def save_report(report: dict, output_path: str | Path, report_format: str = "json") -> None:
    """Write the report to disk as either indented JSON or human-readable text.

    Creates any missing parent directories of ``output_path``.

    Args:
        report: The report dict, as returned by ``build_report``.
        output_path: Path to write the report to.
        report_format: Either ``"json"`` (machine-readable, full fidelity) or
            ``"txt"`` (human-readable prose rendering of the same findings).

    Raises:
        ValueError: If ``report_format`` is not ``"json"`` or ``"txt"``.
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


def _fmt_counts(counts: dict[str, int]) -> str:
    """Format a mapping of counts as a compact ``key=value, ...`` string.

    Args:
        counts: The counts to format.

    Returns:
        A comma-separated ``key=value`` string.

    """
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def _build_summary_lines(report: dict) -> list[str]:
    """Build the summary dashboard as a list of text lines.

    Args:
        report: The report dict, as returned by ``build_report``.

    Returns:
        The dashboard lines, banner included but with no trailing "written
        to" line (callers add that themselves if relevant).
    """
    s = report["summary"]
    lines = [
        "=" * 60,
        "challenge-data-checker audit summary",
        "=" * 60,
        f"Rows processed        : train={s['total_train_rows']}  test={s['total_test_rows']}",
        (
            f"Unparseable SMILES    : total={s['total_unparseable']}  "
            f"(train={s['total_unparseable_train']}, test={s['total_unparseable_test']})"
        ),
        "",
        "Train-test leakage",
        (
            f"  Exact match (raw/canonical/InChIKey) - train molecules leaked: "
            f"{s['total_exact_leaked_train_molecules']}"
        ),
        (
            f"  Exact match (raw/canonical/InChIKey) - test molecules leaked : "
            f"{s['total_exact_leaked_test_molecules']}"
        ),
        (
            f"  Identifier overlap events                                    : "
            f"{s['total_identifier_leakage_events']}"
        ),
        (
            f"  Tanimoto similarity flags (>= threshold)                     : "
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
    lines.append(f"  Mixtures/disconnected  : {_fmt_counts(s['total_mixtures'])}")
    lines.append(f"  Salts/metal complexes  : {_fmt_counts(s['total_salts_or_metal_complexes'])}")
    lines.append(f"  Suspicious fragments   : {_fmt_counts(s['total_suspicious_fragments'])}")
    lines.append("=" * 60)
    return lines


def _count_style(count: int) -> str:
    """Pick a rich style for a finding count, green if zero and red otherwise.

    Args:
        count: The finding count to style.

    Returns:
        ``"bold green"`` if ``count`` is zero, ``"bold red"`` otherwise.
    """
    return "bold green" if count == 0 else "bold red"


def _status_cell(count: int) -> str:
    """Render a count as a colored PASS/FAIL check-mark cell for a rich table.

    Args:
        count: The finding count to render.

    Returns:
        A rich markup string: a green checkmark for zero, a red cross with
        the count otherwise.
    """
    if count == 0:
        return "[bold green]✓ 0[/]"
    return f"[bold red]✗ {count}[/]"


def _new_table(columns: list[str]) -> Table:
    """Create a bare rich table with the given column headers.

    The table has no title of its own; callers print a section heading
    separately so it isn't constrained (and wrapped) by the table's
    auto-sized column widths.

    Args:
        columns: Column header labels, in order.

    Returns:
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

    Args:
        title: The section heading text.
        table: The table to render, or ``None`` to render a "(none)"
            placeholder instead (e.g. no identifier columns configured).

    Returns:
        The renderables for this section, in display order.
    """
    placeholder = "  (no identifier columns configured/available)"
    return [f"[bold cyan]{title}[/]", table if table is not None else placeholder, ""]


def _leakage_table(s: dict) -> Table:
    """Build the train-test leakage summary table.

    Args:
        s: The report's ``"summary"`` dict.

    Returns:
        A rich table with one row per leakage check.
    """
    table = _new_table(["Check", "Result"])
    table.add_row(
        "Exact match - train molecules leaked",
        _status_cell(s["total_exact_leaked_train_molecules"]),
    )
    table.add_row(
        "Exact match - test molecules leaked",
        _status_cell(s["total_exact_leaked_test_molecules"]),
    )
    table.add_row("Identifier overlap events", _status_cell(s["total_identifier_leakage_events"]))
    table.add_row(
        "Tanimoto similarity flags (>= threshold)",
        _status_cell(s["total_tanimoto_leakage_pairs"]),
    )
    return table


def _duplicates_table(s: dict) -> Table:
    """Build the internal (within-split) duplicates summary table.

    Args:
        s: The report's ``"summary"`` dict.

    Returns:
        A rich table with one row per representation/identifier column,
        showing train and test counts side by side.
    """
    table = _new_table(["Representation", "Train", "Test"])
    for representation in ("raw_smiles", "canonical_smiles", "inchikey"):
        table.add_row(
            representation,
            _status_cell(s["total_internal_duplicates_train"][representation]),
            _status_cell(s["total_internal_duplicates_test"][representation]),
        )
    for col in s["total_internal_duplicates_train_identifiers"]:
        table.add_row(
            f"identifier[{col}]",
            _status_cell(s["total_internal_duplicates_train_identifiers"][col]),
            _status_cell(s["total_internal_duplicates_test_identifiers"].get(col, 0)),
        )
    return table


def _namespace_table(s: dict) -> Table | None:
    """Build the identifier namespace consistency summary table.

    Args:
        s: The report's ``"summary"`` dict.

    Returns:
        A rich table with one row per identifier column, or ``None`` if no
        identifier columns were configured/available.
    """
    if not s["total_identifier_namespace_issues"]:
        return None
    table = _new_table(["Identifier column", "1 id -> many structures", "1 structure -> many ids"])
    for col, counts in s["total_identifier_namespace_issues"].items():
        table.add_row(
            col,
            _status_cell(counts["one_id_multiple_structures"]),
            _status_cell(counts["one_structure_multiple_ids"]),
        )
    return table


def _quality_table(s: dict) -> Table:
    """Build the quality-filter flags summary table.

    Args:
        s: The report's ``"summary"`` dict.

    Returns:
        A rich table with one row per quality-filter category, showing train
        and test counts side by side.
    """
    table = _new_table(["Filter", "Train", "Test"])
    table.add_row(
        "Mixtures/disconnected",
        _status_cell(s["total_mixtures"]["train"]),
        _status_cell(s["total_mixtures"]["test"]),
    )
    table.add_row(
        "Salts/metal complexes",
        _status_cell(s["total_salts_or_metal_complexes"]["train"]),
        _status_cell(s["total_salts_or_metal_complexes"]["test"]),
    )
    table.add_row(
        "Suspicious fragments",
        _status_cell(s["total_suspicious_fragments"]["train"]),
        _status_cell(s["total_suspicious_fragments"]["test"]),
    )
    return table


def _total_issue_count(s: dict) -> int:
    """Sum every finding count in the summary into a single overall issue count.

    Args:
        s: The report's ``"summary"`` dict.

    Returns:
        The total number of flagged findings across every check.
    """
    return (
        s["total_unparseable"]
        + s["total_exact_leaked_train_molecules"]
        + s["total_exact_leaked_test_molecules"]
        + s["total_identifier_leakage_events"]
        + s["total_tanimoto_leakage_pairs"]
        + sum(s["total_internal_duplicates_train"].values())
        + sum(s["total_internal_duplicates_test"].values())
        + sum(s["total_internal_duplicates_train_identifiers"].values())
        + sum(s["total_internal_duplicates_test_identifiers"].values())
        + sum(
            count
            for counts in s["total_identifier_namespace_issues"].values()
            for count in counts.values()
        )
        + sum(s["total_mixtures"].values())
        + sum(s["total_salts_or_metal_complexes"].values())
        + sum(s["total_suspicious_fragments"].values())
    )


def print_summary(report: dict, report_output: str | Path | None) -> None:
    """Print the stdout summary dashboard for a completed audit report.

    Renders row counts, unparseable SMILES, and one colored table per check
    category (green checkmarks for clean results, red crosses with counts
    for flagged findings), followed by an overall PASS/FAIL banner. The whole
    dashboard is assembled into a single rich ``Group`` and printed with one
    ``console.print`` call so it renders as one contiguous block rather than
    several (in a Jupyter notebook, each individual ``console.print`` call
    renders as its own separate output).

    Args:
        report: The report dict, as returned by ``build_report``.
        report_output: Path the full report was written to, shown in the
            final line of the dashboard, or ``None`` if it wasn't written to
            disk (that line is then omitted).
    """
    s = report["summary"]

    renderables: list[RenderableType] = [
        Rule("[bold cyan]challenge-data-checker audit summary[/]", style="cyan"),
        f"  Rows processed     : [bold]train[/]={s['total_train_rows']}  "
        f"[bold]test[/]={s['total_test_rows']}",
        f"  Unparseable SMILES : "
        f"[{_count_style(s['total_unparseable'])}]{s['total_unparseable']}[/] "
        f"(train={s['total_unparseable_train']}, test={s['total_unparseable_test']})",
        "",
    ]
    renderables.extend(_section_renderables("Train-test leakage", _leakage_table(s)))
    renderables.extend(
        _section_renderables("Internal duplicates (within split)", _duplicates_table(s))
    )
    renderables.extend(_section_renderables("Identifier namespace issues", _namespace_table(s)))
    renderables.extend(_section_renderables("Quality filter flags", _quality_table(s)))

    total_issues = _total_issue_count(s)
    if total_issues == 0:
        renderables.append(
            Rule("[bold green]✓ OVERALL PASS - no issues detected[/]", style="green")
        )
    else:
        renderables.append(
            Rule(f"[bold red]✗ OVERALL FAIL - {total_issues} issue(s) flagged[/]", style="red")
        )
    if report_output is not None:
        renderables.append(f"Full detailed report written to: [bold]{report_output}[/]")

    Console(highlight=False).print(Group(*renderables))


def _fmt_location(location: dict) -> str:
    """Format a record location dict as a single readable line.

    Args:
        location: A location dict, as returned by ``MoleculeRecord.location``.

    Returns:
        A string like ``[train] data/train.csv (row 3): 'CCO'``.
    """
    return (
        f"[{location['pool']}] {location['source_file']} "
        f"(row {location['row_index']}): {location['raw_smiles']!r}"
    )


def _fmt_locations(locations: list[dict], indent: str = "      ") -> list[str]:
    """Format a list of record locations as indented lines.

    Args:
        locations: The location dicts to format.
        indent: Prefix to indent each line with.

    Returns:
        One formatted, indented line per location.
    """
    return [f"{indent}{_fmt_location(loc)}" for loc in locations]


def _section(title: str, body_lines: list[str]) -> list[str]:
    """Build a titled section of the text report, with a placeholder if empty.

    Args:
        title: The section title.
        body_lines: The section's content lines.

    Returns:
        The title, an underline, the body lines (or a "(none)" placeholder
        if ``body_lines`` is empty), and a trailing blank line.
    """
    lines = [title, "-" * len(title)]
    lines.extend(body_lines if body_lines else ["  (none)"])
    lines.append("")
    return lines


def _exact_leakage_lines(exact_leakage: dict[str, list[dict]]) -> list[str]:
    """Render exact train/test leakage findings (raw/canonical SMILES, InChIKey).

    Args:
        exact_leakage: The result of ``checks.train_test_leakage``.

    Returns:
        One block of lines per representation with a leaked value.
    """
    lines = []
    for representation, findings in exact_leakage.items():
        for finding in findings:
            lines.append(f"  {representation} = {finding['value']!r}")
            lines.extend(_fmt_locations(finding["train_occurrences"]))
            lines.extend(_fmt_locations(finding["test_occurrences"]))
    return lines


def _identifier_leakage_lines(id_leakage: dict[str, list[dict]]) -> list[str]:
    """Render identifier-column overlap findings between train and test.

    Args:
        id_leakage: The result of ``checks.identifier_leakage``.

    Returns:
        One block of lines per identifier column value shared across pools.
    """
    lines = []
    for column, findings in id_leakage.items():
        for finding in findings:
            lines.append(f"  {column} = {finding['value']!r}")
            lines.extend(_fmt_locations(finding["train_occurrences"]))
            lines.extend(_fmt_locations(finding["test_occurrences"]))
    return lines


def _tanimoto_lines(tanimoto: list[dict]) -> list[str]:
    """Render Tanimoto similarity leakage findings.

    Args:
        tanimoto: The result of ``checks.tanimoto_leakage``.

    Returns:
        One block of lines per flagged train/test pair.
    """
    lines = []
    for finding in tanimoto:
        lines.append(f"  similarity = {finding['similarity']:.3f}")
        lines.extend(_fmt_locations([finding["train"], finding["test"]]))
    return lines


def _internal_duplicates_lines(dupes: dict) -> list[str]:
    """Render within-split internal duplicate findings.

    Args:
        dupes: The result of ``checks.internal_duplicates`` for one pool.

    Returns:
        One block of lines per duplicated value, across representations and
        identifier columns.
    """
    lines = []
    for representation, findings in dupes.items():
        if representation == "identifiers":
            for column, id_findings in findings.items():
                for finding in id_findings:
                    lines.append(f"  identifier[{column}] = {finding['value']!r}")
                    lines.extend(_fmt_locations(finding["occurrences"]))
        else:
            for finding in findings:
                lines.append(f"  {representation} = {finding['value']!r}")
                lines.extend(_fmt_locations(finding["occurrences"]))
    return lines


def _identifier_namespace_lines(namespace_issues: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Render identifier-to-structure namespace consistency findings.

    Args:
        namespace_issues: The result of ``checks.identifier_namespace_issues``.

    Returns:
        One block of lines per identifier column with namespace issues.
    """
    lines = []
    for column, issues in namespace_issues.items():
        for finding in issues["one_id_multiple_structures"]:
            lines.append(f"  {column}: identifier {finding['identifier']!r} -> multiple structures")
            for structure in finding["structures"]:
                lines.append(
                    f"    canonical_smiles={structure['canonical_smiles']!r} "
                    f"inchikey={structure['inchikey']!r}"
                )
                lines.extend(_fmt_locations(structure["occurrences"], indent="        "))
        for finding in issues["one_structure_multiple_ids"]:
            lines.append(
                f"  {column}: structure (canonical_smiles={finding['canonical_smiles']!r}, "
                f"inchikey={finding['inchikey']!r}) -> multiple identifiers"
            )
            for identifier in finding["identifiers"]:
                lines.append(f"    identifier={identifier['identifier']!r}")
                lines.extend(_fmt_locations(identifier["occurrences"], indent="        "))
    return lines


def _quality_flag_lines(quality_flags: dict[str, list[dict]]) -> list[str]:
    """Render quality-filter findings (mixtures, salts/metals, suspicious fragments).

    Args:
        quality_flags: The result of ``quality_flag_entries`` for one pool.

    Returns:
        One block of lines per flagged category with findings.
    """
    lines = []
    for category, locations in quality_flags.items():
        if not locations:
            continue
        lines.append(f"  {category}:")
        lines.extend(_fmt_locations(locations, indent="    "))
    return lines


def format_report_as_text(report: dict) -> str:
    """Render the full audit report as human-readable, indented plain text.

    Contains the same findings as the JSON report (exact row indices, source
    files, values, and similarity scores) rendered as prose instead of raw
    JSON, for easier manual review.

    Args:
        report: The report dict, as returned by ``build_report``.

    Returns:
        The full text report as a single string.
    """
    cfg = report["config"]
    lines = [
        "=" * 72,
        "challenge-data-checker audit report",
        "=" * 72,
        "",
        "Configuration",
        "-------------",
        f"  train_files              : {cfg['train_files']}",
        f"  test_files               : {cfg['test_files']}",
        f"  smiles_column            : {cfg['smiles_column']}",
        f"  identifier_columns       : {cfg['identifier_columns']}",
        f"  tautomer_standardisation : {cfg['tautomer_standardisation']}",
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
                f"  {_fmt_location(entry)} -- {entry['error']}"
                for entry in report["unparseable_smiles"]
            ],
        )
    )
    lines.extend(
        _section(
            "Train-Test Leakage: Exact Matches",
            _exact_leakage_lines(report["train_test_leakage"]["exact"]),
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
            _tanimoto_lines(report["train_test_leakage"]["tanimoto_similarity"]),
        )
    )
    lines.extend(
        _section(
            "Internal Duplicates: Train",
            _internal_duplicates_lines(report["internal_duplicates"]["train"]),
        )
    )
    lines.extend(
        _section(
            "Internal Duplicates: Test",
            _internal_duplicates_lines(report["internal_duplicates"]["test"]),
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
            _quality_flag_lines(report["quality_flags"]["train"]),
        )
    )
    lines.extend(
        _section(
            "Quality Flags: Test",
            _quality_flag_lines(report["quality_flags"]["test"]),
        )
    )

    return "\n".join(lines).rstrip() + "\n"
