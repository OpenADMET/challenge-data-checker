import sqlite3

import pytest
from chembl_fixture import (
    CYP3A4_ASSAY,
    CYP3A4_DOCUMENT,
    CYP3A4_DOCUMENT_2,
    CYP3A4_DOCUMENT_3,
    CYP3A4_TARGET,
    OTHER_VERSION_NAME,
    P450_FAMILY_TARGET,
    PIPELINE_VERSION_NAME,
    RELEASE_DATE,
    RELEASE_NAME,
    build_fixture,
)

from challenge_data_checker.chembl import ChemblSourceError, connect, read_source
from challenge_data_checker.chemistry import MoleculeProcessor
from challenge_data_checker.config import (
    ColumnsConfig,
    ConfigError,
    PublicSearchConfig,
    PublicSearchPathsConfig,
    SettingsConfig,
    parse_public_search,
)
from challenge_data_checker.public_search import _is_measured
from challenge_data_checker.report import (
    build_public_search_report,
    format_report_as_text,
    print_public_search_summary,
)
from challenge_data_checker.report_types import ChemblActivity
from challenge_data_checker.severity import TIERS, Severity, TierContext, severity_of

_PROCESSOR = MoleculeProcessor(SettingsConfig())

# Test-set structures, each exercising one matching path against the fixture.
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
PROPRANOLOL_FREE_BASE = "CC(C)NCC(O)COc1cccc2ccccc12"
S_IBUPROFEN = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
RACEMIC_IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
# In ChEMBL with a CYP3A4 record whose activity_comment says the endpoint was
# never measured.
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
# Its only CYP3A4 record is a text result with no number.
WARFARIN = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
# Measured in one paper itself, and in another through its salt form.
METOPROLOL_FREE_BASE = "CC(C)NCC(O)COc1ccc(CCOC)cc1"
# Its activity's own doc_id differs from its assay's.
THEOPHYLLINE = "Cn1c(=O)c2[nH]cnc2n(C)c1=O"
# Measured only against the whole P450 family target.
NICOTINE = "CN1CCC[C@H]1c1cccnc1"
# Not in the fixture at all: the negative control.
OCTANOL = "CCCCCCCCO"


@pytest.fixture(scope="module")
def chembl(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("chembl") / "chembl_fixture.sqlite")


def search(chembl_path, smiles, **public_search):
    """Search the fixture release for some test SMILES, without touching disk."""
    settings = {"chembl_sqlite": str(chembl_path), **public_search}
    settings.setdefault("target_chembl_ids", [CYP3A4_TARGET])
    config = PublicSearchConfig(
        paths=PublicSearchPathsConfig(test_files=["test.csv"], report_output=None),
        columns=ColumnsConfig(smiles_column="SMILES", identifier_columns=[]),
        settings=SettingsConfig(),
        public_search=parse_public_search(settings),
        config_path=None,
    )
    records = [_PROCESSOR.process_row("test", "test.csv", i, s, {}) for i, s in enumerate(smiles)]
    return build_public_search_report(config, records, [])


def matches(report):
    """Reduce a report's findings to (representation, match_type) pairs."""
    return [(f["representation"], f["match_type"]) for f in report["findings"]]


def keys_of(smiles):
    """The InChIKey and salt-stripped parent InChIKey the checker computes."""
    record = _PROCESSOR.process_row("test", "<keys>", 0, smiles, {})
    return record.inchikey, record.parent_inchikey


def test_every_public_finding_type_has_a_tier():
    for finding_type in (
        "public_assay_overlap",
        "public_assay_stereo_overlap",
        "public_target_type_overlap",
    ):
        assert finding_type in TIERS


def test_only_a_named_assay_on_an_exact_structure_blocks():
    # Weaken either axis of the match and the finding stops blocking.
    tiers = {
        name: severity_of(name, None, TierContext())
        for name in (
            "public_assay_overlap",
            "public_assay_stereo_overlap",
            "public_target_type_overlap",
        )
    }
    assert tiers == {
        "public_assay_overlap": Severity.FAIL,
        "public_assay_stereo_overlap": Severity.WARN,
        "public_target_type_overlap": Severity.WARN,
    }


def test_exact_inchikey_match_with_named_target_blocks(chembl):
    report = search(chembl, [CAFFEINE])
    assert report["summary"]["status"] == "FAIL"
    assert report["summary"]["blocking"]["by_type"]["public_assay_overlap"]["findings"] == 1
    # One leak, reported once: the same compound also matches through the
    # parent and connectivity keys, which must not become extra findings.
    assert matches(report) == [("inchikey", "explicit_id")]
    inchikey, _parent = keys_of(CAFFEINE)
    assert report["findings"][0]["value"] == inchikey


def test_salt_form_match_goes_through_molecule_hierarchy(chembl):
    # The test set holds the free base; ChEMBL's CYP3A4 measurement is on the
    # hydrochloride. They only meet through molecule_hierarchy's parent, which
    # is the leakage mode salts actually cause.
    report = search(chembl, [PROPRANOLOL_FREE_BASE])
    assert report["summary"]["status"] == "FAIL"
    assert matches(report) == [("parent_inchikey", "explicit_id")]

    _inchikey, parent = keys_of(PROPRANOLOL_FREE_BASE)
    finding = report["findings"][0]
    assert finding["value"] == parent
    assert "CHEMBL1200640" in {compound["chembl_id"] for compound in finding["compounds"]}
    assert all(activity["target_chembl_id"] == CYP3A4_TARGET for activity in finding["activities"])


def test_stereo_difference_matches_only_on_connectivity_and_is_flagged(chembl):
    # (S)-ibuprofen against racemic ibuprofen in ChEMBL: no exact or parent
    # match, so this is the case connectivity matching exists for.
    off = search(chembl, [S_IBUPROFEN], match_connectivity=False)
    assert off["findings"] == []
    assert off["summary"]["status"] == "PASS"

    on = search(chembl, [S_IBUPROFEN], match_connectivity=True)
    assert matches(on) == [("inchikey_connectivity", "explicit_id")]
    assert on["summary"]["status"] == "WARN"
    assert on["summary"]["flagged"]["by_type"]["public_assay_stereo_overlap"]["findings"] == 1


def test_target_resolved_from_a_gene_symbol_is_flagged_not_blocking(chembl):
    report = search(
        chembl,
        [CAFFEINE],
        target_chembl_ids=[],
        target_names=["CYP3A4"],
        standard_types=["IC50"],
    )
    assert report["summary"]["status"] == "WARN"
    assert matches(report) == [("inchikey", "target_type")]


def test_an_unresolvable_target_name_matches_nothing(chembl):
    report = search(chembl, [CAFFEINE], target_chembl_ids=[], target_names=["NOT_A_REAL_TARGET"])
    assert report["findings"] == []
    assert report["summary"]["status"] == "PASS"


def test_standard_types_filter_out_the_wrong_measurement(chembl):
    # Ibuprofen's only CYP3A4 record is a Ki, so an IC50-only config must not
    # match it, while an empty standard_types accepts any measurement.
    filtered = search(
        chembl,
        [S_IBUPROFEN],
        match_connectivity=True,
        target_chembl_ids=[],
        target_names=["CYP3A4"],
        standard_types=["IC50"],
    )
    assert filtered["findings"] == []

    unfiltered = search(
        chembl,
        [S_IBUPROFEN],
        match_connectivity=True,
        target_chembl_ids=[],
        target_names=["CYP3A4"],
        standard_types=[],
    )
    assert matches(unfiltered) == [("inchikey_connectivity", "target_type")]


def test_presence_in_chembl_without_the_endpoint_is_counted_not_flagged(chembl):
    # Aspirin is in the fixture, but only measured against an unrelated
    # target. Presence alone must never produce a finding (#18).
    report = search(chembl, [ASPIRIN])
    assert report["findings"] == []
    assert report["summary"]["status"] == "PASS"
    assert report["summary"]["total_found_in_source"] == 1
    assert report["summary"]["total_test_rows"] == 1
    assert report["presence"] != []
    assert {cid for entry in report["presence"] for cid in entry["chembl_ids"]} == {"CHEMBL25"}


def test_negative_control_compound_absent_from_chembl_matches_nothing(chembl):
    # The control the matchers must stay silent on: if this ever produces a
    # finding, the matching is wrong rather than the data being leaked.
    report = search(chembl, [OCTANOL], match_connectivity=True)
    assert report["findings"] == []
    assert report["presence"] == []
    assert report["summary"]["total_found_in_source"] == 0
    assert report["summary"]["status"] == "PASS"


def test_negative_control_alongside_a_real_leak_does_not_contaminate_it(chembl):
    report = search(chembl, [OCTANOL, CAFFEINE, OCTANOL], match_connectivity=True)
    assert report["summary"]["total_test_rows"] == 3
    assert report["summary"]["total_found_in_source"] == 1
    assert len(report["findings"]) == 1
    assert [loc["row_index"] for loc in report["findings"][0]["test_occurrences"]] == [1]


def test_document_summary_reports_the_share_of_the_test_set(chembl):
    report = search(chembl, [CAFFEINE, OCTANOL, OCTANOL, OCTANOL])
    (document,) = report["by_document"]
    assert document["document_chembl_id"] == CYP3A4_DOCUMENT
    assert document["test_rows"] == 1
    assert document["fraction_of_test_set"] == pytest.approx(0.25)
    assert "J Med Chem" in document["reference"]


def test_report_records_which_release_it_searched(chembl):
    report = search(chembl, [CAFFEINE])
    assert report["mode"] == "public_search"
    assert report["source"]["release"] == RELEASE_NAME
    assert report["source"]["release_date"] == RELEASE_DATE


def test_named_assay_id_matches_as_well_as_a_target_id(chembl):
    report = search(chembl, [CAFFEINE], target_chembl_ids=[], assay_chembl_ids=[CYP3A4_ASSAY])
    assert matches(report) == [("inchikey", "explicit_id")]
    assert report["summary"]["status"] == "FAIL"


def test_unparsed_rows_are_skipped_without_failing_the_search(chembl):
    report = search(chembl, ["not-a-smiles", CAFFEINE])
    assert len(report["findings"]) == 1
    assert report["summary"]["total_test_rows"] == 2


def test_text_and_dashboard_renderings_cover_every_section(chembl, capsys):
    report = search(chembl, [CAFFEINE, ASPIRIN, OCTANOL], match_connectivity=True)
    text = format_report_as_text(report)
    assert "public-source search report" in text
    assert RELEASE_NAME in text
    assert CYP3A4_DOCUMENT in text
    assert "presence alone is not a finding" in text

    print_public_search_summary(report, None)
    assert "public-source search summary" in capsys.readouterr().out


def test_missing_release_is_a_clear_error(tmp_path):
    with pytest.raises(ChemblSourceError, match="not found"):
        connect(tmp_path / "absent.sqlite")


def test_a_sqlite_file_that_is_not_chembl_is_rejected(tmp_path):
    path = tmp_path / "other.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ChemblSourceError, match="does not look like a ChEMBL release"):
        connect(path)


def test_a_release_without_a_version_table_still_searches(tmp_path, chembl):
    # A trimmed dump should degrade to an unknown release rather than refuse to
    # run: the lookup tables are what the search actually needs.
    path = tmp_path / "no_version.sqlite"
    path.write_bytes(chembl.read_bytes())
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE version")
    conn.commit()
    conn.close()

    opened = connect(path)
    try:
        source = read_source(opened, path)
    finally:
        opened.close()
    assert source["release"] is None

    report = search(path, [CAFFEINE])
    assert report["source"]["release"] is None
    assert report["summary"]["status"] == "FAIL"


def test_config_requires_an_endpoint_to_match():
    with pytest.raises(ConfigError, match="at least one of"):
        parse_public_search({"chembl_sqlite": "chembl.sqlite"})


def test_config_requires_a_release_path():
    with pytest.raises(ConfigError, match="chembl_sqlite"):
        parse_public_search({"target_names": ["CYP3A4"]})


def test_config_rejects_a_non_list_matcher():
    with pytest.raises(ConfigError, match="must be a list of strings"):
        parse_public_search({"chembl_sqlite": "c.sqlite", "target_names": "CYP3A4"})


def test_partially_overlapping_groups_do_not_report_a_row_twice(chembl):
    # Regression: searching a racemate together with one of its enantiomers,
    # the connectivity group holds both rows while the exact group holds only
    # the racemate. The stereo finding must report the enantiomer alone;
    # otherwise the racemate sits in a blocking and a flagged finding at once
    # and is counted twice, which #4 forbids.
    report = search(chembl, [RACEMIC_IBUPROFEN, S_IBUPROFEN], match_connectivity=True)
    assert matches(report) == [
        ("inchikey", "explicit_id"),
        ("inchikey_connectivity", "explicit_id"),
    ]
    rows_per_finding = [
        {loc["row_index"] for loc in finding["test_occurrences"]} for finding in report["findings"]
    ]
    assert rows_per_finding == [{0}, {1}]
    assert report["summary"]["blocking"]["rows"] == 1
    assert report["summary"]["flagged"]["rows"] == 1


def test_evidence_from_a_second_document_survives_deduplication(chembl):
    # Regression: metoprolol is measured in document 2 as the free base and in
    # document 1 as its hydrochloride. The exact match wins the row, so only
    # one finding is reported, but the document summary must still count both
    # papers: dropping one hides a publication that holds the answer.
    report = search(chembl, [METOPROLOL_FREE_BASE])
    assert matches(report) == [("inchikey", "explicit_id")]
    assert {summary["document_chembl_id"] for summary in report["by_document"]} == {
        CYP3A4_DOCUMENT,
        CYP3A4_DOCUMENT_2,
    }


def test_an_activity_marked_not_determined_is_not_a_leak(chembl):
    # Regression: ChEMBL holds CYP3A4 rows whose activity_comment says the
    # endpoint was never measured. It was looked at, not measured, so no
    # answer is exposed and the run must not be blocked.
    report = search(chembl, [PARACETAMOL])
    assert report["findings"] == []
    assert report["summary"]["status"] == "PASS"
    assert report["summary"]["total_found_in_source"] == 1


def test_only_the_comment_decides_whether_a_record_was_measured(chembl):
    # The narrow rule, stated as a unit: the comment is the only thing that
    # rejects a record, matched case- and whitespace-insensitively, and a
    # record with no comment counts however little else it carries.
    blank: ChemblActivity = {
        "assay_chembl_id": None,
        "assay_description": None,
        "target_chembl_id": None,
        "target_name": None,
        "organism": None,
        "standard_type": None,
        "standard_relation": None,
        "standard_value": None,
        "standard_units": None,
        "pchembl_value": None,
        "standard_text_value": None,
        "activity_comment": None,
        "document_chembl_id": None,
        "document_reference": None,
    }
    assert _is_measured(blank)
    for comment in ("Not Determined", "not determined", "  NOT TESTED  ", "Inconclusive"):
        assert not _is_measured({**blank, "activity_comment": comment}), comment
    # A qualitative verdict is a result, not a missing one: a compound
    # reported inactive against the challenge endpoint is still an answer.
    for comment in ("Not Active", "Inactive", "Active", "Potency determined"):
        assert _is_measured({**blank, "activity_comment": comment}), comment


def test_a_text_only_result_is_a_leak(chembl):
    # Regression: warfarin's only CYP3A4 record has no number, just the text
    # result "Not active". Requiring a numeric value dropped every
    # qualitative endpoint, so this must be a finding. Negative control for
    # the narrowed rule: restoring the numeric requirement makes it fail.
    report = search(chembl, [WARFARIN])
    assert matches(report) == [("inchikey", "explicit_id")]
    assert report["summary"]["status"] == "FAIL"
    (activity,) = report["findings"][0]["activities"]
    assert activity["standard_value"] is None
    assert activity["standard_text_value"] == "Not active"
    # The rendering must show the result, not an empty measurement.
    assert "Activity = Not active" in format_report_as_text(report)


def test_activity_is_cited_to_its_own_document_not_its_assays(chembl):
    # Regression: ChEMBL lets an activity carry a doc_id different from its
    # assay's. Citing the assay's document attributes the measurement to the
    # wrong paper and miscounts document coverage.
    report = search(chembl, [THEOPHYLLINE])
    documents = {
        activity["document_chembl_id"]
        for finding in report["findings"]
        for activity in finding["activities"]
    }
    assert documents == {CYP3A4_DOCUMENT_3}
    assert [summary["document_chembl_id"] for summary in report["by_document"]] == [
        CYP3A4_DOCUMENT_3
    ]


def test_release_is_chosen_by_name_not_by_row_order(chembl):
    # Regression: the version table is keyed by name and need not hold only
    # the database release. The fixture puts a non-ChEMBL row first, so taking
    # whichever row comes first would report the wrong release.
    conn = connect(chembl)
    try:
        names = {str(row[0]) for row in conn.execute("SELECT name FROM version")}
        source = read_source(conn, chembl)
    finally:
        conn.close()
    assert OTHER_VERSION_NAME in names, "fixture should hold a non-ChEMBL version row"
    assert source["release"] == RELEASE_NAME


def test_a_markup_like_report_path_does_not_break_the_dashboard(chembl, capsys):
    # Regression: the path is interpolated into Rich markup, so one containing
    # "[/]" would close the bold tag early and raise MarkupError.
    report = search(chembl, [CAFFEINE])
    print_public_search_summary(report, "out[/].json")
    out = capsys.readouterr().out
    assert "public-source search summary" in out
    assert "out[/].json" in out


def test_release_is_not_confused_with_a_chembl_named_tool_version(chembl):
    # Regression from real data: ChEMBL 37's version table has 11 rows and
    # "ChEMBL_Structure_Pipeline 1.2.0" also starts with "ChEMBL". The fixture
    # inserts it before the release row, so a prefix match would report the
    # structure pipeline as the database release.
    conn = connect(chembl)
    try:
        names = [str(row[0]) for row in conn.execute("SELECT name FROM version")]
        source = read_source(conn, chembl)
    finally:
        conn.close()
    assert names.index(PIPELINE_VERSION_NAME) < names.index(RELEASE_NAME)
    assert source["release"] == RELEASE_NAME


def test_gene_symbol_resolves_to_single_proteins_by_default(chembl):
    # Regression from real data: in ChEMBL 37 "CYP3A4" also resolves to three
    # protein families, the broadest being every cytochrome P450. Nicotine is
    # measured only against that family, so it must not be flagged as CYP3A4.
    report = search(chembl, [NICOTINE], target_chembl_ids=[], target_names=["CYP3A4"])
    assert report["findings"] == []
    assert report["summary"]["status"] == "PASS"
    assert report["config"]["target_types"] == ["SINGLE PROTEIN"]


def test_target_types_can_be_widened_to_protein_families(chembl):
    report = search(
        chembl,
        [NICOTINE],
        target_chembl_ids=[],
        target_names=["CYP3A4"],
        target_types=["SINGLE PROTEIN", "PROTEIN FAMILY"],
    )
    assert matches(report) == [("inchikey", "target_type")]
    targets = {a["target_chembl_id"] for f in report["findings"] for a in f["activities"]}
    assert targets == {P450_FAMILY_TARGET}


def test_config_rejects_empty_target_types():
    with pytest.raises(ConfigError, match="target_types"):
        parse_public_search(
            {"chembl_sqlite": "c.sqlite", "target_names": ["CYP3A4"], "target_types": []}
        )
