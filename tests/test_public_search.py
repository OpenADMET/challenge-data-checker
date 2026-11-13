import sqlite3

import pytest
from chembl_fixture import (
    CYP3A4_ASSAY,
    CYP3A4_DOCUMENT,
    CYP3A4_DOCUMENT_2,
    CYP3A4_DOCUMENT_3,
    CYP3A4_DOCUMENT_4,
    CYP3A4_TARGET,
    OTHER_VERSION_NAME,
    P450_FAMILY_TARGET,
    PIPELINE_VERSION_NAME,
    RELEASE_DATE,
    RELEASE_NAME,
    build_fixture,
)

from challenge_data_checker.chembl import (
    _CONNECTIVITY_SQL,
    ChemblSourceError,
    _load_keys,
    connect,
    find_structures,
    read_source,
)
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
# Its CYP3A4 number carries a data_validity_comment.
DICLOFENAC = "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"
# Measured in document 4, cited again in document 1 as a potential duplicate.
NAPROXEN = "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1"
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


def test_connectivity_matching_is_on_unless_turned_off(chembl):
    # Undefined stereochemistry is one of the two leakage modes seen in
    # practice, so a default run has to catch it. Negative control for the
    # default: with match_connectivity off again, this search passes.
    default = search(chembl, [S_IBUPROFEN])
    assert default["config"]["match_connectivity"] is True
    assert matches(default) == [("inchikey_connectivity", "explicit_id")]
    assert default["summary"]["status"] == "WARN"

    assert parse_public_search(
        {"chembl_sqlite": "c.sqlite", "target_names": ["CYP3A4"]}
    ).match_connectivity
    assert not parse_public_search(
        {"chembl_sqlite": "c.sqlite", "target_names": ["CYP3A4"], "match_connectivity": False}
    ).match_connectivity


def test_a_default_connectivity_match_is_flagged_while_exact_stereo_blocks(chembl):
    # The two tiers of the default, in one place: the same compound found
    # with its stereochemistry intact must block, while finding only its
    # connectivity must not.
    exact = search(chembl, [RACEMIC_IBUPROFEN])
    assert exact["summary"]["status"] == "FAIL"
    assert matches(exact) == [("inchikey", "explicit_id")]

    stereo = search(chembl, [S_IBUPROFEN])
    assert stereo["summary"]["status"] == "WARN"
    assert matches(stereo) == [("inchikey_connectivity", "explicit_id")]


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


def test_a_flagged_value_is_still_a_finding_and_says_so(chembl):
    # ChEMBL's data_validity_comment says the number may be wrong, not that
    # the endpoint was unmeasured. A wrong number in a public paper still
    # unblinds the compound, so the finding stands and carries the flag.
    report = search(chembl, [DICLOFENAC])
    assert matches(report) == [("inchikey", "explicit_id")]
    assert report["summary"]["status"] == "FAIL"
    (activity,) = report["findings"][0]["activities"]
    assert activity["data_validity_comment"] == "Potential transcription error"
    assert "data_validity=Potential transcription error" in format_report_as_text(report)


def test_a_repeat_citation_is_a_finding_but_not_document_coverage(chembl):
    # Naproxen is measured in document 4 and cited again in document 1, the
    # citation flagged potential_duplicate. by_document answers "how much of
    # the test set does one paper hold?", and a citation is not that paper
    # measuring the compound, so only document 4 may appear. Negative
    # control: without the exclusion both documents are listed.
    report = search(chembl, [NAPROXEN])
    assert matches(report) == [("inchikey", "explicit_id")]
    assert [summary["document_chembl_id"] for summary in report["by_document"]] == [
        CYP3A4_DOCUMENT_4
    ]
    assert report["summary"]["total_documents"] == 1
    # The duplicate is kept as evidence, and marked.
    activities = report["findings"][0]["activities"]
    assert [a["potential_duplicate"] for a in activities] == [False, True]
    assert "potential_duplicate" in format_report_as_text(report)


def test_a_repeat_citation_does_not_inflate_a_shared_document(chembl):
    # The inflation the exclusion exists to stop: document 1 genuinely
    # measures caffeine, and only cites naproxen. It must be credited with
    # one test row, not two.
    report = search(chembl, [CAFFEINE, NAPROXEN])
    shares = {s["document_chembl_id"]: s["test_rows"] for s in report["by_document"]}
    assert shares == {CYP3A4_DOCUMENT: 1, CYP3A4_DOCUMENT_4: 1}


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


def test_the_report_lists_what_each_target_name_resolved_to(chembl):
    # Resolution goes through component_synonyms, where one component belongs
    # to several targets, so a gene symbol can reach a family as well as the
    # single protein. The report has to say which, or a reviewer cannot tell
    # a correct search from one matching every P450.
    report = search(chembl, [NICOTINE], target_chembl_ids=[], target_names=["CYP3A4"])
    assert report["resolved_targets"] == {"CYP3A4": [CYP3A4_TARGET]}

    widened = search(
        chembl,
        [NICOTINE],
        target_chembl_ids=[],
        target_names=["CYP3A4"],
        target_types=["SINGLE PROTEIN", "PROTEIN FAMILY"],
    )
    assert widened["resolved_targets"] == {"CYP3A4": sorted([CYP3A4_TARGET, P450_FAMILY_TARGET])}
    # The whole point is that a widened run is visibly different, so the two
    # must not report the same resolution.
    assert widened["resolved_targets"] != report["resolved_targets"]


def test_a_target_name_that_resolved_to_nothing_is_reported_as_such(chembl):
    # The other way the resolution misleads: nothing matched, so the search
    # could only ever pass. Silence would read as a clean result.
    report = search(chembl, [CAFFEINE], target_chembl_ids=[], target_names=["NOT_A_REAL_TARGET"])
    assert report["resolved_targets"] == {"NOT_A_REAL_TARGET": []}
    assert "nothing matched" in format_report_as_text(report)


def test_resolved_targets_are_rendered_for_review(chembl, capsys):
    report = search(chembl, [CAFFEINE], target_chembl_ids=[], target_names=["CYP3A4"])
    assert f"resolved CYP3A4          : ['{CYP3A4_TARGET}']" in format_report_as_text(report)
    print_public_search_summary(report, None)
    assert f"CYP3A4 -> {CYP3A4_TARGET}" in capsys.readouterr().out


def test_the_connectivity_lookup_is_served_by_the_inchikey_index(chembl):
    # The connectivity lookup is a range over standard_inchi_key precisely so
    # ChEMBL's index on that column can serve it, with the small TEMP table
    # driving the join. Negative control for that shape: a prefix test such
    # as substr(standard_inchi_key, 1, 14) = k.key cannot use the index, so
    # the plan scans compound_structures instead and this fails.
    conn = connect(chembl)
    try:
        _load_keys(conn, "cdc_keys_connectivity", ["AAAAAAAAAAAAAA"])
        plan = [str(row[-1]) for row in conn.execute("EXPLAIN QUERY PLAN " + _CONNECTIVITY_SQL)]
    finally:
        conn.close()
    assert any(step.startswith("SCAN k") for step in plan), plan
    searches = [step for step in plan if step.startswith("SEARCH cs")]
    assert searches, plan
    assert "idx_cs_inchikey" in searches[0], plan
    assert "standard_inchi_key>" in searches[0], plan


def test_the_connectivity_range_matches_exactly_one_connectivity_block(tmp_path, chembl):
    # The boundary of the range form, which no real InChIKey can exercise
    # because the checker computes them. A standard InChIKey is 14
    # connectivity characters then '-', so the range ['<block>-', '<block>.')
    # must take every stereo variant of the block and nothing else.
    path = tmp_path / "boundary.sqlite"
    path.write_bytes(chembl.read_bytes())
    writable = sqlite3.connect(path)
    keys = {
        101: "AAAAAAAAAAAAAA-BBBBBBBBFV-N",
        102: "AAAAAAAAAAAAAA-CCCCCCCCFV-S",  # same connectivity, other stereo
        103: "AAAAAAAAAAAAAB-BBBBBBBBFV-N",  # the next connectivity block
        104: "AAAAAAAAAAAAA-BBBBBBBBFV-N",  # a 13-character first block
        105: "AAAAAAAAAAAAAAX-BBBBBBBBFV-N",  # a 15-character first block
        106: "AAAAAAAAAAAAAA.BBBBBBBBFV-N",  # not '-' as the separator
        107: "AAAAAAAAAAAAAA",  # a bare block, with no key after it
        108: None,
    }
    for molregno, key in keys.items():
        writable.execute(
            "INSERT INTO compound_structures (molregno, standard_inchi_key) VALUES (?, ?)",
            (molregno, key),
        )
    writable.commit()
    writable.close()

    conn = connect(path)
    try:
        found = find_structures(conn, {"inchikey_connectivity": ["AAAAAAAAAAAAAA"]})
    finally:
        conn.close()
    # 105-107 are the rows the two forms disagree on: substr() would take
    # them, the range does not. None of them is a standard InChIKey.
    assert {match.compound["molregno"] for match in found} == {101, 102}


def test_config_rejects_empty_target_types():
    with pytest.raises(ConfigError, match="target_types"):
        parse_public_search(
            {"chembl_sqlite": "c.sqlite", "target_names": ["CYP3A4"], "target_types": []}
        )
