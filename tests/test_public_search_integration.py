"""Opt-in integration tests against a real ChEMBL SQLite release.

Skipped unless ``CDC_CHEMBL_SQLITE`` points at a downloaded release, so the
default test run stays offline and fast:

    CDC_CHEMBL_SQLITE=/data/chembl_35/chembl_35_sqlite/chembl_35.db pytest \\
        tests/test_public_search_integration.py -v

The assertions are deliberately chosen to hold across releases: the release
names itself, a very well known compound is present, a gene symbol resolves
to its target, and a target that does not exist produces no findings.
"""

import os

import pandas as pd
import pytest

from challenge_data_checker.api import search_public
from challenge_data_checker.chembl import connect, read_source, resolve_target_names

CHEMBL_SQLITE = os.environ.get("CDC_CHEMBL_SQLITE")

pytestmark = pytest.mark.skipif(
    not CHEMBL_SQLITE,
    reason="set CDC_CHEMBL_SQLITE to a local ChEMBL SQLite release to run these",
)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
# Well-studied CYP3A4 inhibitors. Asserting that at least one of them has
# CYP3A4 data, rather than a particular one, keeps this stable across
# releases and curation changes.
CYP3A4_INHIBITORS = {
    "ketoconazole": "CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",
    "itraconazole": (
        "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)"
        "(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O"
    ),
    "ritonavir": (
        "CC(C)c1nc(CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@@H](Cc2ccccc2)"
        "C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)cs1"
    ),
}
CYP3A4_TARGET = "CHEMBL340"
# A target ID no release has, used as the endpoint negative control.
NONEXISTENT_TARGET = "CHEMBL0000000"


@pytest.fixture(scope="module")
def chembl():
    return CHEMBL_SQLITE


def test_release_identifies_itself(chembl):
    conn = connect(chembl)
    try:
        source = read_source(conn, chembl)
    finally:
        conn.close()
    # ChEMBL's schema documents version.name only as "Name of release
    # version", with no promised format, so this asserts that the release
    # names itself rather than guessing at the spelling.
    assert source["release"], "a real release should name itself in version.name"


def test_gene_symbol_resolves_to_the_cyp3a4_target(chembl):
    conn = connect(chembl)
    try:
        resolved = resolve_target_names(conn, ["CYP3A4"])
    finally:
        conn.close()
    assert CYP3A4_TARGET in resolved["CYP3A4"]


def test_a_well_known_compound_is_found_in_the_release(chembl):
    report = search_public(
        pd.DataFrame({"SMILES": [CAFFEINE]}),
        chembl_sqlite=chembl,
        target_chembl_ids=[CYP3A4_TARGET],
        print_report=False,
    )
    assert report["summary"]["total_found_in_source"] == 1


def test_known_cyp3a4_inhibitors_produce_a_blocking_finding(chembl):
    report = search_public(
        pd.DataFrame({"SMILES": list(CYP3A4_INHIBITORS.values())}),
        chembl_sqlite=chembl,
        target_chembl_ids=[CYP3A4_TARGET],
        print_report=False,
    )
    assert report["summary"]["status"] == "FAIL", (
        "at least one of ketoconazole, itraconazole or ritonavir should have "
        f"CYP3A4 data in {report['source']['release']}"
    )
    assert all(finding["match_type"] == "explicit_id" for finding in report["findings"])


def test_presence_without_a_matching_target_produces_no_findings(chembl):
    # The endpoint negative control on real data: the compound is certainly in
    # ChEMBL, but no activity can match a target that does not exist, so
    # presence must not turn into a finding.
    report = search_public(
        pd.DataFrame({"SMILES": [CAFFEINE]}),
        chembl_sqlite=chembl,
        target_chembl_ids=[NONEXISTENT_TARGET],
        print_report=False,
    )
    assert report["summary"]["total_found_in_source"] == 1
    assert report["findings"] == []
    assert report["summary"]["status"] == "PASS"


def test_connectivity_matching_finds_at_least_as_much_as_exact(chembl):
    compounds = pd.DataFrame({"SMILES": list(CYP3A4_INHIBITORS.values())})
    exact = search_public(
        compounds,
        chembl_sqlite=chembl,
        target_chembl_ids=[CYP3A4_TARGET],
        match_connectivity=False,
        print_report=False,
    )
    connectivity = search_public(
        compounds,
        chembl_sqlite=chembl,
        target_chembl_ids=[CYP3A4_TARGET],
        match_connectivity=True,
        print_report=False,
    )
    assert (
        connectivity["summary"]["total_found_in_source"]
        >= exact["summary"]["total_found_in_source"]
    )
