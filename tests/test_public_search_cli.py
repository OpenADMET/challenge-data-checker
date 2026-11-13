import json

import pandas as pd
import pytest
from chembl_fixture import CYP3A4_TARGET, build_fixture

from challenge_data_checker.api import search_public
from challenge_data_checker.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main
from challenge_data_checker.config import ConfigError, load_public_search_config

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
OCTANOL = "CCCCCCCCO"
# ChEMBL holds the racemate, so this only matches on connectivity.
S_IBUPROFEN = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"


@pytest.fixture(scope="module")
def chembl(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("chembl") / "chembl_fixture.sqlite")


@pytest.fixture
def test_set(tmp_path):
    path = tmp_path / "blinded_test.csv"
    pd.DataFrame({"SMILES": [CAFFEINE, OCTANOL], "compound_id": ["T1", "T2"]}).to_csv(
        path, index=False
    )
    return path


def write_config(tmp_path, test_set, chembl, report_filename="public.json", extra=""):
    content = f"""
[paths]
test_files = ["{test_set}"]
report_output = "{tmp_path / report_filename}"

[columns]
smiles_column = "SMILES"
identifier_columns = ["compound_id"]

[public_search]
chembl_sqlite = "{chembl}"
target_chembl_ids = ["{CYP3A4_TARGET}"]
standard_types = ["IC50"]
{extra}
"""
    path = tmp_path / "public.toml"
    path.write_text(content)
    return path


def test_cli_reports_blocking_findings_with_exit_code_two(tmp_path, test_set, chembl, capsys):
    config = write_config(tmp_path, test_set, chembl)
    assert main(["public-search", "--config", str(config)]) == EXIT_FINDINGS
    assert "public-source search summary" in capsys.readouterr().out

    report = json.loads((tmp_path / "public.json").read_text())
    assert report["mode"] == "public_search"
    assert report["summary"]["status"] == "FAIL"
    assert report["summary"]["total_test_rows"] == 2
    assert len(report["findings"]) == 1


def test_cli_passes_when_nothing_is_public(tmp_path, chembl, capsys):
    clean = tmp_path / "clean.csv"
    pd.DataFrame({"SMILES": [OCTANOL], "compound_id": ["T1"]}).to_csv(clean, index=False)
    config = write_config(tmp_path, clean, chembl)
    assert main(["public-search", "--config", str(config)]) == EXIT_OK
    capsys.readouterr()


def test_cli_strict_turns_a_flagged_finding_into_a_failure(tmp_path, test_set, chembl, capsys):
    # Resolving the target from a gene symbol is only ever flagged, so --strict
    # is what makes it stop a release.
    content = f"""
[paths]
test_files = ["{test_set}"]
report_output = "{tmp_path / "public.json"}"

[columns]
smiles_column = "SMILES"

[public_search]
chembl_sqlite = "{chembl}"
target_names = ["CYP3A4"]
standard_types = ["IC50"]
"""
    config = tmp_path / "flagged.toml"
    config.write_text(content)

    assert main(["public-search", "--config", str(config)]) == EXIT_OK
    assert main(["public-search", "--config", str(config), "--strict"]) == EXIT_FINDINGS
    capsys.readouterr()


def test_cli_accepts_flags_instead_of_a_config(tmp_path, test_set, chembl, capsys):
    code = main(
        [
            "public-search",
            "--test-files",
            str(test_set),
            "--chembl-sqlite",
            str(chembl),
            "--target-chembl-ids",
            CYP3A4_TARGET,
            "--report-output",
            str(tmp_path / "flags.json"),
        ]
    )
    assert code == EXIT_FINDINGS
    capsys.readouterr()
    assert json.loads((tmp_path / "flags.json").read_text())["summary"]["status"] == "FAIL"


def test_cli_refuses_a_config_combined_with_data_flags(tmp_path, test_set, chembl, capsys):
    config = write_config(tmp_path, test_set, chembl)
    code = main(["public-search", "--config", str(config), "--chembl-sqlite", str(chembl)])
    assert code == EXIT_ERROR
    assert "can't be combined" in capsys.readouterr().err


def test_cli_reports_a_missing_release_as_a_configuration_error(tmp_path, test_set, capsys):
    code = main(
        [
            "public-search",
            "--test-files",
            str(test_set),
            "--chembl-sqlite",
            str(tmp_path / "absent.sqlite"),
            "--target-chembl-ids",
            CYP3A4_TARGET,
        ]
    )
    assert code == EXIT_ERROR
    assert "not found" in capsys.readouterr().err


def test_cli_needs_an_endpoint_matcher(tmp_path, test_set, chembl, capsys):
    code = main(
        [
            "public-search",
            "--test-files",
            str(test_set),
            "--chembl-sqlite",
            str(chembl),
        ]
    )
    assert code == EXIT_ERROR
    assert "at least one of" in capsys.readouterr().err


def test_cli_writes_a_text_report(tmp_path, test_set, chembl, capsys):
    config = write_config(tmp_path, test_set, chembl, report_filename="public.txt")
    assert main(["public-search", "--config", str(config)]) == EXIT_FINDINGS
    capsys.readouterr()
    assert "public-source search report" in (tmp_path / "public.txt").read_text()


def test_config_file_rejects_a_training_pool(tmp_path, chembl):
    path = tmp_path / "audit_shaped.toml"
    path.write_text(f"""
[paths]
train_files = ["train.csv"]
test_files = ["test.csv"]

[columns]
smiles_column = "SMILES"

[public_search]
chembl_sqlite = "{chembl}"
target_names = ["CYP3A4"]
""")
    with pytest.raises(ConfigError, match="names a training pool"):
        load_public_search_config(path)


def test_config_file_requires_the_public_search_section(tmp_path, test_set):
    path = tmp_path / "no_section.toml"
    path.write_text(f"""
[paths]
test_files = ["{test_set}"]

[columns]
smiles_column = "SMILES"
""")
    with pytest.raises(ConfigError, match=r"\[public_search\]"):
        load_public_search_config(path)


def test_config_file_rejects_audit_only_settings(tmp_path, test_set, chembl):
    path = tmp_path / "audit_setting.toml"
    path.write_text(f"""
[paths]
test_files = ["{test_set}"]

[columns]
smiles_column = "SMILES"

[settings]
max_train_test_similarity = 0.9

[public_search]
chembl_sqlite = "{chembl}"
target_names = ["CYP3A4"]
""")
    with pytest.raises(ConfigError, match="only apply to the train/test audit"):
        load_public_search_config(path)


def test_python_api_matches_the_cli(tmp_path, chembl, capsys):
    report = search_public(
        pd.DataFrame({"SMILES": [CAFFEINE, OCTANOL]}),
        chembl_sqlite=chembl,
        target_chembl_ids=[CYP3A4_TARGET],
        print_report=False,
    )
    capsys.readouterr()
    assert report["summary"]["status"] == "FAIL"
    assert report["summary"]["total_test_rows"] == 2
    assert len(report["findings"]) == 1


def test_cli_and_api_match_connectivity_by_default(tmp_path, chembl, capsys):
    # The flag's absence must leave the default on, not coerce it off, in
    # both front ends. Negative control for the default: flipping it back
    # reports match_connectivity false here.
    stereo = tmp_path / "stereo.csv"
    pd.DataFrame({"SMILES": [S_IBUPROFEN], "compound_id": ["T1"]}).to_csv(stereo, index=False)
    code = main(
        [
            "public-search",
            "--test-files",
            str(stereo),
            "--chembl-sqlite",
            str(chembl),
            "--target-chembl-ids",
            CYP3A4_TARGET,
            "--report-output",
            str(tmp_path / "default.json"),
        ]
    )
    capsys.readouterr()
    # Flagged, not blocking: the match is on connectivity alone.
    assert code == EXIT_OK
    report = json.loads((tmp_path / "default.json").read_text())
    assert report["config"]["match_connectivity"] is True
    assert report["summary"]["status"] == "WARN"

    api_report = search_public(
        pd.DataFrame({"SMILES": [S_IBUPROFEN]}),
        chembl_sqlite=chembl,
        target_chembl_ids=[CYP3A4_TARGET],
        print_report=False,
    )
    capsys.readouterr()
    assert api_report["config"]["match_connectivity"] is True
    assert api_report["summary"]["status"] == "WARN"


def test_cli_can_turn_connectivity_matching_off(tmp_path, chembl, capsys):
    stereo = tmp_path / "stereo.csv"
    pd.DataFrame({"SMILES": [S_IBUPROFEN], "compound_id": ["T1"]}).to_csv(stereo, index=False)
    code = main(
        [
            "public-search",
            "--test-files",
            str(stereo),
            "--chembl-sqlite",
            str(chembl),
            "--target-chembl-ids",
            CYP3A4_TARGET,
            "--no-match-connectivity",
            "--report-output",
            str(tmp_path / "off.json"),
        ]
    )
    capsys.readouterr()
    assert code == EXIT_OK
    report = json.loads((tmp_path / "off.json").read_text())
    assert report["config"]["match_connectivity"] is False
    assert report["summary"]["status"] == "PASS"


def test_python_api_needs_an_endpoint_matcher(chembl):
    with pytest.raises(ConfigError, match="at least one of"):
        search_public(
            pd.DataFrame({"SMILES": [CAFFEINE]}),
            chembl_sqlite=chembl,
            print_report=False,
        )
