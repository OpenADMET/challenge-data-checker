import json

import pandas as pd
import pytest

from challenge_data_checker import check
from challenge_data_checker.api import check as check_from_module
from challenge_data_checker.cli import main
from challenge_data_checker.report_types import CheckReport, CheckSummary


def test_check_is_exported_from_package_root():
    assert check is check_from_module


def test_check_runs_single_pool_checks_only():
    report = check(
        pd.DataFrame({"SMILES": ["CCO", "CCO", "c1ccccc1"], "id": ["A", "B", "C"]}),
        identifier_columns=["id"],
        print_report=False,
    )
    assert report["mode"] == "check"
    assert set(report) == set(CheckReport.__annotations__)
    assert set(report["summary"]) == set(CheckSummary.__annotations__)
    assert "train_test_leakage" not in report
    assert {
        loc["pool"] for d in report["internal_duplicates"]["inchikey"] for loc in d["occurrences"]
    } == {"dataset"}


def test_duplicates_in_a_checked_dataset_only_warn():
    # Unlike a test set in an audit: a standalone dataset isn't known to be scored.
    report = check(pd.DataFrame({"SMILES": ["c1ccccc1CCO", "c1ccccc1CCO"]}), print_report=False)
    assert report["summary"]["status"] == "WARN"
    assert "duplicate_structure" in report["summary"]["flagged"]["by_type"]


def test_unparseable_smiles_fail_a_check():
    report = check(pd.DataFrame({"SMILES": ["c1ccccc1CCO", "C1CC"]}), print_report=False)
    assert report["summary"]["status"] == "FAIL"


def test_check_pools_several_sources(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.tsv"
    pd.DataFrame({"SMILES": ["c1ccccc1CCO"]}).to_csv(a, index=False)
    pd.DataFrame({"standardized_smiles": ["c1ccccc1CCO"]}).to_csv(b, sep="\t", index=False)
    report = check([str(a), str(b)], print_report=False)
    assert report["summary"]["total_rows"] == 2
    assert len(report["internal_duplicates"]["inchikey"]) == 1
    assert report["config"]["resolved_smiles_columns"] == {
        str(a): "SMILES",
        str(b): "standardized_smiles",
    }


@pytest.mark.parametrize(("filename", "marker"), [("r.json", None), ("r.txt", "check report")])
def test_check_writes_report(tmp_path, filename, marker):
    out = tmp_path / filename
    check(pd.DataFrame({"SMILES": ["CCO"]}), report_output=out, print_report=False)
    text = out.read_text()
    if marker is None:
        assert json.loads(text)["mode"] == "check"
    else:
        assert marker in text
        assert "Quality Flags" in text


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"SMILES": ["c1ccccc1CCO", "c1ccccc1CCO", "c1ccccc1CCN"], "id": ["A", "B", "C"]}
    ).to_csv(path, index=False)
    return path


def test_cli_check_with_flags(dataset, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = main(
        [
            "check",
            "--files",
            str(dataset),
            "--identifier-columns",
            "id",
            "--report-output",
            str(out),
        ]
    )
    assert code == 0  # duplicate only: WARN
    assert "check summary" in capsys.readouterr().out
    report = json.loads(out.read_text())
    assert report["summary"]["status"] == "WARN"
    assert report["config"]["identifier_columns"] == ["id"]


def test_cli_check_strict_fails_on_warnings(dataset):
    assert main(["check", "--files", str(dataset), "--strict"]) == 2


def test_cli_check_with_toml_config(dataset, tmp_path):
    config = tmp_path / "check.toml"
    config.write_text(f"""
[paths]
files = ["{dataset}"]

[columns]
smiles_column = "SMILES"
identifier_columns = ["id"]

[settings]
strip_salts = false
""")
    assert main(["check", "--config", str(config)]) == 0  # report_output is optional
    assert main(["check", "--config", str(config), "--strict"]) == 2


def test_cli_check_flags_map_to_settings(dataset, tmp_path):
    out = tmp_path / "r.json"
    main(
        [
            "check",
            "--files",
            str(dataset),
            "--no-strip-salts",
            "--tautomer-standardisation",
            "--identifier-case-insensitive",
            "--multi-component-min-heavy-atoms",
            "4",
            "--report-output",
            str(out),
        ]
    )
    cfg = json.loads(out.read_text())["config"]
    assert cfg["strip_salts"] is False
    assert cfg["tautomer_standardisation"] is True
    assert cfg["identifier_case_insensitive"] is True
    assert cfg["multi_component_min_heavy_atoms"] == 4


def test_cli_check_rejects_config_combined_with_data_flags(dataset, tmp_path, capsys):
    config = tmp_path / "check.toml"
    config.write_text(f'[paths]\nfiles = ["{dataset}"]\n[columns]\nsmiles_column = "SMILES"\n')
    assert main(["check", "--config", str(config), "--files", str(dataset)]) == 1
    assert "can't be combined" in capsys.readouterr().err


def test_cli_check_needs_files_or_config(capsys):
    assert main(["check"]) == 1
    assert "--config or --files" in capsys.readouterr().err


def test_cli_check_rejects_an_audit_config(tmp_path, capsys):
    config = tmp_path / "audit.toml"
    config.write_text(
        '[paths]\ntrain_files = ["a.csv"]\ntest_files = ["b.csv"]\nreport_output = "r.json"\n'
        '[columns]\nsmiles_column = "SMILES"\n'
    )
    assert main(["check", "--config", str(config)]) == 1
    assert "train/test audit config" in capsys.readouterr().err


def test_audit_rejects_a_check_config(tmp_path, capsys):
    config = tmp_path / "check.toml"
    config.write_text('[paths]\nfiles = ["a.csv"]\n[columns]\nsmiles_column = "SMILES"\n')
    assert main([str(config)]) == 1
    assert "check --config" in capsys.readouterr().err


def test_check_config_rejects_audit_only_settings(dataset, tmp_path, capsys):
    config = tmp_path / "check.toml"
    config.write_text(
        f'[paths]\nfiles = ["{dataset}"]\n[columns]\nsmiles_column = "SMILES"\n'
        "[settings]\nmax_train_test_similarity = 0.9\n"
    )
    assert main(["check", "--config", str(config)]) == 1
    assert "only apply to the train/test audit" in capsys.readouterr().err


def test_usage_errors_exit_1_not_2(capsys):
    # 2 is reserved for blocking findings.
    with pytest.raises(SystemExit) as exc:
        main(["check", "--no-such-flag"])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
