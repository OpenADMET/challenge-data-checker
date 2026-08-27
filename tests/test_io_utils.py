import logging

import pandas as pd
import pytest

from challenge_data_checker.io_utils import (
    RESOLVED_SMILES_COL,
    ROW_INDEX_COL,
    SOURCE_FILE_COL,
    ColumnResolutionError,
    check_identifier_columns,
    load_pool,
    resolve_smiles_column,
)


def test_resolve_smiles_column_exact_match():
    assert resolve_smiles_column(["SMILES", "value"], "SMILES") == "SMILES"


def test_resolve_smiles_column_case_insensitive():
    assert resolve_smiles_column(["smiles", "value"], "SMILES") == "smiles"


def test_resolve_smiles_column_alias_fallback():
    assert resolve_smiles_column(["structure", "value"], "SMILES") == "structure"


def test_resolve_smiles_column_no_match_raises():
    with pytest.raises(ColumnResolutionError, match="Could not find"):
        resolve_smiles_column(["foo", "bar"], "SMILES")


def test_resolve_smiles_column_ambiguous_alias_raises():
    with pytest.raises(ColumnResolutionError, match="multiple candidate"):
        resolve_smiles_column(["smi", "structure"], "SMILES")


def test_check_identifier_columns_warns_on_missing(caplog):
    with caplog.at_level(logging.WARNING):
        present = check_identifier_columns(
            ["compound_id", "value"], ["compound_id", "external_reg_no"], "file.csv"
        )
    assert present == ["compound_id"]
    assert "external_reg_no" in caplog.text
    assert "file.csv" in caplog.text


def test_load_pool_renames_smiles_and_tags_provenance(tmp_path):
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    pd.DataFrame({"SMILES": ["CCO"], "compound_id": ["C1"]}).to_csv(file_a, index=False)
    pd.DataFrame({"structure": ["CCN"], "compound_id": ["C2"]}).to_csv(file_b, index=False)

    pooled, identifier_cols = load_pool([str(file_a), str(file_b)], "SMILES", ["compound_id"])

    assert list(pooled[RESOLVED_SMILES_COL]) == ["CCO", "CCN"]
    assert identifier_cols == ["compound_id"]
    assert list(pooled[SOURCE_FILE_COL]) == [str(file_a), str(file_b)]
    assert list(pooled[ROW_INDEX_COL]) == [0, 0]


def test_load_pool_resolves_remote_url_source(tmp_path, monkeypatch):
    downloaded_file = tmp_path / "downloaded.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(downloaded_file, index=False)

    url = "https://huggingface.co/datasets/org/dataset/resolve/main/test.csv"
    monkeypatch.setattr(
        "challenge_data_checker.io_utils.resolve_remote_source",
        lambda source: downloaded_file,
    )

    pooled, _ = load_pool([url], "SMILES", [])

    assert list(pooled[RESOLVED_SMILES_COL]) == ["CCO"]
    assert list(pooled[SOURCE_FILE_COL]) == [url]


def test_load_pool_missing_identifier_in_one_file_does_not_crash(tmp_path, caplog):
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    pd.DataFrame({"SMILES": ["CCO"], "compound_id": ["C1"]}).to_csv(file_a, index=False)
    pd.DataFrame({"SMILES": ["CCN"]}).to_csv(file_b, index=False)

    with caplog.at_level(logging.WARNING):
        pooled, identifier_cols = load_pool([str(file_a), str(file_b)], "SMILES", ["compound_id"])

    assert identifier_cols == ["compound_id"]
    assert pooled["compound_id"].isna().sum() == 1
