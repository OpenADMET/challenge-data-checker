import logging

import pandas as pd
import pytest

from challenge_data_checker.io_utils import (
    RESOLVED_SMILES_COL,
    ROW_INDEX_COL,
    SMILES_COLUMN_ALIASES,
    SOURCE_FILE_COL,
    ColumnResolutionError,
    check_identifier_columns,
    load_pool,
    load_table,
    resolve_smiles_column,
)


def test_resolve_smiles_column_exact_match():
    assert resolve_smiles_column(["SMILES", "value"], "SMILES") == "SMILES"


def test_resolve_smiles_column_case_insensitive():
    assert resolve_smiles_column(["smiles", "value"], "SMILES") == "smiles"


def test_resolve_smiles_column_alias_fallback():
    assert resolve_smiles_column(["structure", "value"], "SMILES") == "structure"


def test_resolve_smiles_column_alias_fallback_warns(caplog):
    with caplog.at_level(logging.WARNING):
        resolved = resolve_smiles_column(["structure", "value"], "SMILES", "file.csv")
    assert resolved == "structure"
    assert "SMILES" in caplog.text
    assert "structure" in caplog.text
    assert "file.csv" in caplog.text


def test_resolve_smiles_column_case_insensitive_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_smiles_column(["smiles", "value"], "SMILES", "file.csv")
    assert caplog.text == ""


def test_resolve_smiles_column_no_match_raises():
    with pytest.raises(ColumnResolutionError, match="Could not find"):
        resolve_smiles_column(["foo", "bar"], "SMILES")


def test_resolve_smiles_column_prefers_earlier_alias(caplog):
    with caplog.at_level(logging.WARNING):
        resolved = resolve_smiles_column(["structure", "smi"], "SMILES", "file.csv")
    assert resolved == "smi"
    assert "preferred over ['structure']" in caplog.text


def test_resolve_smiles_column_prefers_smiles_over_standardized_smiles():
    columns = ["standardized_smiles", "Smiles", "value"]
    assert resolve_smiles_column(columns, "compound_smiles") == "Smiles"


@pytest.mark.parametrize(
    "alias",
    ["standardized_smiles", "standardised_smiles", "std_smiles", "isomeric_smiles", "CXSMILES"],
)
def test_resolve_smiles_column_new_aliases(alias):
    assert resolve_smiles_column([alias, "value"], "SMILES") == alias


def test_resolve_smiles_column_alias_differing_only_in_case_raises():
    with pytest.raises(ColumnResolutionError, match="match the alias 'smi'"):
        resolve_smiles_column(["smi", "SMI"], "SMILES")


def test_smiles_column_aliases_are_lowercase_and_unique():
    assert all(alias == alias.lower() for alias in SMILES_COLUMN_ALIASES)
    assert len(set(SMILES_COLUMN_ALIASES)) == len(SMILES_COLUMN_ALIASES)


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

    pooled, identifier_cols, resolved_smiles_cols = load_pool(
        [str(file_a), str(file_b)], "SMILES", ["compound_id"]
    )

    assert list(pooled[RESOLVED_SMILES_COL]) == ["CCO", "CCN"]
    assert identifier_cols == ["compound_id"]
    assert list(pooled[SOURCE_FILE_COL]) == [str(file_a), str(file_b)]
    assert list(pooled[ROW_INDEX_COL]) == [0, 0]
    assert resolved_smiles_cols == {str(file_a): "SMILES", str(file_b): "structure"}


def test_load_pool_warns_on_alias_fallback(tmp_path, caplog):
    file_a = tmp_path / "a.csv"
    pd.DataFrame({"canonical_smiles": ["CCO"]}).to_csv(file_a, index=False)

    with caplog.at_level(logging.WARNING):
        load_pool([str(file_a)], "SMILES", [])

    assert "SMILES" in caplog.text
    assert "canonical_smiles" in caplog.text
    assert str(file_a) in caplog.text


def test_load_pool_resolves_remote_url_source(tmp_path, monkeypatch):
    downloaded_file = tmp_path / "downloaded.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(downloaded_file, index=False)

    url = "https://huggingface.co/datasets/org/dataset/resolve/main/test.csv"
    monkeypatch.setattr(
        "challenge_data_checker.remote.resolve_remote_source",
        lambda source: downloaded_file,
    )

    pooled, _, resolved_smiles_cols = load_pool([url], "SMILES", [])

    assert list(pooled[RESOLVED_SMILES_COL]) == ["CCO"]
    assert list(pooled[SOURCE_FILE_COL]) == [url]
    assert resolved_smiles_cols == {url: "SMILES"}
    # HF Hub downloads live in huggingface_hub's cache; the checker must not delete them.
    assert downloaded_file.exists()


def test_load_pool_reads_remote_tsv_and_deletes_temp_file(tmp_path, monkeypatch):
    import requests

    content = b"standardized_smiles\tid\nCCO\tOCNT-1\n"

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield content

    monkeypatch.setattr(requests, "get", lambda url, stream, timeout: FakeResponse())
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    url = "https://example.com/release/inhibition.tsv"
    pooled, _, resolved = load_pool([url], "SMILES", ["id"])

    assert list(pooled[RESOLVED_SMILES_COL]) == ["CCO"]
    assert list(pooled["id"]) == ["OCNT-1"]
    assert resolved == {url: "standardized_smiles"}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["data.tsv", "data.TSV", "data.tsv.gz"])
def test_load_table_reads_tab_separated(tmp_path, filename):
    path = tmp_path / filename
    pd.DataFrame({"SMILES": ["CCO", "CCN"], "id": ["A", "B"]}).to_csv(path, sep="\t", index=False)
    df = load_table(path)
    assert list(df.columns) == ["SMILES", "id"]
    assert list(df["SMILES"]) == ["CCO", "CCN"]


def test_load_table_reads_gzipped_csv(tmp_path):
    path = tmp_path / "data.csv.gz"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(path, index=False)
    assert list(load_table(path)["SMILES"]) == ["CCO"]


def test_load_table_unsupported_format_lists_supported_ones(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"'\.xlsx'.*\.tsv\.gz"):
        load_table(path)


def test_load_pool_missing_identifier_in_one_file_does_not_crash(tmp_path, caplog):
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    pd.DataFrame({"SMILES": ["CCO"], "compound_id": ["C1"]}).to_csv(file_a, index=False)
    pd.DataFrame({"SMILES": ["CCN"]}).to_csv(file_b, index=False)

    with caplog.at_level(logging.WARNING):
        pooled, identifier_cols, _ = load_pool(
            [str(file_a), str(file_b)], "SMILES", ["compound_id"]
        )

    assert identifier_cols == ["compound_id"]
    assert pooled["compound_id"].isna().sum() == 1
