import sys
import types

import pytest

from challenge_data_checker.remote import (
    RemoteSourceError,
    is_remote_source,
    parse_hf_hub_url,
    resolve_remote_source,
)


def test_is_remote_source_true_for_http_and_https():
    assert is_remote_source("http://example.com/file.csv")
    assert is_remote_source("https://example.com/file.csv")


def test_is_remote_source_false_for_local_path():
    assert not is_remote_source("data/file.csv")


def test_parse_hf_hub_url_resolve_form():
    url = "https://huggingface.co/datasets/org/dataset/resolve/main/data/file.csv"
    assert parse_hf_hub_url(url) == ("org/dataset", "main", "data/file.csv")


def test_parse_hf_hub_url_blob_form():
    url = "https://huggingface.co/datasets/org/dataset/blob/v1.0/file.parquet"
    assert parse_hf_hub_url(url) == ("org/dataset", "v1.0", "file.parquet")


def test_parse_hf_hub_url_non_matching_returns_none():
    assert parse_hf_hub_url("https://example.com/data/file.csv") is None
    assert parse_hf_hub_url("https://huggingface.co/org/model/resolve/main/f.csv") is None


def _install_fake_hf_hub(monkeypatch, download_fn, http_error_cls):
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = download_fn
    errors_module = types.ModuleType("huggingface_hub.errors")
    errors_module.HfHubHTTPError = http_error_cls
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)


class _FakeHfHubHTTPError(Exception):
    def __init__(self, response):
        super().__init__("hf hub error")
        self.response = response


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_resolve_remote_source_downloads_from_hf_hub(monkeypatch, tmp_path):
    local_file = tmp_path / "downloaded.csv"
    local_file.write_text("SMILES\nCCO\n")
    captured = {}

    def fake_download(*, repo_id, filename, repo_type, revision):
        captured.update(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
        return str(local_file)

    _install_fake_hf_hub(monkeypatch, fake_download, _FakeHfHubHTTPError)

    url = "https://huggingface.co/datasets/org/dataset/resolve/main/data/file.csv"
    assert resolve_remote_source(url) == local_file
    assert captured == {
        "repo_id": "org/dataset",
        "filename": "data/file.csv",
        "repo_type": "dataset",
        "revision": "main",
    }


def test_resolve_remote_source_hf_auth_error_is_actionable(monkeypatch):
    def fake_download(**kwargs):
        raise _FakeHfHubHTTPError(_FakeResponse(401))

    _install_fake_hf_hub(monkeypatch, fake_download, _FakeHfHubHTTPError)

    url = "https://huggingface.co/datasets/org/dataset/resolve/main/file.csv"
    with pytest.raises(RemoteSourceError, match="HF_TOKEN"):
        resolve_remote_source(url)


def test_resolve_remote_source_hf_hub_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    url = "https://huggingface.co/datasets/org/dataset/resolve/main/file.csv"
    with pytest.raises(RemoteSourceError, match="huggingface_hub"):
        resolve_remote_source(url)


def test_resolve_remote_source_downloads_generic_url(monkeypatch, tmp_path):
    import requests

    content = b"SMILES\nCCO\n"

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield content

    def fake_get(url, stream, timeout):
        assert stream is True
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    url = "https://raw.githubusercontent.com/org/repo/main/data/file.csv?token=abc"
    result = resolve_remote_source(url)
    assert result.suffix == ".csv"
    assert result.read_bytes() == content


def test_resolve_remote_source_generic_url_auth_error(monkeypatch):
    import requests

    def fake_get(url, stream, timeout):
        class FakeResponse:
            def raise_for_status(self):
                error = requests.HTTPError("403 Forbidden")
                error.response = _FakeResponse(403)
                raise error

        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    url = "https://raw.githubusercontent.com/org/repo/main/file.csv"
    with pytest.raises(RemoteSourceError, match="Access denied"):
        resolve_remote_source(url)


def test_resolve_remote_source_generic_url_not_found(monkeypatch):
    import requests

    def fake_get(url, stream, timeout):
        class FakeResponse:
            def raise_for_status(self):
                error = requests.HTTPError("404 Not Found")
                error.response = _FakeResponse(404)
                raise error

        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    url = "https://example.com/missing.csv"
    with pytest.raises(RemoteSourceError, match="404"):
        resolve_remote_source(url)


def test_resolve_remote_source_generic_url_network_error(monkeypatch):
    import requests

    def fake_get(url, stream, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "get", fake_get)

    url = "https://example.com/file.csv"
    with pytest.raises(RemoteSourceError, match="Failed to download"):
        resolve_remote_source(url)


def test_resolve_remote_source_requests_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    url = "https://example.com/file.csv"
    with pytest.raises(RemoteSourceError, match="requests"):
        resolve_remote_source(url)
