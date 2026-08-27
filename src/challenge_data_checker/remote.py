"""Fetching train/test data sources given as a URL (HuggingFace Hub or plain HTTP(S))."""

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

HF_HUB_URL_RE = re.compile(
    r"^https?://huggingface\.co/datasets/(?P<repo_id>[^/]+/[^/]+)/"
    r"(?:resolve|blob)/(?P<revision>[^/]+)/(?P<path>.+)$"
)


class RemoteSourceError(ValueError):
    """Raised when a remote (URL) data source can't be downloaded."""


def is_remote_source(source: str) -> bool:
    """Check whether a data source string is a URL rather than a local path.

    Args:
        source: A train/test data source string, as given in a config or
            passed to the Python API.

    Returns:
        ``True`` if ``source`` starts with ``http://`` or ``https://``.

    """
    return source.startswith(("http://", "https://"))


def parse_hf_hub_url(url: str) -> tuple[str, str, str] | None:
    """Parse a HuggingFace Hub dataset-file URL into its components.

    Matches URLs of the form
    ``https://huggingface.co/datasets/<namespace>/<name>/resolve/<revision>/<path>``
    (or the equivalent ``/blob/`` form used by the "Files" tab in the Hub UI).

    Args:
        url: The URL to parse.

    Returns:
        A ``(repo_id, revision, path)`` tuple, or ``None`` if ``url`` isn't a
        recognised HuggingFace Hub dataset-file URL.

    """
    match = HF_HUB_URL_RE.match(url)
    if match is None:
        return None
    return match.group("repo_id"), match.group("revision"), match.group("path")


def _download_from_hf_hub(repo_id: str, revision: str, path: str, url: str) -> Path:
    """Download a file from HuggingFace Hub via ``huggingface_hub``.

    Args:
        repo_id: The dataset repo id, e.g. ``"namespace/name"``.
        revision: The branch, tag, or commit to download from.
        path: The file's path within the repo.
        url: The original URL, used only in error messages.

    Returns:
        The local path of the downloaded (and cached) file.

    Raises:
        RemoteSourceError: If ``huggingface_hub`` isn't installed, the file
            or repo doesn't exist, or access is denied.

    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise RemoteSourceError(
            f"Cannot download {url}: the 'huggingface_hub' package is required to load "
            "data from HuggingFace Hub URLs. Install it with `pip install -e '.[remote]'`."
        ) from exc

    try:
        return Path(
            hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset", revision=revision)
        )
    except HfHubHTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in (401, 403):
            raise RemoteSourceError(
                f"Access denied fetching {url} from HuggingFace Hub. If this is a "
                "private/gated dataset, set the HF_TOKEN environment variable or run "
                "`huggingface-cli login`."
            ) from exc
        raise RemoteSourceError(f"Failed to download {url} from HuggingFace Hub: {exc}") from exc


def _download_generic_url(url: str) -> Path:
    """Download a plain HTTP(S) URL to a local temporary file.

    Args:
        url: The URL to download.

    Returns:
        The path of a local temporary file holding the downloaded content,
        with the same file extension as the URL's path (so ``.csv``/
        ``.parquet`` detection downstream still works even with a query
        string, e.g. a GitHub raw URL with a ``?token=...`` suffix).

    Raises:
        RemoteSourceError: If ``requests`` isn't installed, the request
            fails (HTTP error status or network error).

    """
    try:
        import requests
    except ImportError as exc:
        raise RemoteSourceError(
            f"Cannot download {url}: the 'requests' package is required to load data "
            "from a URL. Install it with `pip install -e '.[remote]'`."
        ) from exc

    suffix = Path(urlparse(url).path).suffix

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in (401, 403):
            raise RemoteSourceError(
                f"Access denied fetching {url} (HTTP {status_code}). If this is a "
                "private repository, make sure the URL includes a valid access token."
            ) from exc
        if status_code == 404:
            raise RemoteSourceError(f"URL not found: {url} (HTTP 404). Check the URL.") from exc
        raise RemoteSourceError(f"Failed to download {url}: {exc}") from exc
    except requests.RequestException as exc:
        raise RemoteSourceError(f"Failed to download {url}: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        for chunk in response.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
        return Path(fh.name)


def resolve_remote_source(url: str) -> Path:
    """Download a train/test data source URL to a local file.

    HuggingFace Hub dataset-file URLs are downloaded via ``huggingface_hub``
    (which handles authentication and caches by repo/revision); any other
    URL is downloaded directly with a plain HTTP GET, to a fresh temporary
    file every time.

    Args:
        url: The URL to download.

    Returns:
        The local path of the downloaded file.

    """
    hf_match = parse_hf_hub_url(url)
    if hf_match is not None:
        repo_id, revision, path = hf_match
        return _download_from_hf_hub(repo_id, revision, path, url)
    return _download_generic_url(url)
