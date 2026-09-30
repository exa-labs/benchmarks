"""Download, verify and cache pinned upstream benchmark sources.

Two kinds of source are supported:

* a plain URL whose bytes are verified against a pinned SHA-256 and cached under
  ``~/.cache/benchmarks`` (override with ``BENCHMARKS_CACHE_DIR``);
* a file in a Hugging Face dataset repository at a pinned commit, fetched through
  ``huggingface_hub`` so it lands in the normal Hugging Face cache.

Nothing here writes benchmark content anywhere else: callers parse the cached
upstream files in memory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

CACHE_ENV = "BENCHMARKS_CACHE_DIR"


class SourceError(RuntimeError):
    """An upstream source could not be fetched, verified or parsed."""


def cache_dir() -> Path:
    """Return the directory that holds verified URL downloads."""
    override = os.environ.get(CACHE_ENV)
    return Path(override) if override else Path.home() / ".cache" / "benchmarks"


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fetch_verified(url: str, sha256: str, filename: str) -> bytes:
    """Return the bytes at ``url``, downloading once and verifying the pinned hash.

    A cached file whose hash no longer matches is treated as corrupt and fetched
    again; a fresh download that does not match raises ``SourceError`` and is not
    cached.
    """
    path = cache_dir() / filename
    if path.exists():
        data = path.read_bytes()
        if sha256_hex(data) == sha256:
            return data
    try:
        response = httpx.get(url, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise SourceError(f"failed to download {url}: {error}") from error
    data = response.content
    actual = sha256_hex(data)
    if actual != sha256:
        raise SourceError(f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
    Path(handle.name).replace(path)
    return data


def hf_file(
    repo_id: str,
    filename: str,
    revision: str,
    *,
    token: str | bool | None = None,
) -> Path:
    """Return the local path of one file in a Hugging Face dataset at a pinned commit."""
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="dataset",
                token=token,
            )
        )
    except (GatedRepoError, RepositoryNotFoundError):
        raise
    except Exception as error:
        raise SourceError(
            f"failed to fetch {filename} from {repo_id}@{revision}: {error}"
        ) from error


def hf_snapshot(repo_id: str, revision: str, patterns: list[str]) -> Path:
    """Return the local root of a pinned dataset snapshot restricted to ``patterns``."""
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                repo_type="dataset",
                allow_patterns=patterns,
            )
        )
    except Exception as error:
        raise SourceError(
            f"failed to fetch {patterns} from {repo_id}@{revision}: {error}"
        ) from error


def csv_rows(data: bytes, *, delimiter: str = ",") -> list[dict[str, str]]:
    """Parse CSV/TSV bytes into row dicts, keeping every value as the literal string."""
    csv.field_size_limit(2**31 - 1)
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def require_count(name: str, actual: int, expected: int) -> None:
    """Raise if a loaded source does not have the pinned number of rows."""
    if actual != expected:
        raise SourceError(f"{name}: expected {expected} rows at the pinned revision, got {actual}")
