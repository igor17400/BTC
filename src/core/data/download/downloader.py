"""Generic dataset download and extraction with retry logic.

Handles SSL (macOS certifi), retries with exponential backoff,
download validation, and zip flattening.
"""

from __future__ import annotations

import logging
import shutil
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

from rich.progress import Progress

logger = logging.getLogger(__name__)


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context with certifi certs (macOS often lacks system certs)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_with_curl(url: str, dest: Path) -> None:
    """Download using curl (more reliable for large files / HuggingFace)."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "curl",
            "-L",
            "-C",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-o",
            str(dest),
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"curl failed: {result.stderr}")


def _download_with_urllib(
    url: str, dest: Path, progress: Progress | None, task_id
) -> None:
    """Download using urllib (works when curl is unavailable)."""
    ssl_ctx = _build_ssl_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))

    req = urllib.request.Request(url, headers={"User-Agent": "NewsReX/1.0"})
    with opener.open(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if progress and task_id is not None and total > 0:
            progress.update(task_id, total=total)
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress and task_id is not None:
                    progress.update(task_id, completed=downloaded)

    # Validate completeness
    actual_size = dest.stat().st_size
    if total > 0 and actual_size < total:
        raise RuntimeError(f"Incomplete: got {actual_size}, expected {total}")


def _has_curl() -> bool:
    """Check if curl is available on the system."""
    import subprocess

    try:
        subprocess.run(["curl", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_file_with_retries(
    url: str,
    dest: Path,
    progress: Progress | None = None,
    task_id=None,
    max_retries: int = 3,
) -> None:
    """Download a file with retry logic.

    Prefers curl (handles redirects, resume, and large files better).
    Falls back to urllib if curl is unavailable.

    Args:
        url: URL to download from.
        dest: Local path to save to.
        progress: Optional Rich progress bar.
        task_id: Optional progress task ID.
        max_retries: Number of retry attempts.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Prefer curl for reliability with large files
    if _has_curl():
        try:
            _download_with_curl(url, dest)
            return
        except Exception as exc:
            logger.warning(f"curl download failed: {exc}. Falling back to urllib.")

    # Fallback: urllib with retries
    for attempt in range(1, max_retries + 1):
        try:
            _download_with_urllib(url, dest, progress, task_id)
            return
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    f"Download attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                if progress and task_id is not None:
                    progress.update(task_id, completed=0)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts: {exc}"
                ) from exc


def extract_and_flatten(zip_path: Path, extract_path: Path) -> None:
    """Extract a zip file and flatten if it contains a single subdirectory.

    Handles the common pattern where a zip contains a single top-level
    directory (e.g., ``MINDsmall_train/``) and moves its contents up
    one level so files are directly in ``extract_path``.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    zip_path.unlink()

    # Flatten: e.g. train/MINDsmall_train/news.tsv -> train/news.tsv
    subdirs = [d for d in extract_path.iterdir() if d.is_dir()]
    files_at_root = [f for f in extract_path.iterdir() if f.is_file()]
    if len(subdirs) == 1 and not files_at_root:
        nested = subdirs[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(extract_path / item.name))
        nested.rmdir()


def has_data_files(dataset_path: Path) -> bool:
    """Check if actual data splits (train/valid/test with news.tsv) exist."""
    if not dataset_path.exists():
        return False
    for split_dir in ("train", "valid", "test"):
        split_path = dataset_path / split_dir
        if split_path.exists() and (split_path / "news.tsv").exists():
            return True
    return False


def download_dataset(
    dataset_path: Path,
    urls: dict[str, str],
    use_knowledge_graph: bool = False,
) -> None:
    """Download and extract a dataset if not already present.

    Args:
        dataset_path: Root directory for the dataset.
        urls: Mapping of split name -> URL (e.g. {"train": "...", "valid": "..."}).
        use_knowledge_graph: Whether to also download KG data.
    """
    if has_data_files(dataset_path):
        logger.info(f"Found existing dataset at {dataset_path}")
        return

    if not urls:
        logger.warning(
            f"No URLs provided for downloading dataset. "
            f"Assuming data exists at {dataset_path}"
        )
        return

    dataset_path.mkdir(parents=True, exist_ok=True)

    for split, url in urls.items():
        zip_path = dataset_path / f"{split}.zip"
        extract_path = dataset_path / split

        if not extract_path.exists():
            logger.info(f"Downloading {split} set from {url}")
            download_file_with_retries(url, zip_path)

            logger.info(f"Extracting {split} set...")
            extract_and_flatten(zip_path, extract_path)

            logger.info(f"Successfully processed {split} set")
        else:
            logger.info(f"Found existing {split} set at {extract_path}")

    if use_knowledge_graph:
        _download_knowledge_graph(dataset_path)


def _download_knowledge_graph(dataset_path: Path) -> None:
    """Download and extract knowledge graph data."""
    graph_extract_path = dataset_path / "download" / "wikidata-graph"

    if graph_extract_path.exists():
        logger.info("Found existing knowledge graph data")
        return

    graph_url = (
        "https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip"
    )
    graph_zip_path = dataset_path / "download" / "wikidata-graph.zip"
    graph_zip_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading knowledge graph from {graph_url}")
    download_file_with_retries(graph_url, graph_zip_path)
    extract_and_flatten(graph_zip_path, graph_extract_path)
    logger.info("Successfully downloaded knowledge graph data")
