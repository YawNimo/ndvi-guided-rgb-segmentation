"""Download utilities for retrieving imagery ZIP archives from CSV URLs."""

import csv
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

try:
    from ..env_vars import INPUT_PATH, CSV_COLUMN_NAME
except ImportError:
    from env_vars import INPUT_PATH, CSV_COLUMN_NAME

# Download settings

CHUNK_SIZE_BYTES = 1024 * 1024
BYTES_PER_MB = 1024 * 1024
PROGRESS_BAR_WIDTH = 30

# Paths
CSV_PATH = INPUT_PATH / "2016, 2019, 2023 Imagery & Elevation Tiles (Proximity).csv"
ZIPS_DIR = INPUT_PATH / "zips"


def _iter_zip_urls(csv_path: Path, column_name: str):
    """Yield valid ZIP URLs from a configured CSV column."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column_name not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Column '{column_name}' not found in CSV. Available headers: {available}"
            )

        for row in reader:
            raw_url = (row.get(column_name) or "").strip()
            if not raw_url:
                continue
            if not raw_url.lower().startswith(("http://", "https://")):
                continue
            if not urlparse(raw_url).path.lower().endswith(".zip"):
                continue
            yield raw_url


def _filename_from_url(url: str) -> str:
    """Extract a destination filename from a URL path."""
    path = urlparse(url).path
    name = Path(path).name
    return name or "download.zip"


def _print_progress(file_label: str, downloaded_bytes: int, total_bytes: int):
    """Print terminal progress status for a single download."""
    downloaded_mb = downloaded_bytes / BYTES_PER_MB

    if total_bytes > 0:
        progress = min(downloaded_bytes / total_bytes, 1.0)
        total_mb = total_bytes / BYTES_PER_MB
        filled = int(PROGRESS_BAR_WIDTH * progress)
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        print(
            f"\r{file_label} [{bar}] {progress * 100:6.2f}% {downloaded_mb:8.2f}/{total_mb:8.2f} MB",
            end="",
            flush=True,
        )
    else:
        print(
            f"\r{file_label} Downloaded: {downloaded_mb:8.2f} MB",
            end="",
            flush=True,
        )


def _download_file(url: str, destination: Path, file_label: str):
    """Download one URL to a destination path with progress output."""
    with urlopen(url) as response, destination.open("wb") as out_file:
        content_length = response.headers.get("Content-Length")
        total_bytes = int(content_length) if content_length and content_length.isdigit() else 0
        downloaded_bytes = 0

        while True:
            chunk = response.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break

            out_file.write(chunk)
            downloaded_bytes += len(chunk)
            _print_progress(file_label, downloaded_bytes, total_bytes)

    print()


def download_zips(max_downloads=3):
    """Download ZIP files listed in the preprocessing CSV.

    Args:
        max_downloads (int): Maximum number of unique ZIP URLs to download.

    Returns:
        None: Files are written under ``input/zips`` as a side effect.
    """
    ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    urls = list(dict.fromkeys(_iter_zip_urls(CSV_PATH, CSV_COLUMN_NAME)))
    print(f"Found {len(urls)} URL(s) in '{CSV_COLUMN_NAME}'.")
    urls = urls[:max_downloads]
    total = len(urls)
    print(f"Processing up to {len(urls)} URL(s) after applying the limit of {max_downloads}.")

    for index, url in enumerate(urls, start=1):
        filename = _filename_from_url(url)
        destination = ZIPS_DIR / filename

        if destination.exists():
            print(f"[{index}/{total}] Skipping existing file: {destination.name}")
            continue

        print(f"[{index}/{total}] Downloading {destination.name}")
        file_label = f"[{index}/{total}] {destination.name}"
        _download_file(url, destination, file_label)

        # for debugging:
        # if index >= 2:
        #     print(f"Stopping after {index+1} downloads for testing purposes.")
        #     break

    print(f"Done. ZIP files are in: {ZIPS_DIR.resolve()}")

if __name__ == "__main__":
    download_zips()