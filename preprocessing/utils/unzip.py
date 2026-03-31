#!/usr/bin/env python3

"""ZIP extraction helpers for preprocessing source data."""

import sys
import zipfile
from pathlib import Path

try:
    from ..env_vars import INPUT_PATH, UNTILED_IMAGES_DIR
except ImportError:
    from env_vars import INPUT_PATH, UNTILED_IMAGES_DIR

ZIPS_DIR = INPUT_PATH / "zips"


def extract_all_zips():
    """Extract every ZIP file from ``input/zips`` into untiled imagery directory."""
    # Create output directory if it doesn't exist
    UNTILED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all zip files
    zip_files = sorted(ZIPS_DIR.glob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in {ZIPS_DIR}")
        return
    
    # Extract each zip file
    for zip_file in zip_files:
        print(f"Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(UNTILED_IMAGES_DIR)
        except zipfile.BadZipFile:
            print(f"Error: {zip_file.name} is not a valid zip file", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error extracting {zip_file.name}: {e}", file=sys.stderr)
            continue
    
    print(f"Done. Zips are in {ZIPS_DIR} and extracted images are in {UNTILED_IMAGES_DIR}")


if __name__ == "__main__":
    extract_all_zips()
