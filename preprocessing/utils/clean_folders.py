#!/usr/bin/env python3

"""Folder cleanup utilities for preprocessing artifacts."""

import shutil
import sys

try:
    from ..env_vars import BLURRED_IMAGES_DIR, INPUT_PATH, UNTILED_IMAGES_DIR
except ImportError:
    from env_vars import BLURRED_IMAGES_DIR, INPUT_PATH, UNTILED_IMAGES_DIR


ZIPS_DIR = INPUT_PATH / "zips"


def clean_non_tif():
    """Remove non-TIFF files from untiled and blurred image directories."""
    for folder in (UNTILED_IMAGES_DIR, BLURRED_IMAGES_DIR):
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() != ".tif":
                print(f"Removing non-TIF file: {file.name}")
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error removing {file.name}: {e}", file=sys.stderr)


def remove_zip_and_untiled_dirs():
    """Remove ZIP and intermediate source directories created during preprocessing."""
    for folder in (ZIPS_DIR, UNTILED_IMAGES_DIR, BLURRED_IMAGES_DIR):
        if folder.exists():
            print(f"Removing folder: {folder}")
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Error removing {folder}: {e}", file=sys.stderr)
        else:
            print(f"Skipping missing folder: {folder}")


if __name__ == "__main__":
    clean_non_tif()
    remove_zip_and_untiled_dirs()
