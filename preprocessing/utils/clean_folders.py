#!/usr/bin/env python3

import shutil
import sys

try:
    from ..env_vars import INPUT_PATH, UNTILED_IMAGES_DIR
except ImportError:
    from env_vars import INPUT_PATH, UNTILED_IMAGES_DIR


ZIPS_DIR = INPUT_PATH / "zips"


def clean_non_tif():
    if not UNTILED_IMAGES_DIR.exists():
        print(f"Skipping missing folder: {UNTILED_IMAGES_DIR}")
        return

    for file in UNTILED_IMAGES_DIR.iterdir():
        if file.is_file() and file.suffix.lower() != ".tif":
            print(f"Removing non-TIF file: {file.name}")
            try:
                file.unlink()
            except Exception as e:
                print(f"Error removing {file.name}: {e}", file=sys.stderr)


def remove_zip_and_untiled_dirs():
    for folder in (ZIPS_DIR, UNTILED_IMAGES_DIR):
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
