#!/usr/bin/env python3

"""Script entrypoint to download source ZIP files and extract them."""

from utils.download import download_zips
from utils.unzip import extract_all_zips
from env_vars import MAX_DOWNLOADS

def download_and_unzip():
	"""Download ZIP files using CSV URLs, then extract all downloaded archives."""
	download_zips(max_downloads=MAX_DOWNLOADS)
	extract_all_zips()


if __name__ == "__main__":
	download_and_unzip()