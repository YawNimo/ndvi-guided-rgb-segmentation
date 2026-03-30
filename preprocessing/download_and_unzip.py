#!/usr/bin/env python3

from utils.download import download_zips
from utils.unzip import extract_all_zips
from env_vars import MAX_DOWNLOADS

def download_and_unzip():
	download_zips(max_downloads=MAX_DOWNLOADS)
	extract_all_zips()


if __name__ == "__main__":
	download_and_unzip()