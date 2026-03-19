#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAMES_FILE="$SCRIPT_DIR/imagenames.txt"
ZIPS_DIR="$SCRIPT_DIR/input/zips"
IMAGES_DIR="$SCRIPT_DIR/input/images"
BASE_URL="https://cteco.uconn.edu/download/aerial/2019/tiles/tif"

if [[ ! -f "$NAMES_FILE" ]]; then
	echo "Missing input file: $NAMES_FILE" >&2
	exit 1
fi

for cmd in curl unzip; do
	if ! command -v "$cmd" >/dev/null 2>&1; then
		echo "Required command not found: $cmd" >&2
		exit 1
	fi
done

mkdir -p "$ZIPS_DIR" "$IMAGES_DIR"

declare -A seen_names=()

while IFS= read -r raw_name || [[ -n "$raw_name" ]]; do
	# Remove CRLF endings and surrounding whitespace.
	name="${raw_name//$'\r'/}"
	name="${name#"${name%%[![:space:]]*}"}"
	name="${name%"${name##*[![:space:]]}"}"

	if [[ -z "$name" ]]; then
		continue
	fi

	if [[ -n "${seen_names[$name]:-}" ]]; then
		continue
	fi
	seen_names["$name"]=1

	zip_path="$ZIPS_DIR/${name}.zip"
	url="$BASE_URL/${name}.zip"

	if [[ -f "$zip_path" ]]; then
		echo "Skipping existing zip: $zip_path"
		continue
	fi

	echo "Downloading: $url"
	curl -fL --retry 3 --retry-delay 2 --output "$zip_path" "$url"
done < "$NAMES_FILE"

shopt -s nullglob
zip_files=("$ZIPS_DIR"/*.zip)

if (( ${#zip_files[@]} == 0 )); then
	echo "No zip files found in $ZIPS_DIR"
	exit 0
fi

for zip_file in "${zip_files[@]}"; do
	echo "Extracting: $(basename "$zip_file")"
	unzip -o -q "$zip_file" -d "$IMAGES_DIR"
done

echo "Done. Zips are in $ZIPS_DIR and extracted images are in $IMAGES_DIR"
