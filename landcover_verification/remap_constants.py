"""Shared landcover label mapping constants for verification workflows."""

from __future__ import annotations

LANDCOVER_CLASS_MAPPING: dict[int, int] = {
    0: 2,
    1: 1,
    2: 3,
    3: 0,
    4: 2,
}

EXPECTED_REMAP_OUTPUT_LABELS: set[int] = set(LANDCOVER_CLASS_MAPPING.values())
