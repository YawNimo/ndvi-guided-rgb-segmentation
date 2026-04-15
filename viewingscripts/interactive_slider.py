#!/usr/bin/env python3
"""Interactive RGB vs color-mask comparison viewer.

Shows RGB image on the left and predicted color mask on the right with an
interactive vertical split controlled by a slider and mouse drag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive side-by-side slider for RGB images and colorized masks."
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path("input") / "images",
        help="Directory containing RGB tiles (.tif or .tiff)",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("viewingscripts") / "out",
        help="Directory containing colorized mask tiles (.png)",
    )
    parser.add_argument(
        "--tile",
        type=str,
        default=None,
        help="Tile stem to open directly (for example: 930645_ne_tile_0)",
    )
    parser.add_argument(
        "--start-split",
        type=float,
        default=0.5,
        help="Initial split position as a fraction from 0.0 to 1.0",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Write a static preview PNG and exit (useful in headless environments)",
    )
    return parser.parse_args()


def is_non_interactive_backend() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _rgb_stem_map(rgb_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for ext in ("*.tif", "*.tiff"):
        for path in sorted(rgb_dir.glob(ext)):
            mapping[path.stem] = path
    return mapping


def _mask_stem_map(mask_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*.png")):
        mapping[path.stem] = path
    return mapping


def find_pairs(rgb_dir: Path, mask_dir: Path) -> list[tuple[str, Path, Path]]:
    rgb_map = _rgb_stem_map(rgb_dir)
    mask_map = _mask_stem_map(mask_dir)
    common_stems = sorted(set(rgb_map) & set(mask_map))
    return [(stem, rgb_map[stem], mask_map[stem]) for stem in common_stems]


def load_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported RGB shape for {path.name}: {arr.shape}")
    return arr.astype(np.uint8)


def load_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    target_h, target_w = target_hw
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.array(img, dtype=np.uint8)


def blend_split(left_img: np.ndarray, right_img: np.ndarray, split_x: int) -> np.ndarray:
    out = right_img.copy()
    out[:, :split_x, :] = left_img[:, :split_x, :]
    return out


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def first(iterable: Iterable[tuple[str, Path, Path]]) -> tuple[str, Path, Path] | None:
    for item in iterable:
        return item
    return None


class SliderViewer:
    def __init__(self, pairs: list[tuple[str, Path, Path]], start_split: float) -> None:
        self.pairs = pairs
        self.index = 0
        self.start_split = clamp01(start_split)

        self.fig = plt.figure(figsize=(11, 8))
        self.ax = self.fig.add_axes([0.06, 0.15, 0.88, 0.78])
        self.slider_ax = self.fig.add_axes([0.15, 0.06, 0.7, 0.03])

        self.slider = Slider(self.slider_ax, "Split", 0.0, 1.0, valinit=self.start_split)
        self.slider.on_changed(self._on_slider_change)

        self.dragging = False
        self.current_rgb: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.image_artist = None
        self.split_line = None

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_down)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_up)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._load_current_pair()

    def _load_current_pair(self) -> None:
        stem, rgb_path, mask_path = self.pairs[self.index]
        self.current_rgb = load_rgb(rgb_path)
        h, w, _ = self.current_rgb.shape
        self.current_mask = load_mask(mask_path, (h, w))

        self.ax.clear()
        composed = blend_split(self.current_rgb, self.current_mask, int(self.slider.val * w))
        self.image_artist = self.ax.imshow(composed)
        self.split_line = self.ax.axvline(self.slider.val * w, color="white", linewidth=2)

        self.ax.set_title(
            f"{stem}  |  left: RGB  right: mask  |  arrows: prev/next  r: reset  q: quit",
            fontsize=11,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw_idle()

    def _update_display(self) -> None:
        if self.current_rgb is None or self.current_mask is None:
            return

        h, w, _ = self.current_rgb.shape
        split_x = int(clamp01(self.slider.val) * w)
        composed = blend_split(self.current_rgb, self.current_mask, split_x)

        if self.image_artist is not None:
            self.image_artist.set_data(composed)
        if self.split_line is not None:
            self.split_line.set_xdata([split_x, split_x])

        self.fig.canvas.draw_idle()

    def _on_slider_change(self, _value: float) -> None:
        self._update_display()

    def _on_mouse_down(self, event) -> None:
        if event.inaxes != self.ax:
            return
        self.dragging = True
        self._set_split_from_event(event)

    def _on_mouse_up(self, _event) -> None:
        self.dragging = False

    def _on_mouse_move(self, event) -> None:
        if not self.dragging or event.inaxes != self.ax:
            return
        self._set_split_from_event(event)

    def _set_split_from_event(self, event) -> None:
        if self.current_rgb is None or event.xdata is None:
            return
        _, w, _ = self.current_rgb.shape
        fraction = clamp01(event.xdata / max(1.0, w - 1))
        self.slider.set_val(fraction)

    def _step(self, delta: int) -> None:
        if not self.pairs:
            return
        self.index = (self.index + delta) % len(self.pairs)
        self._load_current_pair()

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"right", "n"}:
            self._step(1)
        elif key in {"left", "p"}:
            self._step(-1)
        elif key == "r":
            self.slider.set_val(0.5)
        elif key in {"q", "escape"}:
            plt.close(self.fig)

    def show(self) -> None:
        plt.show()


def main() -> int:
    args = parse_args()

    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        raise SystemExit(
            "Run this script from the project root directory. "
            f"Expected: {REPO_ROOT} | Current: {cwd}"
        )

    rgb_dir = (cwd / args.rgb_dir).resolve() if not args.rgb_dir.is_absolute() else args.rgb_dir
    mask_dir = (cwd / args.mask_dir).resolve() if not args.mask_dir.is_absolute() else args.mask_dir

    if not rgb_dir.exists():
        raise SystemExit(f"RGB directory does not exist: {rgb_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Mask directory does not exist: {mask_dir}")

    pairs = find_pairs(rgb_dir, mask_dir)
    if not pairs:
        raise SystemExit(
            "No matched RGB/mask pairs found. Expected matching stems between "
            f"{rgb_dir} (.tif/.tiff) and {mask_dir} (.png)."
        )

    if args.tile:
        selected = first(pair for pair in pairs if pair[0] == args.tile)
        if selected is None:
            raise SystemExit(
                f"Tile '{args.tile}' not found in matched pairs. "
                "Check stem names in rgb and mask directories."
            )
        pairs = [selected]

    if args.save_preview is not None:
        stem, rgb_path, mask_path = pairs[0]
        rgb = load_rgb(rgb_path)
        h, w, _ = rgb.shape
        mask = load_mask(mask_path, (h, w))
        split_x = int(clamp01(args.start_split) * w)
        preview = blend_split(rgb, mask, split_x)
        output_path = args.save_preview if args.save_preview.is_absolute() else (cwd / args.save_preview)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(output_path)
        print(f"Saved preview: {output_path}")
        return 0

    if is_non_interactive_backend():
        raise SystemExit(
            "Matplotlib is using a non-interactive backend, so no window can open. "
            "Current backend: "
            f"{matplotlib.get_backend()}. "
            "Run from a desktop session with GUI support or use --save-preview to write an image."
        )

    viewer = SliderViewer(pairs=pairs, start_split=args.start_split)
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
