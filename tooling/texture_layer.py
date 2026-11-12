#!/usr/bin/env python3
"""CLI utility to layer, composite, and blend multiple PNG textures into one for block-game.

Supports:
- Standard alpha compositing ('normal' / Porter-Duff 'over' mode).
- Advanced blend modes: multiply, screen, overlay, add, darken, lighten, mask.
- Layer opacity / alpha scaling (percentages or float values).
- Offsets and canvas alignment for overlays of different dimensions.
- Stacking arbitrary numbers of layers (bottom to top).
- Batch layering:
  - Applying one or more overlays across a collection of base textures (e.g. CTM sets).
  - Pairwise 1-to-1 layering across matching lists of base and overlay files.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def parse_offset(offset_str: str) -> Tuple[int, int]:
    """Parse offset string like '0,0' or '2x4' or '4 8' into (x, y)."""
    offset_str = offset_str.strip()
    parts = [p.strip() for p in re.split(r"[,xX\s]+", offset_str) if p.strip()]
    if len(parts) == 1:
        v = int(parts[0])
        return (v, v)
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError(f"Invalid offset string: '{offset_str}'. Use format like '0,0' or '2x4'.")


def parse_opacity(opacity_str: Union[str, float, int]) -> float:
    """Parse opacity string like '0.8', '80%', or float."""
    if isinstance(opacity_str, (int, float)):
        return float(np.clip(float(opacity_str), 0.0, 1.0))
    s = str(opacity_str).strip()
    if s.endswith("%"):
        return float(np.clip(float(s[:-1]) / 100.0, 0.0, 1.0))
    return float(np.clip(float(s), 0.0, 1.0))


def blend_arrays(
    base: np.ndarray,
    top: np.ndarray,
    mode: str = "normal",
    opacity: float = 1.0,
    offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Blend top image array onto base image array.

    Args:
        base: uint8 RGBA array (H, W, 4).
        top: uint8 RGBA array (H2, W2, 4).
        mode: Blend mode: 'normal', 'multiply', 'screen', 'overlay', 'add', 'darken', 'lighten', 'mask'.
        opacity: Layer opacity [0.0, 1.0].
        offset: (x, y) offset on the base canvas.

    Returns:
        New uint8 RGBA array (H, W, 4).
    """
    out = np.array(base, copy=True)
    bh, bw = out.shape[:2]
    th, tw = top.shape[:2]
    ox, oy = offset

    # Compute overlapping bounding box
    bx1 = max(0, ox)
    by1 = max(0, oy)
    bx2 = min(bw, ox + tw)
    by2 = min(bh, oy + th)

    if bx1 >= bx2 or by1 >= by2:
        # No overlap
        return out

    tx1 = bx1 - ox
    ty1 = by1 - oy
    tx2 = tx1 + (bx2 - bx1)
    ty2 = ty1 + (by2 - by1)

    sub_base = out[by1:by2, bx1:bx2]
    sub_top = top[ty1:ty2, tx1:tx2]

    cb = sub_base[..., :3].astype(np.float32) / 255.0
    ab = sub_base[..., 3].astype(np.float32) / 255.0

    ca = sub_top[..., :3].astype(np.float32) / 255.0
    aa = (sub_top[..., 3].astype(np.float32) / 255.0) * float(opacity)

    mode = mode.lower()
    if mode in ("normal", "over"):
        c_blend = ca
    elif mode == "multiply":
        c_blend = cb * ca
    elif mode == "screen":
        c_blend = 1.0 - (1.0 - cb) * (1.0 - ca)
    elif mode == "overlay":
        c_blend = np.where(cb < 0.5, 2.0 * cb * ca, 1.0 - 2.0 * (1.0 - cb) * (1.0 - ca))
    elif mode in ("add", "addition", "linear_dodge"):
        c_blend = np.clip(cb + ca, 0.0, 1.0)
    elif mode == "darken":
        c_blend = np.minimum(cb, ca)
    elif mode == "lighten":
        c_blend = np.maximum(cb, ca)
    elif mode == "mask":
        # Uses top layer alpha or grayscale as a mask on base
        mask_val = (ca[..., 0] * 0.299 + ca[..., 1] * 0.587 + ca[..., 2] * 0.114) * aa
        out[by1:by2, bx1:bx2, 3] = np.clip(
            np.round(ab * mask_val * 255.0), 0, 255
        ).astype(np.uint8)
        return out
    else:
        raise ValueError(f"Unsupported blend mode: '{mode}'")

    a_out = aa + ab * (1.0 - aa)
    safe_a = np.maximum(a_out, 1e-10)

    c_out = (c_blend * aa[..., None] + cb * (ab * (1.0 - aa))[..., None]) / safe_a[..., None]
    c_out_uint8 = np.clip(np.round(c_out * 255.0), 0, 255).astype(np.uint8)
    a_out_uint8 = np.clip(np.round(a_out * 255.0), 0, 255).astype(np.uint8)

    out[by1:by2, bx1:bx2, :3] = c_out_uint8
    out[by1:by2, bx1:bx2, 3] = a_out_uint8

    return out


def layer_images(
    images: List[Union[Image.Image, np.ndarray, Path, str]],
    mode: str = "normal",
    opacity: float = 1.0,
    offset: Tuple[int, int] = (0, 0),
) -> Image.Image:
    """Composite a list of images in order from bottom to top."""
    if not images:
        raise ValueError("Cannot layer an empty list of images.")

    def to_array(item: Union[Image.Image, np.ndarray, Path, str]) -> np.ndarray:
        if isinstance(item, (str, Path)):
            img = Image.open(item).convert("RGBA")
            return np.array(img, dtype=np.uint8)
        elif isinstance(item, Image.Image):
            return np.array(item.convert("RGBA"), dtype=np.uint8)
        elif isinstance(item, np.ndarray):
            arr = np.array(item, copy=True)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
            elif arr.shape[-1] == 3:
                alpha = np.full((*arr.shape[:2], 1), 255, dtype=arr.dtype)
                arr = np.concatenate([arr, alpha], axis=-1)
            return arr.astype(np.uint8)
        raise TypeError(f"Unsupported image type: {type(item)}")

    curr = to_array(images[0])
    for top_item in images[1:]:
        top_arr = to_array(top_item)
        curr = blend_arrays(curr, top_arr, mode=mode, opacity=opacity, offset=offset)

    return Image.fromarray(curr, mode="RGBA")


def resolve_output_path(
    base_path: Optional[Path],
    overlay_path: Optional[Path],
    output_arg: Optional[str],
    output_dir_arg: Optional[str],
    naming_pattern: str,
    index: int = 1,
) -> Path:
    """Resolve destination path for a layered texture."""
    base_stem = base_path.stem if base_path else "base"
    overlay_stem = overlay_path.stem if overlay_path else "overlay"
    ext = base_path.suffix if (base_path and base_path.suffix) else ".png"
    parent_dir = Path(output_dir_arg) if output_dir_arg else (base_path.parent if base_path else Path("."))

    v_vars = {
        "name": base_stem,
        "base": base_stem,
        "overlay": overlay_stem,
        "stem": base_stem,
        "ext": ext,
        "i": index,
        "idx": index,
    }

    if output_arg:
        if any(f"{{{k}}}" in output_arg for k in v_vars):
            try:
                return Path(output_arg.format(**v_vars))
            except KeyError:
                pass
        return Path(output_arg)

    filename = naming_pattern.format(**v_vars)
    return parent_dir / filename


def expand_paths(patterns: List[str]) -> List[Path]:
    """Expand file paths and glob patterns."""
    results: List[Path] = []
    for pat in patterns:
        matched = [Path(p) for p in glob.glob(pat, recursive=True)]
        if matched:
            results.extend(sorted(matched))
        else:
            results.append(Path(pat))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool to layer, composite, and blend multiple PNG textures into one for block-game.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "layers",
        nargs="*",
        type=str,
        help="Input image paths ordered from bottom to top (base, layer1, layer2...).",
    )
    parser.add_argument(
        "-b",
        "--base",
        dest="bases",
        action="append",
        help="Base image path or glob pattern (can be specified multiple times).",
    )
    parser.add_argument(
        "-l",
        "--layer",
        "--overlay",
        dest="overlays",
        action="append",
        help="Overlay image path or glob pattern to place on top (can be specified multiple times).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        "--blend",
        dest="mode",
        choices=["normal", "multiply", "screen", "overlay", "add", "darken", "lighten", "mask"],
        default="normal",
        help="Blend mode (default: normal).\n"
        "  normal   : Standard alpha compositing (Porter-Duff over).\n"
        "  multiply : Darkens pixels by multiplying colors.\n"
        "  screen   : Lightens pixels (inverts, multiplies, inverts).\n"
        "  overlay  : Combination of multiply and screen based on base lightness.\n"
        "  add      : Linear dodge / additive blending.\n"
        "  darken   : Keeps minimum pixel value per channel.\n"
        "  lighten  : Keeps maximum pixel value per channel.\n"
        "  mask     : Uses overlay as an opacity mask on the base.",
    )
    parser.add_argument(
        "-a",
        "--opacity",
        "--alpha",
        dest="opacity",
        type=str,
        default="1.0",
        help="Overlay opacity (default: 1.0, e.g. '0.75' or '75%%').",
    )
    parser.add_argument(
        "--offset",
        type=str,
        default="0,0",
        help="Placement offset for overlay on base as 'x,y' (default: 0,0).",
    )
    parser.add_argument(
        "--pair",
        "--pairwise",
        action="store_true",
        help="When multiple bases and overlays are given, pair them 1-to-1 instead of all overlays on each base.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output filename or pattern (supports placeholders like {base}, {overlay}, {name}, {ext}).",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        help="Directory to save layered textures.",
    )
    parser.add_argument(
        "--naming",
        type=str,
        default=None,
        help="Naming pattern (default: '{base}_{overlay}{ext}' or '{base}_layered{ext}').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate and print output paths without saving images.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    opacity = parse_opacity(args.opacity)
    offset = parse_offset(args.offset)

    # Determine bases and overlays
    base_paths: List[Path] = []
    overlay_paths: List[Path] = []

    if args.bases:
        base_paths.extend(expand_paths(args.bases))
    if args.overlays:
        overlay_paths.extend(expand_paths(args.overlays))

    # If positional layers are passed
    if args.layers:
        expanded_pos = expand_paths(args.layers)
        if not base_paths and not overlay_paths:
            if len(expanded_pos) < 2:
                parser.error("At least 2 images required to layer (base and overlay).")
            # Base is the first, rest are overlays stacked in order
            base_paths = [expanded_pos[0]]
            overlay_paths = expanded_pos[1:]
        elif not base_paths:
            base_paths.extend(expanded_pos)
        else:
            overlay_paths.extend(expanded_pos)

    if not base_paths:
        parser.error("No base image(s) specified. Use positional args or -b / --base.")
    if not overlay_paths:
        parser.error("No overlay image(s) specified. Use positional args or -l / --layer.")

    # Determine default naming pattern
    if args.naming:
        naming_pattern = args.naming
    elif args.pair or len(overlay_paths) == 1:
        naming_pattern = "{base}_{overlay}{ext}"
    else:
        naming_pattern = "{base}_layered{ext}"

    total_created = 0

    # Pairwise mode: combine base[i] and overlay[i]
    if args.pair:
        count = min(len(base_paths), len(overlay_paths))
        if count == 0:
            print("Error: In pairwise mode, both bases and overlays must have at least 1 image.", file=sys.stderr)
            return 1

        for i in range(count):
            b_path = base_paths[i]
            o_path = overlay_paths[i]

            if not b_path.exists():
                print(f"Error: Base file not found: {b_path}", file=sys.stderr)
                return 1
            if not o_path.exists():
                print(f"Error: Overlay file not found: {o_path}", file=sys.stderr)
                return 1

            result_img = layer_images([b_path, o_path], mode=args.mode, opacity=opacity, offset=offset)

            out_path = resolve_output_path(
                base_path=b_path,
                overlay_path=o_path,
                output_arg=args.output,
                output_dir_arg=args.output_dir,
                naming_pattern=naming_pattern,
                index=i + 1,
            )

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                result_img.save(out_path)

            total_created += 1
            if not args.quiet:
                status = "[DRY-RUN] Would create" if args.dry_run else "Created"
                dims = f"{result_img.width}x{result_img.height}"
                print(f"{status}: {out_path} ({dims}, {b_path.name} + {o_path.name}, mode={args.mode})")

    else:
        # Standard mode: For each base, stack all overlays in order
        for b_idx, b_path in enumerate(base_paths):
            if not b_path.exists():
                print(f"Error: Base file not found: {b_path}", file=sys.stderr)
                return 1

            stack_paths = [b_path] + overlay_paths
            for p in overlay_paths:
                if not p.exists():
                    print(f"Error: Overlay file not found: {p}", file=sys.stderr)
                    return 1

            result_img = layer_images(stack_paths, mode=args.mode, opacity=opacity, offset=offset)

            first_overlay = overlay_paths[0] if overlay_paths else None
            out_path = resolve_output_path(
                base_path=b_path,
                overlay_path=first_overlay,
                output_arg=args.output,
                output_dir_arg=args.output_dir,
                naming_pattern=naming_pattern,
                index=b_idx + 1,
            )

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                result_img.save(out_path)

            total_created += 1
            if not args.quiet:
                status = "[DRY-RUN] Would create" if args.dry_run else "Created"
                dims = f"{result_img.width}x{result_img.height}"
                layers_desc = f"{b_path.name} + {len(overlay_paths)} overlay(s)"
                print(f"{status}: {out_path} ({dims}, {layers_desc}, mode={args.mode})")

    if not args.quiet:
        print(f"Done. Processed {total_created} composite texture(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
