#!/usr/bin/env python3
"""CLI utility to adjust hue, saturation, and brightness/value (HSB/HSV) for block-game textures.

Supports:
- Hue shifts (relative degrees +/- 360) and absolute hue overrides (0..360).
- Saturation scaling (multipliers, percentage strings) and additive deltas.
- Brightness / Value scaling and additive shifts.
- Contrast adjustments around midtones.
- Color tinting / colorizing with controllable blend strength.
- Alpha channel preservation (transparent and semi-transparent pixels kept intact).
- Generating multiple stepped variants (e.g. hue palette sweeps, brightness gradients).
- Batch processing multiple input textures or generating from solid base colors.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageColor


def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    """Parse a color string (hex, rgb/rgba string, or color name) into an RGBA tuple."""
    color_str = color_str.strip()

    # Comma-separated or space-separated numbers: "r,g,b" or "r,g,b,a"
    parts = [p.strip() for p in re.split(r"[,;\s]+", color_str) if p.strip()]
    if len(parts) in (3, 4):
        try:
            nums = [int(p) for p in parts]
            if len(nums) == 3:
                nums.append(255)
            for v in nums:
                if not (0 <= v <= 255):
                    raise ValueError(f"Color channel {v} out of range [0, 255]")
            return (nums[0], nums[1], nums[2], nums[3])
        except ValueError:
            pass

    # Hex without hash
    if re.fullmatch(r"[0-9a-fA-F]{6}|[0-9a-fA-F]{8}|[0-9a-fA-F]{3}|[0-9a-fA-F]{4}", color_str):
        color_str = "#" + color_str

    try:
        rgba = ImageColor.getcolor(color_str, "RGBA")
        return rgba
    except ValueError as e:
        raise ValueError(f"Invalid color specification '{color_str}': {e}") from e


def parse_size(size_str: str) -> Tuple[int, int]:
    """Parse size string like '16', '16x16', or '16,16' into (width, height)."""
    size_str = size_str.lower().strip()
    parts = re.split(r"[x,\s]+", size_str)
    if len(parts) == 1:
        s = int(parts[0])
        return (s, s)
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError(f"Invalid size string: '{size_str}'. Use format like '16x16' or '16'.")


def parse_amount(val_str: Union[str, float, int], default_multiplier: bool = True) -> float:
    """Parse float, integer, or percentage string (e.g. '120%', '+20%', '-0.15').

    If percentage syntax ('%') is used:
    - In multiplier context: '150%' -> 1.5, '+20%' -> 1.2, '-20%' -> 0.8
    - In delta context: '20%' -> 0.2, '-10%' -> -0.1
    """
    if isinstance(val_str, (int, float)):
        return float(val_str)

    val_str = str(val_str).strip()
    if val_str.endswith("%"):
        num = float(val_str[:-1].strip()) / 100.0
        if default_multiplier:
            if val_str.startswith("+") or val_str.startswith("-"):
                return 1.0 + num
            return num
        return num
    return float(val_str)


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float array [0..1] to HSV float array [H: 0..1, S: 0..1, V: 0..1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc

    s = np.where(maxc > 0.0, deltac / np.maximum(maxc, 1e-10), 0.0)

    rc = (maxc - r) / np.maximum(deltac, 1e-10)
    gc = (maxc - g) / np.maximum(deltac, 1e-10)
    bc = (maxc - b) / np.maximum(deltac, 1e-10)

    h = np.zeros_like(maxc)
    mask_r = (r == maxc) & (deltac > 0.0)
    mask_g = (g == maxc) & (deltac > 0.0) & ~mask_r
    mask_b = (b == maxc) & (deltac > 0.0) & ~mask_r & ~mask_g

    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = (2.0 + rc - bc)[mask_g]
    h[mask_b] = (4.0 + gc - rc)[mask_b]

    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV float array [H: 0..1, S: 0..1, V: 0..1] to RGB float array [0..1]."""
    h = hsv[..., 0] % 1.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, 1.0)

    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def generate_color_image(color_spec: str, size: Tuple[int, int] = (16, 16)) -> Image.Image:
    """Create a uniform color image with dimensions (width, height)."""
    rgba = parse_color(color_spec)
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[:, :] = rgba
    return Image.fromarray(arr, mode="RGBA")


def adjust_hsb(
    img: Union[Image.Image, np.ndarray],
    hue_shift: float = 0.0,
    set_hue: Optional[float] = None,
    saturation: float = 1.0,
    saturation_delta: float = 0.0,
    brightness: float = 1.0,
    brightness_delta: float = 0.0,
    contrast: float = 1.0,
    tint_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    tint_strength: float = 0.0,
    preserve_alpha: bool = True,
    alpha_mult: float = 1.0,
) -> Image.Image:
    """Apply Hue, Saturation, Brightness (HSB/HSV), contrast, and tint adjustments.

    Args:
        img: Input PIL Image or uint8 numpy array.
        hue_shift: Degrees to rotate hue (-360 to +360).
        set_hue: If provided, set absolute hue in degrees [0, 360], overriding original hue.
        saturation: Multiplier for saturation (1.0 = no change, 0.0 = desaturated/grayscale).
        saturation_delta: Additive shift to saturation [-1.0, +1.0].
        brightness: Multiplier for brightness/value (1.0 = no change, 0.0 = black).
        brightness_delta: Additive shift to brightness [-1.0, +1.0].
        contrast: Contrast multiplier (1.0 = no change, > 1.0 increases contrast).
        tint_color: Optional tint color (hex string, name, or RGB tuple) to blend.
        tint_strength: Blend factor for tint [0.0 = no tint, 1.0 = fully tinted].
        preserve_alpha: If True, transparent pixels (alpha == 0) remain untouched.
        alpha_mult: Optional multiplier for alpha channel (1.0 = unchanged).

    Returns:
        New PIL Image with adjustments applied.
    """
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGBA")
        arr = np.array(pil_img, dtype=np.uint8)
    else:
        arr = np.array(img, copy=True)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.full((*arr.shape[:2], 1), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)

    has_alpha = True
    alpha_channel = arr[..., 3].astype(np.float32) / 255.0

    # Mask for non-transparent pixels
    if preserve_alpha:
        active_mask = (arr[..., 3] > 0)
    else:
        active_mask = np.ones(arr.shape[:2], dtype=bool)

    # RGB float in [0, 1]
    rgb_float = arr[..., :3].astype(np.float32) / 255.0

    # Convert to HSV
    hsv = rgb_to_hsv_np(rgb_float)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    # 1. Hue Adjustment
    if set_hue is not None:
        target_h = (set_hue / 360.0) % 1.0
        h = np.where(active_mask, target_h, h)
        # If original was completely desaturated (s == 0), give it slight saturation to show hue
        s = np.where(active_mask & (s == 0.0), 0.5, s)
    elif hue_shift != 0.0:
        shift_norm = (hue_shift / 360.0) % 1.0
        h = np.where(active_mask, (h + shift_norm) % 1.0, h)

    # 2. Saturation Adjustment
    if saturation != 1.0:
        s = s * saturation
    if saturation_delta != 0.0:
        s = s + saturation_delta
    s = np.clip(s, 0.0, 1.0)

    # 3. Brightness / Value Adjustment
    if brightness != 1.0:
        v = v * brightness
    if brightness_delta != 0.0:
        v = v + brightness_delta
    v = np.clip(v, 0.0, 1.0)

    # 4. Contrast Adjustment (applied on Value channel around midpoint 0.5)
    if contrast != 1.0:
        v = (v - 0.5) * contrast + 0.5
        v = np.clip(v, 0.0, 1.0)

    # Recombine to RGB
    new_hsv = np.stack([h, s, v], axis=-1)
    new_rgb = hsv_to_rgb_np(new_hsv)

    # 5. Optional Tinting
    if tint_color is not None and tint_strength > 0.0:
        tint_rgba = parse_color(tint_color) if isinstance(tint_color, str) else (*tint_color, 255)
        tint_rgb = np.array(tint_rgba[:3], dtype=np.float32) / 255.0
        tint_blend = np.clip(tint_strength, 0.0, 1.0)
        new_rgb = (1.0 - tint_blend) * new_rgb + tint_blend * tint_rgb
        new_rgb = np.clip(new_rgb, 0.0, 1.0)

    # Convert back to uint8
    out_arr = np.zeros_like(arr)
    rgb_uint8 = np.clip(np.round(new_rgb * 255.0), 0, 255).astype(np.uint8)

    if preserve_alpha:
        out_arr[..., :3] = np.where(active_mask[..., None], rgb_uint8, arr[..., :3])
    else:
        out_arr[..., :3] = rgb_uint8

    # Final Alpha
    if alpha_mult != 1.0:
        final_alpha = np.clip(np.round(alpha_channel * alpha_mult * 255.0), 0, 255).astype(np.uint8)
        out_arr[..., 3] = final_alpha
    else:
        out_arr[..., 3] = arr[..., 3]

    return Image.fromarray(out_arr, mode="RGBA")


def resolve_output_path(
    input_path: Optional[Path],
    output_arg: Optional[str],
    output_dir_arg: Optional[str],
    naming_pattern: str,
    variant_idx: int,
    total_variants: int,
    default_name: str,
    hue: float = 0.0,
    sat: float = 1.0,
    bright: float = 1.0,
) -> Path:
    """Resolve destination path for an adjusted texture."""
    stem = input_path.stem if input_path else default_name
    ext = input_path.suffix if (input_path and input_path.suffix) else ".png"
    parent_dir = Path(output_dir_arg) if output_dir_arg else (input_path.parent if input_path else Path("."))

    v_vars = {
        "name": stem,
        "stem": stem,
        "ext": ext,
        "i": variant_idx,
        "idx": variant_idx,
        "hue": int(round(hue)),
        "sat": f"{sat:.2f}".rstrip("0").rstrip("."),
        "bright": f"{bright:.2f}".rstrip("0").rstrip("."),
    }

    if output_arg:
        out_p = Path(output_arg)
        # Check if placeholders exist in output string
        if any(f"{{{k}}}" in output_arg for k in v_vars):
            try:
                formatted = output_arg.format(**v_vars)
                return Path(formatted)
            except KeyError:
                pass

        if total_variants == 1:
            return out_p

        # If multiple variants and no placeholders, append index before suffix
        p_stem = out_p.stem
        p_suffix = out_p.suffix or ext
        p_dir = out_p.parent
        return p_dir / f"{p_stem}_{variant_idx}{p_suffix}"

    # Default pattern naming
    filename = naming_pattern.format(**v_vars)
    return parent_dir / filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool to adjust hue, saturation, and brightness for game textures and generate palette variants.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        type=str,
        help="Input image path(s). Can be omitted if --color is specified.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="flag_inputs",
        action="append",
        help="Additional or alternative input file path (can be used multiple times).",
    )
    parser.add_argument(
        "-n",
        "--count",
        "--variants",
        type=int,
        default=1,
        help="Number of variants to generate per input image (default: 1).",
    )
    parser.add_argument(
        "-c",
        "--color",
        type=str,
        help="Generate from base solid color instead of an input file.\n"
        "Accepts hex ('#4a752c', '4a752c'), rgb/rgba ('74,117,44' or '74,117,44,255'), or color names.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="16x16",
        help="Texture size (width x height) when using --color (e.g. '16x16' or '16', default: 16x16).",
    )

    # Hue
    parser.add_argument(
        "-H",
        "--hue",
        "--hue-shift",
        dest="hue_shift",
        type=float,
        default=0.0,
        help="Hue rotation in degrees (-360 to +360, e.g. 120 turns red to green).",
    )
    parser.add_argument(
        "--set-hue",
        type=float,
        default=None,
        help="Set absolute hue in degrees [0, 360], overriding original hue.",
    )
    parser.add_argument(
        "--hue-step",
        type=float,
        default=0.0,
        help="Additional hue shift added per variant (useful with -n / --count > 1).",
    )

    # Saturation
    parser.add_argument(
        "-S",
        "--saturation",
        "--sat",
        dest="saturation",
        type=str,
        default="1.0",
        help="Saturation multiplier (1.0 = unchanged, 0.0 = grayscale, 1.5 = +50%%, supports '150%%').",
    )
    parser.add_argument(
        "--sat-delta",
        type=float,
        default=0.0,
        help="Additive delta added to saturation (-1.0 to +1.0).",
    )
    parser.add_argument(
        "--sat-step",
        type=float,
        default=0.0,
        help="Additional saturation delta added per variant.",
    )

    # Brightness / Value
    parser.add_argument(
        "-B",
        "-V",
        "--brightness",
        "--value",
        dest="brightness",
        type=str,
        default="1.0",
        help="Brightness / Value multiplier (1.0 = unchanged, 0.5 = darker, 1.2 = +20%%, supports '120%%').",
    )
    parser.add_argument(
        "--brightness-delta",
        "--value-delta",
        dest="brightness_delta",
        type=float,
        default=0.0,
        help="Additive delta added to brightness (-1.0 to +1.0).",
    )
    parser.add_argument(
        "--brightness-step",
        "--value-step",
        dest="brightness_step",
        type=float,
        default=0.0,
        help="Additional brightness delta added per variant.",
    )

    # Contrast
    parser.add_argument(
        "-C",
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast multiplier (1.0 = unchanged, >1.0 increases contrast, <1.0 decreases).",
    )

    # Tint
    parser.add_argument(
        "--tint",
        type=str,
        default=None,
        help="Tint color (hex, rgb, or color name) to blend into the texture.",
    )
    parser.add_argument(
        "--tint-strength",
        type=float,
        default=0.2,
        help="Strength of tint blend from 0.0 (no effect) to 1.0 (full color) [default: 0.2].",
    )

    # Alpha & IO
    parser.add_argument(
        "--preserve-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve alpha channel and protect transparent pixels from modifications (default: True).",
    )
    parser.add_argument(
        "--opacity",
        "--alpha-mult",
        dest="alpha_mult",
        type=float,
        default=1.0,
        help="Alpha channel multiplier (1.0 = unchanged, 0.5 = 50%% opacity).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output filename or pattern (supports placeholders like {name}, {hue}, {i}, {ext}).",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        help="Directory to save adjusted textures.",
    )
    parser.add_argument(
        "--naming",
        type=str,
        default=None,
        help="Naming pattern for variants (default: '{name}_var{i}{ext}' for multiple, '{name}_adj{ext}' for 1).",
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

    # Gather inputs
    input_paths: List[Optional[Path]] = []
    if args.inputs:
        input_paths.extend([Path(p) for p in args.inputs])
    if args.flag_inputs:
        input_paths.extend([Path(p) for p in args.flag_inputs])

    # Validate inputs
    if not input_paths and not args.color:
        parser.error("No input specified. Provide one or more input images, or use --color.")

    # Determine default naming pattern
    if args.naming:
        naming_pattern = args.naming
    elif args.count > 1:
        naming_pattern = "{name}_var{i}{ext}"
    else:
        naming_pattern = "{name}_adj{ext}"

    base_sat_mult = parse_amount(args.saturation, default_multiplier=True)
    base_bright_mult = parse_amount(args.brightness, default_multiplier=True)

    items_to_process = input_paths if input_paths else [None]
    total_created = 0

    for item_idx, in_path in enumerate(items_to_process):
        if in_path is not None:
            if not in_path.exists():
                print(f"Error: Input file not found: {in_path}", file=sys.stderr)
                return 1
            try:
                base_img = Image.open(in_path)
                base_img.load()
            except Exception as e:
                print(f"Error opening image '{in_path}': {e}", file=sys.stderr)
                return 1
            default_name = in_path.stem
        else:
            try:
                size = parse_size(args.size)
                base_img = generate_color_image(args.color, size=size)
            except Exception as e:
                print(f"Error generating base color image: {e}", file=sys.stderr)
                return 1
            sanitized_color = re.sub(r"[^a-zA-Z0-9]+", "_", args.color).strip("_")
            default_name = f"texture_{sanitized_color}"

        for v in range(1, args.count + 1):
            # Calculate variant parameters
            step_factor = (v - 1) if args.count > 1 else 0
            cur_hue = args.hue_shift + (step_factor * args.hue_step)
            cur_sat_delta = args.sat_delta + (step_factor * args.sat_step)
            cur_bright_delta = args.brightness_delta + (step_factor * args.brightness_step)

            adjusted_img = adjust_hsb(
                img=base_img,
                hue_shift=cur_hue,
                set_hue=args.set_hue,
                saturation=base_sat_mult,
                saturation_delta=cur_sat_delta,
                brightness=base_bright_mult,
                brightness_delta=cur_bright_delta,
                contrast=args.contrast,
                tint_color=args.tint,
                tint_strength=args.tint_strength,
                preserve_alpha=args.preserve_alpha,
                alpha_mult=args.alpha_mult,
            )

            out_path = resolve_output_path(
                input_path=in_path,
                output_arg=args.output,
                output_dir_arg=args.output_dir,
                naming_pattern=naming_pattern,
                variant_idx=v,
                total_variants=args.count,
                default_name=default_name,
                hue=cur_hue,
                sat=base_sat_mult + cur_sat_delta,
                bright=base_bright_mult + cur_bright_delta,
            )

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adjusted_img.save(out_path)

            total_created += 1
            if not args.quiet:
                status = "[DRY-RUN] Would create" if args.dry_run else "Created"
                dims = f"{adjusted_img.width}x{adjusted_img.height}"
                print(
                    f"{status}: {out_path} ({dims}, {adjusted_img.mode}, "
                    f"hue={cur_hue:+.1f}deg, sat={base_sat_mult:.2f}, bright={base_bright_mult:.2f})"
                )

    if not args.quiet:
        print(f"Done. Processed {len(items_to_process)} texture(s), generated {total_created} file(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

