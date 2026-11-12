#!/usr/bin/env python3
"""CLI utility to generate procedural noise textures and variations for block-game.

Supports:
- Generating multiple texture variants with controlled random seeds.
- Applying noise to existing textures (PNG, RGBA, RGB, grayscale) or solid colors.
- Monochromatic (luminance-preserving) and chromatic (per-channel) noise modes.
- Alpha channel preservation (transparent areas remain transparent).
- Configurable noise intensity (sigma), distribution (gaussian/uniform), mean shift,
  and pixel grain size.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageColor


def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    """Parse a color string (hex, rgb/rgba string, or color name) into RGBA tuple."""
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


def apply_noise(
    img: Union[Image.Image, np.ndarray],
    sigma: float = 12.0,
    mean: float = 0.0,
    mode: str = "mono",
    dist_type: str = "gaussian",
    grain_size: int = 1,
    preserve_alpha: bool = True,
    noise_alpha: bool = False,
    clamp_delta: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Apply controlled noise to an image.

    Args:
        img: Input PIL Image or uint8 numpy array.
        sigma: Standard deviation / strength of the noise.
        mean: Mean shift added to pixel values (positive brightens, negative darkens).
        mode: 'mono' (luminance only, preserving hue) or 'color' (independent RGB noise).
        dist_type: 'gaussian' (normal) or 'uniform'.
        grain_size: Pixel size for coarse/retro grain (default 1).
        preserve_alpha: If True, do not add noise to transparent pixels (alpha == 0),
                        and keep the alpha channel unchanged.
        noise_alpha: If True, also apply noise to the alpha channel (ignored if preserve_alpha is True).
        clamp_delta: Optional maximum absolute noise change per pixel.
        rng: Optional numpy random Generator for deterministic results.

    Returns:
        New PIL Image with noise applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert PIL Image to numpy array
    if isinstance(img, Image.Image):
        orig_mode = img.mode
        if orig_mode == "P":
            # Convert palette to RGBA or RGB
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        arr = np.array(img).astype(np.float32)
        target_mode = img.mode
    else:
        arr = img.astype(np.float32)
        if arr.ndim == 2:
            target_mode = "L"
        elif arr.shape[2] == 3:
            target_mode = "RGB"
        elif arr.shape[2] == 4:
            target_mode = "RGBA"
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    h, w = arr.shape[0], arr.shape[1]
    grain_size = max(1, int(grain_size))

    # Dimensions for noise generation
    gh = (h + grain_size - 1) // grain_size
    gw = (w + grain_size - 1) // grain_size

    has_alpha = (arr.ndim == 3 and arr.shape[2] == 4)
    is_grayscale = (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1))

    # Determine noise channels
    if is_grayscale or mode == "mono":
        noise_channels = 1
    else:
        # Chromatic / color noise across RGB channels
        noise_channels = 3 if has_alpha else arr.shape[2]

    # Generate raw noise
    if dist_type == "gaussian":
        noise = rng.normal(loc=mean, scale=sigma, size=(gh, gw, noise_channels))
    elif dist_type == "uniform":
        # Uniform noise with matching standard deviation: [-sigma * sqrt(3), sigma * sqrt(3)]
        half_width = sigma * math.sqrt(3.0)
        noise = rng.uniform(low=mean - half_width, high=mean + half_width, size=(gh, gw, noise_channels))
    else:
        raise ValueError(f"Unknown noise distribution '{dist_type}'. Choose 'gaussian' or 'uniform'.")

    # Upscale noise if grain_size > 1
    if grain_size > 1:
        noise = np.repeat(np.repeat(noise, grain_size, axis=0), grain_size, axis=1)[:h, :w, :]
    else:
        noise = noise[:h, :w, :]

    if clamp_delta is not None and clamp_delta > 0:
        noise = np.clip(noise, -clamp_delta, clamp_delta)

    result = arr.copy()

    if is_grayscale:
        # Grayscale image
        base = result if result.ndim == 2 else result[:, :, 0]
        noisy = np.clip(base + noise[:, :, 0], 0, 255)
        if result.ndim == 2:
            result = noisy
        else:
            result[:, :, 0] = noisy
    elif has_alpha:
        # RGBA image
        alpha = result[:, :, 3]
        rgb = result[:, :, :3]

        if preserve_alpha:
            # Mask out transparent pixels so they stay untouched
            opaque_mask = (alpha > 0)[:, :, np.newaxis]
            noisy_rgb = np.clip(rgb + noise, 0, 255)
            result[:, :, :3] = np.where(opaque_mask, noisy_rgb, rgb)
        else:
            result[:, :, :3] = np.clip(rgb + noise, 0, 255)
            if noise_alpha:
                # Add separate noise to alpha channel
                if dist_type == "gaussian":
                    alpha_noise = rng.normal(loc=mean, scale=sigma, size=(gh, gw, 1))
                else:
                    half_width = sigma * math.sqrt(3.0)
                    alpha_noise = rng.uniform(low=mean - half_width, high=mean + half_width, size=(gh, gw, 1))
                if grain_size > 1:
                    alpha_noise = np.repeat(np.repeat(alpha_noise, grain_size, axis=0), grain_size, axis=1)[:h, :w, 0]
                else:
                    alpha_noise = alpha_noise[:h, :w, 0]
                if clamp_delta is not None and clamp_delta > 0:
                    alpha_noise = np.clip(alpha_noise, -clamp_delta, clamp_delta)
                result[:, :, 3] = np.clip(alpha + alpha_noise, 0, 255)
    else:
        # RGB image
        result = np.clip(result + noise, 0, 255)

    uint8_result = np.round(result).astype(np.uint8)
    return Image.fromarray(uint8_result, mode=target_mode)


def generate_color_image(color_str: str, size: Tuple[int, int] = (16, 16)) -> Image.Image:
    """Create a flat base image of the given color and size."""
    rgba = parse_color(color_str)
    return Image.new("RGBA", size, rgba)


def resolve_output_path(
    input_path: Optional[Path],
    output_arg: Optional[str],
    output_dir_arg: Optional[str],
    naming_pattern: str,
    variant_idx: int,
    total_variants: int,
    default_name: str = "texture",
) -> Path:
    """Determine the destination file path for a given variant."""
    if input_path is not None:
        base_name = input_path.stem
        ext = input_path.suffix or ".png"
        parent_dir = input_path.parent
    else:
        base_name = default_name
        ext = ".png"
        parent_dir = Path.cwd()

    if output_dir_arg:
        parent_dir = Path(output_dir_arg)

    # Format variables
    v_vars = {
        "name": base_name,
        "ext": ext,
        "i": variant_idx,
        "idx": variant_idx - 1,
        "count": total_variants,
    }

    if output_arg:
        out_p = Path(output_arg)
        # Check if output is intended as a directory
        if output_arg.endswith(("/", "\\")) or (out_p.exists() and out_p.is_dir()):
            parent_dir = out_p
            filename = naming_pattern.format(**v_vars)
            return parent_dir / filename

        # If output contains placeholder like {i} or {name}
        if "{" in output_arg and "}" in output_arg:
            try:
                formatted = output_arg.format(**v_vars)
                return Path(formatted)
            except KeyError:
                pass

        # If single variant, output exact path
        if total_variants == 1:
            return out_p

        # If multiple variants and no placeholder, append _<variant_idx> before suffix
        stem = out_p.stem
        suffix = out_p.suffix or ext
        out_dir = out_p.parent
        return out_dir / f"{stem}_{variant_idx}{suffix}"

    # Default pattern naming
    filename = naming_pattern.format(**v_vars)
    return parent_dir / filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool to add controlled light noise to game textures and generate multiple variants.",
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
        help="Number of noisy variants to generate per input image (default: 1).",
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
    parser.add_argument(
        "-s",
        "--sigma",
        "--intensity",
        type=float,
        default=12.0,
        help="Standard deviation / intensity of the noise grain (default: 12.0).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["mono", "monochrome", "color", "chromatic"],
        default="mono",
        help="Noise mode (default: mono).\n"
        "  mono  : Monochromatic / luminance noise (identical delta across R, G, B).\n"
        "          Preserves the hue and tint of color textures.\n"
        "  color : Independent noise per color channel (chromatic grain).",
    )
    parser.add_argument(
        "-t",
        "--type",
        "--distribution",
        dest="dist_type",
        choices=["gaussian", "uniform"],
        default="gaussian",
        help="Noise distribution type (default: gaussian).",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.0,
        help="Mean shift added to pixels (default: 0.0). Positive brightens, negative darkens.",
    )
    parser.add_argument(
        "-g",
        "--grain-size",
        type=int,
        default=1,
        help="Grain pixel scale (default: 1). Use > 1 for chunky retro pixel-art noise.",
    )
    parser.add_argument(
        "--clamp-delta",
        type=float,
        default=None,
        help="Maximum absolute noise delta per pixel to avoid extreme outliers.",
    )
    parser.add_argument(
        "--preserve-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve alpha channel and protect transparent pixels from noise (default: True).",
    )
    parser.add_argument(
        "--noise-alpha",
        action="store_true",
        default=False,
        help="Apply noise to the alpha channel as well (overrides --preserve-alpha).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for deterministic generation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output filename or pattern (supports placeholders like {name}, {i}, {idx}, {ext}).",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        help="Directory to save generated textures.",
    )
    parser.add_argument(
        "--naming",
        type=str,
        default=None,
        help="Naming pattern for variants (default: '{name}_var{i}{ext}' for multiple, '{name}_noisy{ext}' for 1).",
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
        naming_pattern = "{name}_noisy{ext}"

    # Normalize mode
    mode = "mono" if args.mode in ("mono", "monochrome") else "color"

    # Seed setup
    base_seed = args.seed

    preserve_alpha = False if args.noise_alpha else args.preserve_alpha

    items_to_process = input_paths if input_paths else [None]

    total_created = 0

    for item_idx, in_path in enumerate(items_to_process):
        # Load or generate base image
        if in_path is not None:
            if not in_path.exists():
                print(f"Error: Input file not found: {in_path}", file=sys.stderr)
                return 1
            try:
                base_img = Image.open(in_path)
                # Load pixel data to avoid keeping file handles open
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
            # Clean name for color output
            sanitized_color = re.sub(r"[^a-zA-Z0-9]+", "_", args.color).strip("_")
            default_name = f"texture_{sanitized_color}"

        for v in range(1, args.count + 1):
            # Variant seed calculation
            if base_seed is not None:
                v_seed = base_seed + (item_idx * 1000) + v
                rng = np.random.default_rng(v_seed)
            else:
                rng = np.random.default_rng()

            noisy_img = apply_noise(
                img=base_img,
                sigma=args.sigma,
                mean=args.mean,
                mode=mode,
                dist_type=args.dist_type,
                grain_size=args.grain_size,
                preserve_alpha=preserve_alpha,
                noise_alpha=args.noise_alpha,
                clamp_delta=args.clamp_delta,
                rng=rng,
            )

            out_path = resolve_output_path(
                input_path=in_path,
                output_arg=args.output,
                output_dir_arg=args.output_dir,
                naming_pattern=naming_pattern,
                variant_idx=v,
                total_variants=args.count,
                default_name=default_name,
            )

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                noisy_img.save(out_path)

            total_created += 1
            if not args.quiet:
                status = "[DRY-RUN] Would create" if args.dry_run else "Created"
                dims = f"{noisy_img.width}x{noisy_img.height}"
                print(f"{status}: {out_path} ({dims}, {noisy_img.mode}, sigma={args.sigma}, mode={mode})")

    if not args.quiet:
        print(f"Done. Processed {len(items_to_process)} texture(s), generated {total_created} file(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
