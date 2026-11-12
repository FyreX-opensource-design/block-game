import numpy as np
import pytest
from PIL import Image

from tooling.texture_noise import (
    apply_noise,
    build_parser,
    generate_color_image,
    main,
    parse_color,
    parse_size,
    resolve_output_path,
)


def test_parse_color():
    assert parse_color("#ff0000") == (255, 0, 0, 255)
    assert parse_color("ff0000") == (255, 0, 0, 255)
    assert parse_color("#00ff0080") == (0, 255, 0, 128)
    assert parse_color("100, 150, 200") == (100, 150, 200, 255)
    assert parse_color("100, 150, 200, 128") == (100, 150, 200, 128)
    assert parse_color("blue") == (0, 0, 255, 255)

    with pytest.raises(ValueError):
        parse_color("invalid_color_123xyz")


def test_parse_size():
    assert parse_size("16") == (16, 16)
    assert parse_size("16x16") == (16, 16)
    assert parse_size("32,64") == (32, 64)
    with pytest.raises(ValueError):
        parse_size("16x16x16")


def test_apply_noise_mono_preserves_hue():
    # Solid color image: 16x16 RGB
    img = Image.new("RGB", (16, 16), (100, 120, 140))
    rng = np.random.default_rng(42)
    noisy = apply_noise(img, sigma=15.0, mode="mono", rng=rng)

    arr = np.array(noisy, dtype=np.int32)
    diff = arr - np.array([100, 120, 140], dtype=np.int32)

    # In mono mode, R, G, and B must have identical delta (unless clamped at 0 or 255)
    # Since 100, 120, 140 is well within [0, 255], clipping won't occur for sigma=15
    assert (diff[:, :, 0] == diff[:, :, 1]).all()
    assert (diff[:, :, 1] == diff[:, :, 2]).all()
    # Ensure noise actually changed pixels
    assert not (arr == 100).all()


def test_apply_noise_chromatic_color_mode():
    img = Image.new("RGB", (16, 16), (128, 128, 128))
    rng = np.random.default_rng(42)
    noisy = apply_noise(img, sigma=15.0, mode="color", rng=rng)

    arr = np.array(noisy, dtype=np.int32)
    diff = arr - 128

    # In color mode, deltas between R, G, B are independent
    assert not (diff[:, :, 0] == diff[:, :, 1]).all()


def test_preserve_alpha():
    # 2x2 image with one transparent pixel, one semi-transparent, two opaque
    data = np.array([
        [[200, 100, 50, 0], [200, 100, 50, 128]],
        [[200, 100, 50, 255], [200, 100, 50, 255]]
    ], dtype=np.uint8)
    img = Image.fromarray(data, mode="RGBA")

    rng = np.random.default_rng(42)
    noisy = apply_noise(img, sigma=20.0, preserve_alpha=True, rng=rng)
    noisy_arr = np.array(noisy)

    # Transparent pixel (0, 0) must remain 100% untouched
    np.testing.assert_array_equal(noisy_arr[0, 0], data[0, 0])

    # Alpha channels on all pixels must be exactly preserved
    np.testing.assert_array_equal(noisy_arr[:, :, 3], data[:, :, 3])

    # Opaque pixels have noise
    assert not np.array_equal(noisy_arr[1, 0, :3], data[1, 0, :3])


def test_grain_size():
    img = Image.new("RGB", (16, 16), (128, 128, 128))
    rng = np.random.default_rng(42)
    grain = 4
    noisy = apply_noise(img, sigma=20.0, grain_size=grain, rng=rng)
    arr = np.array(noisy)

    # Within each 4x4 block, all pixels should have identical values
    for gy in range(0, 16, grain):
        for gx in range(0, 16, grain):
            block = arr[gy:gy+grain, gx:gx+grain]
            assert np.all(block == block[0, 0])


def test_clamp_delta():
    img = Image.new("RGB", (16, 16), (128, 128, 128))
    rng = np.random.default_rng(42)
    noisy = apply_noise(img, sigma=50.0, clamp_delta=5.0, rng=rng)
    arr = np.array(noisy, dtype=np.int32)
    diff = np.abs(arr - 128)
    assert (diff <= 5).all()


def test_distribution_uniform():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    rng = np.random.default_rng(42)
    noisy = apply_noise(img, sigma=10.0, dist_type="uniform", rng=rng)
    arr = np.array(noisy, dtype=np.int32)
    diff = np.abs(arr - 128)
    # For uniform with sigma 10, max half width is 10 * sqrt(3) ~ 17.32 + rounding <= 18
    assert (diff <= 18).all()


def test_seed_determinism():
    img = Image.new("RGB", (16, 16), (100, 100, 100))
    noisy1 = apply_noise(img, sigma=15.0, rng=np.random.default_rng(999))
    noisy2 = apply_noise(img, sigma=15.0, rng=np.random.default_rng(999))
    np.testing.assert_array_equal(np.array(noisy1), np.array(noisy2))


def test_cli_multiple_variants_and_color(tmp_path):
    out_dir = tmp_path / "variants"
    ret = main([
        "--color", "#6b7280",
        "--size", "16x16",
        "-n", "3",
        "-d", str(out_dir),
        "-s", "8.0",
        "--seed", "123",
        "-q",
    ])
    assert ret == 0
    # Files should be named texture_6b7280_var1.png, _var2.png, _var3.png
    v1 = out_dir / "texture_6b7280_var1.png"
    v2 = out_dir / "texture_6b7280_var2.png"
    v3 = out_dir / "texture_6b7280_var3.png"
    assert v1.exists()
    assert v2.exists()
    assert v3.exists()

    img1 = np.array(Image.open(v1))
    img2 = np.array(Image.open(v2))
    assert not np.array_equal(img1, img2)


def test_cli_existing_image_and_naming_template(tmp_path):
    base_file = tmp_path / "stone.png"
    Image.new("RGBA", (16, 16), (120, 120, 120, 255)).save(base_file)

    ret = main([
        str(base_file),
        "-n", "2",
        "-o", str(tmp_path / "stone_variant_{i}.png"),
        "--sigma", "10",
        "-q",
    ])
    assert ret == 0
    assert (tmp_path / "stone_variant_1.png").exists()
    assert (tmp_path / "stone_variant_2.png").exists()


def test_cli_dry_run(tmp_path, capsys):
    base_file = tmp_path / "test_block.png"
    Image.new("RGB", (16, 16), (50, 50, 50)).save(base_file)

    ret = main([
        str(base_file),
        "-n", "2",
        "--dry-run",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN] Would create" in captured.out
    assert not (tmp_path / "test_block_var1.png").exists()

