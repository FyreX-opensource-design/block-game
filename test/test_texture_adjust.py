import numpy as np
import pytest
from PIL import Image

from tooling.texture_adjust import (
    adjust_hsb,
    build_parser,
    generate_color_image,
    hsv_to_rgb_np,
    main,
    parse_amount,
    parse_color,
    parse_size,
    resolve_output_path,
    rgb_to_hsv_np,
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


def test_parse_amount():
    assert parse_amount(1.5) == 1.5
    assert parse_amount("1.25") == 1.25
    assert parse_amount("150%") == 1.5
    assert parse_amount("+20%") == 1.2
    assert parse_amount("-20%") == 0.8
    assert parse_amount("20%", default_multiplier=False) == 0.2


def test_hsv_rgb_roundtrip():
    test_rgb = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [[0.8, 0.4, 0.2], [0.1, 0.9, 0.5]],
    ], dtype=np.float32)

    hsv = rgb_to_hsv_np(test_rgb)
    rgb_back = hsv_to_rgb_np(hsv)
    np.testing.assert_allclose(test_rgb, rgb_back, atol=1e-5)


def test_hue_shift():
    # Pure red image (255, 0, 0)
    red_img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))

    # Shift +120 degrees -> Green (0, 255, 0)
    green_img = adjust_hsb(red_img, hue_shift=120.0)
    green_arr = np.array(green_img)
    assert green_arr[0, 0, 0] == 0
    assert green_arr[0, 0, 1] == 255
    assert green_arr[0, 0, 2] == 0

    # Shift +240 degrees -> Blue (0, 0, 255)
    blue_img = adjust_hsb(red_img, hue_shift=240.0)
    blue_arr = np.array(blue_img)
    assert blue_arr[0, 0, 0] == 0
    assert blue_arr[0, 0, 1] == 0
    assert blue_arr[0, 0, 2] == 255

    # Shift +360 degrees -> Full circle back to Red
    back_red = adjust_hsb(red_img, hue_shift=360.0)
    back_arr = np.array(back_red)
    assert back_arr[0, 0, 0] == 255
    assert back_arr[0, 0, 1] == 0
    assert back_arr[0, 0, 2] == 0


def test_set_hue():
    img = Image.new("RGBA", (4, 4), (100, 150, 200, 255))
    # Set hue to 0 (red)
    red_h = adjust_hsb(img, set_hue=0.0)
    arr = np.array(red_h)
    # Hue=0 means R is dominant
    assert arr[0, 0, 0] > arr[0, 0, 1]
    assert arr[0, 0, 0] > arr[0, 0, 2]


def test_saturation_adjustment():
    img = Image.new("RGBA", (4, 4), (200, 100, 50, 255))

    # Desaturate completely -> Grayscale (R == G == B)
    gray = adjust_hsb(img, saturation=0.0)
    gray_arr = np.array(gray)
    assert gray_arr[0, 0, 0] == gray_arr[0, 0, 1]
    assert gray_arr[0, 0, 1] == gray_arr[0, 0, 2]

    # Increase saturation
    sat = adjust_hsb(img, saturation=1.5)
    sat_arr = np.array(sat)
    # Delta between max and min channel should be larger
    diff_orig = int(np.max(np.array(img)[0, 0, :3])) - int(np.min(np.array(img)[0, 0, :3]))
    diff_sat = int(np.max(sat_arr[0, 0, :3])) - int(np.min(sat_arr[0, 0, :3]))
    assert diff_sat > diff_orig


def test_brightness_adjustment():
    img = Image.new("RGBA", (4, 4), (100, 100, 100, 255))

    # Double brightness
    brighter = adjust_hsb(img, brightness=1.5)
    b_arr = np.array(brighter)
    assert b_arr[0, 0, 0] > 100

    # Half brightness
    darker = adjust_hsb(img, brightness=0.5)
    d_arr = np.array(darker)
    assert d_arr[0, 0, 0] < 100


def test_contrast_adjustment():
    # Image with light and dark pixels
    data = np.array([
        [[60, 60, 60, 255], [200, 200, 200, 255]]
    ], dtype=np.uint8)
    img = Image.fromarray(data, mode="RGBA")

    higher_contrast = adjust_hsb(img, contrast=1.5)
    hc_arr = np.array(higher_contrast)

    # Dark pixel (< 128) becomes darker, light pixel (> 128) becomes brighter
    assert hc_arr[0, 0, 0] < 60
    assert hc_arr[0, 1, 0] > 200


def test_tint_adjustment():
    img = Image.new("RGBA", (4, 4), (128, 128, 128, 255))
    tinted = adjust_hsb(img, tint_color="#ff0000", tint_strength=0.5)
    arr = np.array(tinted)
    # Red should increase, green and blue should decrease
    assert arr[0, 0, 0] > 128
    assert arr[0, 0, 1] < 128
    assert arr[0, 0, 2] < 128


def test_preserve_alpha():
    data = np.array([
        [[200, 100, 50, 0], [200, 100, 50, 128]],
        [[200, 100, 50, 255], [200, 100, 50, 255]],
    ], dtype=np.uint8)
    img = Image.fromarray(data, mode="RGBA")

    adjusted = adjust_hsb(img, hue_shift=90.0, saturation=1.5, brightness=1.2, preserve_alpha=True)
    arr = np.array(adjusted)

    # Fully transparent pixel must remain untouched (RGB + A)
    np.testing.assert_array_equal(arr[0, 0], data[0, 0])

    # Alpha channel of semi-transparent pixel preserved
    assert arr[0, 1, 3] == 128


def test_cli_execution_and_variants(tmp_path):
    in_file = tmp_path / "test_tex.png"
    img = Image.new("RGBA", (16, 16), (150, 100, 50, 255))
    img.save(in_file)

    out_dir = tmp_path / "out"

    # Run CLI with 3 variants stepped in hue and brightness
    ret = main([
        str(in_file),
        "-n", "3",
        "-d", str(out_dir),
        "--hue", "30",
        "--hue-step", "45",
        "--brightness-step", "-0.1",
        "--naming", "{name}_h{hue}_v{i}{ext}",
    ])
    assert ret == 0

    files = list(out_dir.glob("*.png"))
    assert len(files) == 3
    # Check that all 3 files exist and are valid PNG images
    for f in files:
        loaded = Image.open(f)
        assert loaded.size == (16, 16)


def test_cli_color_base(tmp_path):
    out_file = tmp_path / "from_color.png"
    ret = main([
        "--color", "teal",
        "--size", "16x16",
        "--hue", "60",
        "-o", str(out_file),
    ])
    assert ret == 0
    assert out_file.is_file()
    loaded = Image.open(out_file)
    assert loaded.size == (16, 16)

