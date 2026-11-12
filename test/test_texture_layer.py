import numpy as np
import pytest
from PIL import Image

from tooling.texture_layer import (
    blend_arrays,
    build_parser,
    layer_images,
    main,
    parse_offset,
    parse_opacity,
    resolve_output_path,
)


def test_parse_offset():
    assert parse_offset("0,0") == (0, 0)
    assert parse_offset("4,8") == (4, 8)
    assert parse_offset("2x6") == (2, 6)
    assert parse_offset("5") == (5, 5)

    with pytest.raises(ValueError):
        parse_offset("1,2,3")


def test_parse_opacity():
    assert parse_opacity(1.0) == 1.0
    assert parse_opacity("0.75") == 0.75
    assert parse_opacity("50%") == 0.5
    assert parse_opacity("120%") == 1.0
    assert parse_opacity("-10%") == 0.0


def test_normal_alpha_composite():
    # Solid blue (0, 0, 255) base
    base = np.full((16, 16, 4), [0, 0, 255, 255], dtype=np.uint8)
    # Solid red (255, 0, 0) top with 50% alpha (128)
    top = np.full((16, 16, 4), [255, 0, 0, 128], dtype=np.uint8)

    blended = blend_arrays(base, top, mode="normal", opacity=1.0)

    # Compare with PIL native alpha_composite
    pil_base = Image.fromarray(base, mode="RGBA")
    pil_top = Image.fromarray(top, mode="RGBA")
    expected = np.array(Image.alpha_composite(pil_base, pil_top))

    np.testing.assert_array_equal(blended, expected)
    # Blended pixel should be purple (both red and blue present)
    assert blended[0, 0, 0] > 100
    assert blended[0, 0, 2] > 100
    assert blended[0, 0, 3] == 255


def test_multiply_blend():
    # White base (255, 255, 255)
    base = np.full((4, 4, 4), 255, dtype=np.uint8)
    # Red top (255, 0, 0)
    top = np.zeros((4, 4, 4), dtype=np.uint8)
    top[..., 0] = 255
    top[..., 3] = 255

    blended = blend_arrays(base, top, mode="multiply")
    # White * Red = Red
    np.testing.assert_array_equal(blended[..., :3], top[..., :3])


def test_screen_blend():
    # Black base (0, 0, 0)
    base = np.zeros((4, 4, 4), dtype=np.uint8)
    base[..., 3] = 255
    # Green top (0, 255, 0)
    top = np.zeros((4, 4, 4), dtype=np.uint8)
    top[..., 1] = 255
    top[..., 3] = 255

    blended = blend_arrays(base, top, mode="screen")
    # Screen with black yields top color
    np.testing.assert_array_equal(blended[..., :3], top[..., :3])


def test_add_blend():
    base = np.full((4, 4, 4), [100, 50, 0, 255], dtype=np.uint8)
    top = np.full((4, 4, 4), [50, 50, 50, 255], dtype=np.uint8)

    blended = blend_arrays(base, top, mode="add")
    np.testing.assert_array_equal(blended[0, 0, :3], [150, 100, 50])


def test_opacity_scaling():
    base = np.full((4, 4, 4), [0, 0, 0, 255], dtype=np.uint8)
    top = np.full((4, 4, 4), [200, 200, 200, 255], dtype=np.uint8)

    # 50% opacity
    half = blend_arrays(base, top, mode="normal", opacity=0.5)
    assert 90 <= half[0, 0, 0] <= 110


def test_offset_placement():
    base = np.zeros((16, 16, 4), dtype=np.uint8)
    base[..., 3] = 255  # Solid black 16x16

    # 4x4 red stamp
    stamp = np.full((4, 4, 4), [255, 0, 0, 255], dtype=np.uint8)

    # Place stamp at offset (6, 6)
    blended = blend_arrays(base, stamp, offset=(6, 6))

    # Corner should still be black
    np.testing.assert_array_equal(blended[0, 0, :3], [0, 0, 0])
    # Stamp position should be red
    np.testing.assert_array_equal(blended[7, 7, :3], [255, 0, 0])


def test_multi_layer_stack():
    img1 = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    img2 = Image.new("RGBA", (8, 8), (0, 255, 0, 128))
    img3 = Image.new("RGBA", (8, 8), (0, 0, 255, 64))

    stacked = layer_images([img1, img2, img3])
    assert stacked.size == (8, 8)
    arr = np.array(stacked)
    assert arr[0, 0, 3] == 255


def test_cli_positional_stack(tmp_path):
    base_p = tmp_path / "base.png"
    top_p = tmp_path / "top.png"
    out_p = tmp_path / "out.png"

    Image.new("RGBA", (16, 16), (100, 100, 100, 255)).save(base_p)
    Image.new("RGBA", (16, 16), (200, 0, 0, 128)).save(top_p)

    ret = main([str(base_p), str(top_p), "-o", str(out_p)])
    assert ret == 0
    assert out_p.is_file()

    loaded = Image.open(out_p)
    assert loaded.size == (16, 16)


def test_cli_batch_mode(tmp_path):
    base1 = tmp_path / "base1.png"
    base2 = tmp_path / "base2.png"
    overlay = tmp_path / "overlay.png"
    out_dir = tmp_path / "batch_out"

    Image.new("RGBA", (16, 16), (50, 50, 50, 255)).save(base1)
    Image.new("RGBA", (16, 16), (100, 100, 100, 255)).save(base2)
    Image.new("RGBA", (16, 16), (255, 0, 0, 100)).save(overlay)

    ret = main([
        "-b", str(base1),
        "-b", str(base2),
        "-l", str(overlay),
        "-d", str(out_dir),
        "--naming", "{base}_ore{ext}",
    ])
    assert ret == 0
    assert (out_dir / "base1_ore.png").is_file()
    assert (out_dir / "base2_ore.png").is_file()


def test_cli_pairwise_mode(tmp_path):
    b1 = tmp_path / "b1.png"
    b2 = tmp_path / "b2.png"
    o1 = tmp_path / "o1.png"
    o2 = tmp_path / "o2.png"
    out_dir = tmp_path / "pair_out"

    Image.new("RGBA", (8, 8), (10, 10, 10, 255)).save(b1)
    Image.new("RGBA", (8, 8), (20, 20, 20, 255)).save(b2)
    Image.new("RGBA", (8, 8), (100, 0, 0, 255)).save(o1)
    Image.new("RGBA", (8, 8), (0, 100, 0, 255)).save(o2)

    ret = main([
        "-b", str(b1), "-b", str(b2),
        "-l", str(o1), "-l", str(o2),
        "--pair",
        "-d", str(out_dir),
    ])
    assert ret == 0
    assert (out_dir / "b1_o1.png").is_file()
    assert (out_dir / "b2_o2.png").is_file()

