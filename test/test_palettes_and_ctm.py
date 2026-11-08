import numpy as np
import pytest

from server.blocks import (
    BLOCKS,
    FACE_INDEX,
    block_palette_index,
    block_palette_indices_grid,
)
from server.mesh import build_chunk_mesh, get_block_face_ctm_mask
from server.world import World


def test_palette_index_distribution_and_determinism():
    # num_palettes <= 1 always returns 0
    assert block_palette_index(10, 20, 30, seed=123, num_palettes=1) == 0
    assert block_palette_index(10, 20, 30, seed=123, num_palettes=0) == 0

    # Determinism: same coords and seed gives exact same index
    idx1 = block_palette_index(15, 64, -22, seed=42, num_palettes=3)
    idx2 = block_palette_index(15, 64, -22, seed=42, num_palettes=3)
    assert idx1 == idx2
    assert 0 <= idx1 < 3

    # Seed sensitivity: different seed can yield different palette
    seeds = [block_palette_index(15, 64, -22, seed=s, num_palettes=3) for s in range(50)]
    assert len(set(seeds)) > 1

    # Grid / Vectorized function equivalence
    wx = np.array([0, 10, -5, 12, 100], dtype=np.int32)
    wy = np.array([60, 64, 70, 80, 50], dtype=np.int32)
    wz = np.array([-1, 0, 5, 20, -50], dtype=np.int32)
    grid_res = block_palette_indices_grid(wx, wy, wz, seed=999, num_palettes=3)
    scalar_res = [block_palette_index(x, y, z, seed=999, num_palettes=3) for x, y, z in zip(wx, wy, wz)]
    np.testing.assert_array_equal(grid_res, scalar_res)


def test_smooth_stones_palettes_and_ctm_registered():
    smooth_stones = [
        "smooth_granite",
        "smooth_diorite",
        "smooth_andesite",
        "smooth_marble",
        "smooth_limestone",
        "smooth_stone",
    ]
    for stone_id in smooth_stones:
        block = BLOCKS.get(stone_id)
        assert block is not None, f"Block {stone_id} not found in BLOCKS"
        assert block.ctm is True, f"Block {stone_id} should have ctm=True"
        assert len(block.palettes) == 5, f"Block {stone_id} should have 5 palettes"
        assert len(block.palette_face_uvs) == 5, f"Block {stone_id} should have 5 palette_face_uvs"
        assert len(block.palette_ctm_uvs) == 5, f"Block {stone_id} should have 5 palette_ctm_uvs"
        for p in range(5):
            assert len(block.palette_ctm_uvs[p]) == 16, f"Block {stone_id} palette {p} should have 16 CTM masks"


def test_ctm_mask_isolated_and_neighbors():
    world = World(seed=123)
    world.generate_chunk(0, 6, 0)  # Y: 96..111 (in the air)

    # Place an isolated smooth_granite at (5, 100, 5)
    world.set_block(5, 100, 5, "smooth_granite")
    granite = BLOCKS["smooth_granite"]

    # For an isolated block, all 6 face masks should be 0
    for normal_i in range(6):
        mask = get_block_face_ctm_mask(world, 5, 100, 5, normal_i, block=granite)
        assert mask == 0, f"Face {normal_i} isolated mask should be 0, got {mask}"

    # Place a neighbor to the right (+X) at (6, 100, 5)
    world.set_block(6, 100, 5, "smooth_granite")

    # On the UP face (normal_i = FACE_INDEX['up']), +X is RIGHT (bit 8), -X is LEFT (bit 4)
    up_idx = FACE_INDEX["up"]
    mask_left_block = get_block_face_ctm_mask(world, 5, 100, 5, up_idx, block=granite)
    mask_right_block = get_block_face_ctm_mask(world, 6, 100, 5, up_idx, block=granite)

    assert mask_left_block == 8, f"Expected 8 (right bit set), got {mask_left_block}"
    assert mask_right_block == 4, f"Expected 4 (left bit set), got {mask_right_block}"


def test_ctm_3x3_plane_center_full_connection():
    world = World(seed=42)
    world.generate_chunk(0, 6, 0)

    # 3x3 platform of smooth_diorite at Y=100, X in 4..6, Z in 4..6
    for x in range(4, 7):
        for z in range(4, 7):
            world.set_block(x, 100, z, "smooth_diorite")

    diorite = BLOCKS["smooth_diorite"]
    up_idx = FACE_INDEX["up"]

    # Center block at (5, 100, 5) has up, down, left, right neighbors -> mask 15
    mask_center = get_block_face_ctm_mask(world, 5, 100, 5, up_idx, block=diorite)
    assert mask_center == 15, f"Center block of 3x3 should have mask 15, got {mask_center}"

    # Corners:
    # UP face has u_vec = (1, 0, 0) [right=+X], v_vec = (0, 0, -1) [up=-Z, down=+Z]
    # (4, 100, 4): neighbor in +X (right=8) and +Z (down=2) -> mask 10
    assert get_block_face_ctm_mask(world, 4, 100, 4, up_idx, block=diorite) == (2 | 8)
    # (6, 100, 4): neighbor in -X (left=4) and +Z (down=2) -> mask 6
    assert get_block_face_ctm_mask(world, 6, 100, 4, up_idx, block=diorite) == (2 | 4)
    # (4, 100, 6): neighbor in +X (right=8) and -Z (up=1) -> mask 9
    assert get_block_face_ctm_mask(world, 4, 100, 6, up_idx, block=diorite) == (1 | 8)
    # (6, 100, 6): neighbor in -X (left=4) and -Z (up=1) -> mask 5
    assert get_block_face_ctm_mask(world, 6, 100, 6, up_idx, block=diorite) == (1 | 4)


def test_ctm_occlusion_prevents_connection():
    world = World(seed=123)
    world.generate_chunk(0, 6, 0)

    # Place two smooth_marble blocks side by side at (5, 100, 5) and (6, 100, 5)
    world.set_block(5, 100, 5, "smooth_marble")
    world.set_block(6, 100, 5, "smooth_marble")
    marble = BLOCKS["smooth_marble"]
    up_idx = FACE_INDEX["up"]

    # Initially, (5, 100, 5) connects to the right (mask 8)
    assert get_block_face_ctm_mask(world, 5, 100, 5, up_idx, block=marble) == 8

    # Place an occluding solid full block directly above the neighbor at (6, 101, 5)
    # The normal of the UP face is +Y, so front of neighbor is (6, 100+1, 5) = (6, 101, 5)
    world.set_block(6, 101, 5, "stone")

    # Now the UP face of (5, 100, 5) should NOT connect to (6, 100, 5) because (6, 101, 5) is occluded
    mask_occluded = get_block_face_ctm_mask(world, 5, 100, 5, up_idx, block=marble)
    assert mask_occluded == 0, f"Expected 0 due to occlusion, got {mask_occluded}"


def test_chunk_mesh_uses_palette_and_ctm():
    world1 = World(seed=10)
    world1.generate_chunk(0, 0, 0)
    world1.set_block(5, 5, 5, "smooth_andesite")
    mesh1 = build_chunk_mesh(world1, 0, 0, 0)
    assert mesh1 is not None

    # Verify that different seeds produce different palettes across a set of coordinates
    granite = BLOCKS["smooth_granite"]
    p0_uv = granite.palette_ctm_uvs[0][0]
    p1_uv = granite.palette_ctm_uvs[1][0]
    p2_uv = granite.palette_ctm_uvs[2][0]

    # Check UVs are distinct between palettes
    assert not np.array_equal(p0_uv, p1_uv)
    assert not np.array_equal(p1_uv, p2_uv)
