import pytest
import numpy as np

from server.blocks import (
    BLOCKS,
    GLASS_CTM_UVS,
    GLASS_CTM_RECTS,
    FACE_UV_VECTORS,
    get_glass_ctm_uv,
    is_glass,
    is_slab,
    is_vertical_slab,
)
from server.items import ITEMS
from server.lighting import get_block_opacity
from server.mesh import (
    _get_glass_face_ctm_mask,
    _is_glass_face_culled,
    _should_cull_glass_element_face,
    build_chunk_mesh,
)
from server.recipes import RECIPES
from server.world import World


def test_glass_registration():
    assert "glass" in BLOCKS
    glass = BLOCKS["glass"]
    assert glass.solid is True
    assert glass.cutout is True
    assert glass.full_cube is True
    assert is_glass("glass")
    assert is_glass(glass)

    # Glass slab
    assert "glass_slab" in BLOCKS
    slab = BLOCKS["glass_slab"]
    assert slab.solid is True
    assert slab.cutout is True
    assert is_slab("glass_slab")
    assert is_glass("glass_slab")

    # Glass top slab
    assert "glass_slab_top" in BLOCKS
    slab_top = BLOCKS["glass_slab_top"]
    assert slab_top.cutout is True
    assert is_glass("glass_slab_top")

    # Double glass slab
    assert "double_slab_glass_slab" in BLOCKS
    double_slab = BLOCKS["double_slab_glass_slab"]
    assert double_slab.cutout is True
    assert is_glass("double_slab_glass_slab")

    # Glass vertical slab
    assert "glass_vertical_slab" in BLOCKS
    v_slab = BLOCKS["glass_vertical_slab"]
    assert v_slab.solid is True
    assert v_slab.cutout is True
    assert is_vertical_slab("glass_vertical_slab")
    assert is_glass("glass_vertical_slab")

    # Items
    assert "glass" in ITEMS
    assert "glass_slab" in ITEMS
    assert "glass_vertical_slab" in ITEMS


def test_glass_light_opacity():
    assert get_block_opacity(BLOCKS["glass"]) == 0
    assert get_block_opacity(BLOCKS["glass_slab"]) == 0
    assert get_block_opacity(BLOCKS["glass_slab_top"]) == 0
    assert get_block_opacity(BLOCKS["double_slab_glass_slab"]) == 0
    assert get_block_opacity(BLOCKS["glass_vertical_slab"]) == 0


def test_glass_ctm_textures_loaded():
    assert len(GLASS_CTM_UVS) == 16
    assert len(GLASS_CTM_RECTS) == 16
    for m in range(16):
        uv = get_glass_ctm_uv(m)
        assert uv is not None
        assert len(uv) == 4
        # Each vertex has (u, v) in 0..1 range
        for u, v in uv:
            assert 0.0 <= u <= 1.0
            assert 0.0 <= v <= 1.0


def make_world():
    world = World(seed=123)
    world.load_chunk(0, 0, 0)
    world.chunks[(0, 0, 0)].data.fill(0)
    return world


def test_glass_isolated_ctm():
    world = make_world()
    world.set_block(5, 5, 5, "glass")

    # Isolated glass block: all 6 faces should have mask 0
    for normal_i in range(6):
        mask = _get_glass_face_ctm_mask(world, 5, 5, 5, normal_i, block=BLOCKS["glass"])
        assert mask == 0, f"Face {normal_i} of isolated glass should have CTM mask 0"


def test_glass_horizontal_connection():
    world = make_world()
    # Place two glass blocks along X axis at (5, 5, 5) and (6, 5, 5)
    world.set_block(5, 5, 5, "glass")
    world.set_block(6, 5, 5, "glass")

    # Contact faces between them must be culled
    # Block at 5 facing East (+X, normal 0) touches block at 6
    assert _is_glass_face_culled(world, 5, 5, 5, 0) is True
    # Block at 6 facing West (-X, normal 1) touches block at 5
    assert _is_glass_face_culled(world, 6, 5, 5, 1) is True

    # South face (+Z, normal 4):
    # For normal 4 (+Z): Right is +X (u_vec=(1,0,0)), Left is -X, Up is +Y (v_vec=(0,1,0))
    # Block at (5, 5, 5): Right is (6, 5, 5) which is glass -> Right connected (bit 3 = 8)
    mask_5_south = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=BLOCKS["glass"])
    assert mask_5_south == 8, f"Expected mask 8 (Right connected), got {mask_5_south}"

    # Block at (6, 5, 5): Left is (5, 5, 5) which is glass -> Left connected (bit 2 = 4)
    mask_6_south = _get_glass_face_ctm_mask(world, 6, 5, 5, 4, block=BLOCKS["glass"])
    assert mask_6_south == 4, f"Expected mask 4 (Left connected), got {mask_6_south}"


def test_glass_2x2_window_ctm():
    world = make_world()
    # 2x2 window on XY plane at Z=5:
    # (5, 6, 5) [Top-Left]     (6, 6, 5) [Top-Right]
    # (5, 5, 5) [Bottom-Left]  (6, 5, 5) [Bottom-Right]
    world.set_block(5, 5, 5, "glass")
    world.set_block(6, 5, 5, "glass")
    world.set_block(5, 6, 5, "glass")
    world.set_block(6, 6, 5, "glass")

    # Check South face (+Z, normal 4):
    # Up is +Y (bit 0 = 1), Down is -Y (bit 1 = 2), Left is -X (bit 2 = 4), Right is +X (bit 3 = 8)

    # Bottom-Left (5, 5, 5): Up and Right are connected -> bit 0 (1) | bit 3 (8) = 9
    m_bl = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=BLOCKS["glass"])
    assert m_bl == 9, f"Bottom-Left expected 9, got {m_bl}"

    # Bottom-Right (6, 5, 5): Up and Left are connected -> bit 0 (1) | bit 2 (4) = 5
    m_br = _get_glass_face_ctm_mask(world, 6, 5, 5, 4, block=BLOCKS["glass"])
    assert m_br == 5, f"Bottom-Right expected 5, got {m_br}"

    # Top-Left (5, 6, 5): Down and Right are connected -> bit 1 (2) | bit 3 (8) = 10
    m_tl = _get_glass_face_ctm_mask(world, 5, 6, 5, 4, block=BLOCKS["glass"])
    assert m_tl == 10, f"Top-Left expected 10, got {m_tl}"

    # Top-Right (6, 6, 5): Down and Left are connected -> bit 1 (2) | bit 2 (4) = 6
    m_tr = _get_glass_face_ctm_mask(world, 6, 6, 5, 4, block=BLOCKS["glass"])
    assert m_tr == 6, f"Top-Right expected 6, got {m_tr}"


def test_glass_3x3_window_center():
    world = make_world()
    # 3x3 window in XY plane
    for x in range(3):
        for y in range(3):
            world.set_block(5 + x, 5 + y, 5, "glass")

    # Center block at (6, 6, 5) should have all 4 neighbors connected:
    # 1 (Up) | 2 (Down) | 4 (Left) | 8 (Right) = 15
    center_mask = _get_glass_face_ctm_mask(world, 6, 6, 5, 4, block=BLOCKS["glass"])
    assert center_mask == 15, f"Center block expected mask 15 (all connected), got {center_mask}"


def test_glass_slab_and_block_connection():
    world = make_world()
    world.set_block(5, 5, 5, "glass")
    world.set_block(5, 6, 5, "glass_slab")  # Bottom slab at y=6

    # Full glass top face (facing +Y, normal 2) touches bottom slab bottom face
    assert _is_glass_face_culled(world, 5, 5, 5, 2) is True

    # Slabs cull their bottom face against glass below
    slab_el = BLOCKS["glass_slab"].elements[0]
    assert _should_cull_glass_element_face(world, 5, 6, 5, BLOCKS["glass_slab"], slab_el, 3) is True

    # South face of full glass at y=5 should connect Up to the slab
    m_glass_south = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=BLOCKS["glass"])
    assert (m_glass_south & 1) != 0, "Full glass south face should connect Up to slab"


def test_glass_mesh_generation_and_culling():
    world = make_world()
    chunk = world.chunks[(0, 0, 0)]

    # 1. Isolated glass block: 6 faces = 24 vertices
    world.set_block(5, 5, 5, "glass")
    mesh1 = build_chunk_mesh(world, 0, 0, 0)
    assert mesh1 is not None and mesh1.cutout is not None
    assert len(mesh1.cutout[0]) == 6 * 4

    # 2. Two adjacent glass blocks: 10 faces = 40 vertices (2 contact faces culled)
    world.set_block(6, 5, 5, "glass")
    mesh2 = build_chunk_mesh(world, 0, 0, 0)
    assert len(mesh2.cutout[0]) == 10 * 4

    # 3. 2x2x2 cube of glass (8 blocks): 24 exterior faces = 96 vertices (all 24 internal faces culled)
    chunk.data.fill(0)
    for x in (5, 6):
        for y in (5, 6):
            for z in (5, 6):
                world.set_block(x, y, z, "glass")
    mesh3 = build_chunk_mesh(world, 0, 0, 0)
    assert len(mesh3.cutout[0]) == 24 * 4

    # 4. Glass block + bottom glass slab: 10 faces = 40 vertices (contact face culled)
    chunk.data.fill(0)
    world.set_block(5, 5, 5, "glass")
    world.set_block(5, 6, 5, "glass_slab")
    mesh4 = build_chunk_mesh(world, 0, 0, 0)
    assert len(mesh4.cutout[0]) == 10 * 4

    # 5. Full glass + vertical glass slab: 10 faces = 40 vertices (contact face culled)
    chunk.data.fill(0)
    world.set_block(5, 5, 5, "glass")
    world.set_block(5, 5, 6, "glass_vertical_slab_north")
    mesh5 = build_chunk_mesh(world, 0, 0, 0)
    assert len(mesh5.cutout[0]) == 10 * 4


def test_glass_recipes():
    # Smelting sand -> glass
    smelt_recipes = [r for r in RECIPES if getattr(r, "type", "") == "smelting" and getattr(r, "result", "") == "glass"]
    assert len(smelt_recipes) >= 1

    # Crafting glass -> glass_slab
    slab_recipes = [r for r in RECIPES if getattr(r, "result", "") == "glass_slab"]
    assert len(slab_recipes) >= 1

    # Crafting glass_slab -> glass_vertical_slab
    vslab_recipes = [r for r in RECIPES if getattr(r, "result", "") == "glass_vertical_slab"]
    assert len(vslab_recipes) >= 1


def test_mixed_slab_mesh_separation():
    world = make_world()
    chunk = world.chunks[(0, 0, 0)]

    # 1. Horizontal mixed double slab: stone_slab + glass_slab
    world.set_block(5, 5, 5, "double_slab_stone_slab_glass_slab")
    mesh_h = build_chunk_mesh(world, 0, 0, 0)
    assert mesh_h is not None
    assert mesh_h.vertices is not None and len(mesh_h.vertices) > 0
    assert mesh_h.cutout is not None and len(mesh_h.cutout[0]) > 0

    opaque_uvs_h = mesh_h.vertices[:, 6:8]
    cutout_uvs_h = mesh_h.cutout[0][:, 6:8]

    # Stone UVs are in 0.18..0.25; Glass CTM UVs are in 0.75..0.90
    assert opaque_uvs_h[:, 0].max() < 0.5
    assert cutout_uvs_h[:, 0].min() > 0.5

    # Top face of stone slab (normal +Y) must be present so stone is visible through glass
    opaque_normals = mesh_h.vertices[:, 3:6]
    assert any(np.allclose(n, (0, 1, 0)) for n in opaque_normals)

    # 2. Vertical mixed double slab: stone_vertical_slab + glass_vertical_slab
    chunk.data.fill(0)
    world.set_block(5, 5, 5, "double_vertical_slab_z_stone_vertical_slab_glass_vertical_slab")
    mesh_v = build_chunk_mesh(world, 0, 0, 0)
    assert mesh_v is not None
    assert mesh_v.vertices is not None and len(mesh_v.vertices) > 0
    assert mesh_v.cutout is not None and len(mesh_v.cutout[0]) > 0

    opaque_uvs_v = mesh_v.vertices[:, 6:8]
    cutout_uvs_v = mesh_v.cutout[0][:, 6:8]

    assert opaque_uvs_v[:, 0].max() < 0.5
    assert cutout_uvs_v[:, 0].min() > 0.5
