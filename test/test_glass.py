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


def test_double_glass_slab_ctm_connection():
    world = make_world()
    world.set_block(5, 5, 5, "double_slab_glass_slab")
    b = BLOCKS["double_slab_glass_slab"]
    el0 = b.elements[0]  # bottom slab
    el1 = b.elements[1]  # top slab

    # On South face (+Z, normal 4):
    # Bottom half should connect Up to top half (bit 0 = 1)
    m0_south = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=b, element=el0)
    assert (m0_south & 1) != 0, f"Double slab bottom half south should connect UP, got mask {m0_south}"

    # Top half should connect Down to bottom half (bit 1 = 2)
    m1_south = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=b, element=el1)
    assert (m1_south & 2) != 0, f"Double slab top half south should connect DOWN, got mask {m1_south}"

    # Double vertical glass slab:
    world.set_block(5, 5, 5, "double_vertical_slab_z_glass_vertical_slab_glass_vertical_slab")
    bv = BLOCKS["double_vertical_slab_z_glass_vertical_slab_glass_vertical_slab"]
    v_el0 = bv.elements[0]
    v_el1 = bv.elements[1]

    # On East face (+X, normal 0):
    # Right is -Z (u_vec=(0, 0, -1)), Left is +Z
    m_v0 = _get_glass_face_ctm_mask(world, 5, 5, 5, 0, block=bv, element=v_el0)
    m_v1 = _get_glass_face_ctm_mask(world, 5, 5, 5, 0, block=bv, element=v_el1)
    # The two vertical halves must connect to each other horizontally (one Right=8, one Left=4)
    assert (m_v0 & 8) != 0, f"Double vertical slab el 0 east should connect right to el 1, got {m_v0}"
    assert (m_v1 & 4) != 0, f"Double vertical slab el 1 east should connect left to el 0, got {m_v1}"


def test_full_glass_and_mixed_double_slab():
    world = make_world()
    # 1. Horizontal mixed double slab: stone bottom, glass top at (5, 5, 5)
    # Full glass at (4, 5, 5)
    world.set_block(4, 5, 5, "glass")
    world.set_block(5, 5, 5, "double_slab_stone_slab_glass_slab")
    bm = BLOCKS["double_slab_stone_slab_glass_slab"]
    glass_top = bm.elements[1]

    # Top glass half South face should connect Left (bit 2 = 4) to full glass
    m_top_south = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=bm, element=glass_top)
    assert (m_top_south & 4) != 0, f"Mixed slab glass top south should connect left to full glass, got {m_top_south}"

    # West face of top glass half (normal 1) touches full glass -> culled
    assert _should_cull_glass_element_face(world, 5, 5, 5, bm, glass_top, 1) is True

    # 2. Vertical mixed double slab: glass negative half, stone positive half at (5, 5, 6)
    # Full glass at (5, 5, 5)
    world.chunks[(0, 0, 0)].data.fill(0)
    world.set_block(5, 5, 5, "glass")
    world.set_block(5, 5, 6, "double_vertical_slab_z_stone_vertical_slab_glass_vertical_slab")
    # Positive half (z in [6.5, 7.0]) is glass; let's test with glass touching full glass at z=5.0
    world.set_block(5, 5, 4, "double_vertical_slab_z_glass_vertical_slab_stone_vertical_slab")
    bm_v = BLOCKS["double_vertical_slab_z_glass_vertical_slab_stone_vertical_slab"]
    glass_v_el = bm_v.elements[0]  # positive half is glass, z in [4.5, 5.0]

    # Glass half South face (normal 4) touches full glass at (5, 5, 5) -> culled
    assert _should_cull_glass_element_face(world, 5, 5, 4, bm_v, glass_v_el, 4) is True


def test_offset_glass_culling_and_ctm():
    world = make_world()
    # 1. Two adjacent offset glass blocks at (5, 5, 5) and (6, 5, 5)
    world.set_block(5, 5, 5, "glass_offset")
    world.set_block(6, 5, 5, "glass_offset")
    mesh = build_chunk_mesh(world, 0, 0, 0)
    assert mesh is not None and mesh.cutout is not None
    # 2 blocks * 6 faces - 2 contact faces = 10 faces = 40 vertices
    assert len(mesh.cutout[0]) == 40, f"Expected 40 vertices for 2 offset glass, got {len(mesh.cutout[0])}"

    b = BLOCKS["glass_offset"]
    el = b.elements[0]
    m5 = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=b, element=el)
    m6 = _get_glass_face_ctm_mask(world, 6, 5, 5, 4, block=b, element=el)
    assert (m5 & 8) != 0, f"Offset glass 5 south should connect right, got {m5}"
    assert (m6 & 4) != 0, f"Offset glass 6 south should connect left, got {m6}"

    # 2. Vertically stacked offset glass blocks at (5, 5, 5) and (5, 6, 5)
    world.chunks[(0, 0, 0)].data.fill(0)
    world.set_block(5, 5, 5, "glass_offset")
    world.set_block(5, 6, 5, "glass_offset")
    mesh_v = build_chunk_mesh(world, 0, 0, 0)
    assert mesh_v is not None and mesh_v.cutout is not None
    assert len(mesh_v.cutout[0]) == 40, f"Expected 40 vertices for vertical offset glass, got {len(mesh_v.cutout[0])}"

    m_bot = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=b, element=el)
    m_top = _get_glass_face_ctm_mask(world, 5, 6, 5, 4, block=b, element=el)
    assert (m_bot & 1) != 0, f"Lower offset glass south should connect up, got {m_bot}"
    assert (m_top & 2) != 0, f"Upper offset glass south should connect down, got {m_top}"

    # 3. Capped offset glass: bottom offset connects up to top cap
    world.chunks[(0, 0, 0)].data.fill(0)
    world.set_block(5, 5, 5, "glass_offset_glass_slab")
    b_cap = BLOCKS["glass_offset_glass_slab"]
    el_bot = b_cap.elements[0]
    el_top = b_cap.elements[1]
    m_cap_bot = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=b_cap, element=el_bot)
    m_cap_top = _get_glass_face_ctm_mask(world, 5, 5, 5, 4, block=b_cap, element=el_top)
    assert (m_cap_bot & 1) != 0, f"Capped offset glass bottom should connect up, got {m_cap_bot}"
    assert (m_cap_top & 2) != 0, f"Capped offset glass top should connect down, got {m_cap_top}"


def test_full_glass_and_vertical_slab_connection():
    world = make_world()
    world.set_block(5, 5, 5, "glass")
    world.set_block(5, 5, 6, "glass_vertical_slab_north")

    # Full glass South face (normal 4) touches vertical slab North face (normal 5)
    assert _is_glass_face_culled(world, 5, 5, 5, 4) is True
    vs_el = BLOCKS["glass_vertical_slab_north"].elements[0]
    assert _should_cull_glass_element_face(world, 5, 5, 6, BLOCKS["glass_vertical_slab_north"], vs_el, 5) is True

    # Side faces: East face (normal 0):
    # For East face (+X), Right is -Z (u_vec=(0, 0, -1)), Left is +Z ((-u_vec)=(0, 0, 1))
    # Block at 5: Left (+Z) is (5, 5, 6) which is glass -> Left connected (bit 2 = 4)
    m_east_5 = _get_glass_face_ctm_mask(world, 5, 5, 5, 0, block=BLOCKS["glass"])
    assert (m_east_5 & 4) != 0, f"Full glass east face should connect left (+Z) to vertical slab, got {m_east_5}"

    # Vertical slab at (5, 5, 6): Right (-Z) is (5, 5, 5) which is glass -> Right connected (bit 3 = 8)
    m_east_6 = _get_glass_face_ctm_mask(world, 5, 5, 6, 0, block=BLOCKS["glass_vertical_slab_north"], element=vs_el)
    assert (m_east_6 & 8) != 0, f"Vertical slab east face should connect right (-Z) to full glass, got {m_east_6}"


def test_offset_by_vertical_slab_ctm_and_culling():
    from server.sub_cell import calculate_horizontal_offset_placement

    world = World(seed=1)
    world.load_chunk(0, 5, 0)
    world.chunks[(0, 5, 0)].data.fill(0)

    # Place two cobble vertical slabs side-by-side facing North at (5, 80, 5) and (6, 80, 5)
    world.set_block(5, 80, 5, "cobble_vertical_slab_north")
    world.set_block(6, 80, 5, "cobble_vertical_slab_north")

    # Place glass block horizontally offset against (5, 80, 5) South face
    hit1 = {"x": 5, "y": 80, "z": 5, "nx": 0, "ny": 0, "nz": 1, "local_y": 0.5}
    res1 = calculate_horizontal_offset_placement(world, hit1, "glass")
    assert res1 is not None
    for x, y, z, sc in res1:
        world.set_sub_cell(x, y, z, sc)

    # Place glass block horizontally offset against (6, 80, 5) South face
    hit2 = {"x": 6, "y": 80, "z": 5, "nx": 0, "ny": 0, "nz": 1, "local_y": 0.5}
    res2 = calculate_horizontal_offset_placement(world, hit2, "glass")
    assert res2 is not None
    for x, y, z, sc in res2:
        world.set_sub_cell(x, y, z, sc)

    # Check that world elements exist for subcell glass
    els_1 = _get_world_elements = None  # just testing mesh directly
    mesh = build_chunk_mesh(world, 0, 5, 0)
    assert mesh is not None and mesh.cutout is not None
    verts, inds = mesh.cutout

    # Find quads by normal and verify CTM masks
    # Block 1 South face (norm = 0, 0, 1; x in [5, 6], z = 6.5)
    # Block 2 South face (norm = 0, 0, 1; x in [6, 7], z = 6.5)
    south_uvs_b1 = []
    south_uvs_b2 = []
    top_uvs_b1 = []
    top_uvs_b2 = []
    north_quads = 0
    internal_x6_quads = 0

    for quad_idx in range(len(inds) // 6):
        i0 = inds[quad_idx * 6]
        v0 = verts[i0]
        nx, ny, nz = v0[3], v0[4], v0[5]
        px, py, pz = v0[0], v0[1], v0[2]

        if abs(nx) < 1e-4 and abs(ny) < 1e-4 and abs(nz - 1.0) < 1e-4:
            # South face
            u, v = v0[6], v0[7]
            if px < 5.9:
                south_uvs_b1.append((u, v))
            else:
                south_uvs_b2.append((u, v))
        elif abs(nx) < 1e-4 and abs(ny - 1.0) < 1e-4 and abs(nz) < 1e-4:
            # Top face
            u, v = v0[6], v0[7]
            if px < 5.9:
                top_uvs_b1.append((u, v))
            else:
                top_uvs_b2.append((u, v))
        elif abs(nx) < 1e-4 and abs(ny) < 1e-4 and abs(nz - (-1.0)) < 1e-4:
            # North face (against cobble slab)
            north_quads += 1
        elif abs(nx - 1.0) < 1e-4 or abs(nx - (-1.0)) < 1e-4:
            # Check if any face rendered at the internal contact plane x=6.0
            if abs(px - 6.0) < 1e-4:
                internal_x6_quads += 1

    # North face against cobble slab must be culled
    assert north_quads == 0, f"Expected 0 north quads touching vertical slab, got {north_quads}"
    # Internal face between the two offset blocks at x=6.0 must be culled
    assert internal_x6_quads == 0, f"Expected 0 internal quads at x=6.0, got {internal_x6_quads}"

    # Check South face CTM:
    # Block 1 connects Right (mask 8) -> base UVs start at GLASS_CTM_UVS[8][0]
    expected_b1_uv0 = GLASS_CTM_UVS[8][0]
    expected_b2_uv0 = GLASS_CTM_UVS[4][0]
    assert any(abs(u - expected_b1_uv0[0]) < 1e-4 and abs(v - expected_b1_uv0[1]) < 1e-4 for u, v in south_uvs_b1), "Block 1 south face should have mask 8 CTM"
    assert any(abs(u - expected_b2_uv0[0]) < 1e-4 and abs(v - expected_b2_uv0[1]) < 1e-4 for u, v in south_uvs_b2), "Block 2 south face should have mask 4 CTM"

    # Check Top face CTM:
    assert any(abs(u - expected_b1_uv0[0]) < 1e-4 and abs(v - expected_b1_uv0[1]) < 1e-4 for u, v in top_uvs_b1), "Block 1 top face should have mask 8 CTM"
    assert any(abs(u - expected_b2_uv0[0]) < 1e-4 and abs(v - expected_b2_uv0[1]) < 1e-4 for u, v in top_uvs_b2), "Block 2 top face should have mask 4 CTM"


def test_mixed_double_slab_with_glass_ctm_and_culling():
    world = World(seed=1)
    world.load_chunk(0, 0, 0)
    world.chunks[(0, 0, 0)].data.fill(0)

    # (4, 5, 5): glass_slab_top
    # (5, 5, 5): double_slab_pine_planks_slab_glass_slab
    # (5, 6, 5): full glass
    world.set_block(4, 5, 5, "glass_slab_top")
    world.set_block(5, 5, 5, "double_slab_pine_planks_slab_glass_slab")
    world.set_block(5, 6, 5, "glass")

    mesh = build_chunk_mesh(world, 0, 0, 0)
    assert mesh is not None and mesh.cutout is not None
    verts, inds = mesh.cutout

    # Full glass bottom face at Y=6.0 should be culled
    has_full_glass_bottom = False
    # Center top slab top face at Y=6.0 should be culled
    has_center_slab_top = False
    # Contact face at X=5.0 (between left slab and center slab, Y in [5.5, 6.0]) should be culled
    has_contact_x5 = False

    south_quads = {}
    for q in range(len(inds) // 6):
        i0 = inds[q * 6]
        v = verts[i0]
        norm = (round(float(v[3]), 2), round(float(v[4]), 2), round(float(v[5]), 2))
        pos = (round(float(v[0]), 2), round(float(v[1]), 2), round(float(v[2]), 2))
        u, v_coord = round(float(v[6]), 4), round(float(v[7]), 4)

        if norm == (0.0, -1.0, 0.0) and pos[1] == 6.0:
            has_full_glass_bottom = True
        if norm == (0.0, 1.0, 0.0) and pos[1] == 6.0 and pos[0] == 5.0:
            has_center_slab_top = True
        if norm in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)) and pos[0] == 5.0 and 5.4 <= pos[1] < 5.9:
            has_contact_x5 = True
        if norm == (0.0, 0.0, 1.0):
            south_quads[pos] = (u, v_coord)

    assert not has_full_glass_bottom, "Full glass bottom face touching mixed double slab glass should be culled"
    assert not has_center_slab_top, "Mixed double slab glass top face touching full glass should be culled"
    assert not has_contact_x5, "Contact face between left slab and center mixed double slab should be culled"

    # Full glass South face at (5, 6, 6) should connect Down (mask 2)
    expected_full_uv0 = (round(GLASS_CTM_UVS[2][0][0], 4), round(GLASS_CTM_UVS[2][0][1], 4))
    assert south_quads.get((5.0, 6.0, 6.0)) == expected_full_uv0, f"Full glass south face should connect down, got {south_quads.get((5.0, 6.0, 6.0))}"

    # Left slab South face at (4, 5.5, 6) should connect Right (mask 8)
    expected_left_uv0 = (round(get_glass_ctm_uv(8, [0, 0, 16, 8])[0][0], 4), round(get_glass_ctm_uv(8, [0, 0, 16, 8])[0][1], 4))
    assert south_quads.get((4.0, 5.5, 6.0)) == expected_left_uv0, f"Left slab south face should connect right, got {south_quads.get((4.0, 5.5, 6.0))}"

    # Center mixed double slab glass South face at (5, 5.5, 6) should connect Up and Left (mask 5)
    expected_center_uv0 = (round(get_glass_ctm_uv(5, [0, 0, 16, 8])[0][0], 4), round(get_glass_ctm_uv(5, [0, 0, 16, 8])[0][1], 4))
    assert south_quads.get((5.0, 5.5, 6.0)) == expected_center_uv0, f"Center slab south face should connect up+left, got {south_quads.get((5.0, 5.5, 6.0))}"


def test_glass_full_block_adjacent_to_half_slabs_step_ctm_and_culling():
    world = World(seed=1)
    world.load_chunk(0, 0, 0)
    world.chunks[(0, 0, 0)].data.fill(0)

    # (4, 5, 5): glass_slab_top (y in [5.5, 6.0])
    # (5, 5, 5): glass (full block, y in [5.0, 6.0])
    # (6, 5, 5): glass_slab (bottom slab, y in [5.0, 5.5])
    world.set_block(4, 5, 5, "glass_slab_top")
    world.set_block(5, 5, 5, "glass")
    world.set_block(6, 5, 5, "glass_slab")

    mesh = build_chunk_mesh(world, 0, 0, 0)
    assert mesh is not None and mesh.cutout is not None
    verts, inds = mesh.cutout

    # Verify contact face culling:
    # At x=6.0, y in [5.0, 5.5]: full glass touches bottom slab -> must be culled
    # At x=5.0, y in [5.5, 6.0]: full glass touches top slab -> must be culled
    has_contact_x6_bot = False
    has_contact_x5_top = False
    has_exposed_x6_top = False
    has_exposed_x5_bot = False

    south_quads = {}
    for q in range(len(inds) // 6):
        i0 = inds[q * 6]
        v = verts[i0]
        norm = (round(float(v[3]), 2), round(float(v[4]), 2), round(float(v[5]), 2))
        pos = (round(float(v[0]), 2), round(float(v[1]), 2), round(float(v[2]), 2))
        u, v_coord = round(float(v[6]), 4), round(float(v[7]), 4)

        if norm == (1.0, 0.0, 0.0) and pos[0] == 6.0:
            if pos[1] < 5.4:
                has_contact_x6_bot = True
            else:
                has_exposed_x6_top = True
        elif norm == (-1.0, 0.0, 0.0) and pos[0] == 5.0:
            if pos[1] > 5.4:
                has_contact_x5_top = True
            else:
                has_exposed_x5_bot = True
        elif norm == (0.0, 0.0, 1.0):
            south_quads[pos] = (u, v_coord)

    assert not has_contact_x6_bot, "Contact face between full glass and bottom slab at x=6.0 should be culled"
    assert not has_contact_x5_top, "Contact face between full glass and top slab at x=5.0 should be culled"
    assert has_exposed_x6_top, "Exposed face above bottom slab at x=6.0 should be rendered"
    assert has_exposed_x5_bot, "Exposed face below top slab at x=5.0 should be rendered"

    # Verify South face CTM split on full glass block:
    # Bottom half (5, 5.0, 6): connects Up (1) and Right (8) = mask 9
    expected_bot_uv0 = (round(get_glass_ctm_uv(9, [0, 8, 16, 16])[0][0], 4), round(get_glass_ctm_uv(9, [0, 8, 16, 16])[0][1], 4))
    assert south_quads.get((5.0, 5.0, 6.0)) == expected_bot_uv0, f"Full glass bottom half south face should connect up+right (mask 9), got {south_quads.get((5.0, 5.0, 6.0))}"

    # Upper half (5, 5.5, 6): connects Down (2) and Left (4) = mask 6
    expected_top_uv0 = (round(get_glass_ctm_uv(6, [0, 0, 16, 8])[0][0], 4), round(get_glass_ctm_uv(6, [0, 0, 16, 8])[0][1], 4))
    assert south_quads.get((5.0, 5.5, 6.0)) == expected_top_uv0, f"Full glass top half south face should connect down+left (mask 6), got {south_quads.get((5.0, 5.5, 6.0))}"

    # Left slab South face (4, 5.5, 6): connects Right (8)
    expected_left_uv0 = (round(get_glass_ctm_uv(8, [0, 0, 16, 8])[0][0], 4), round(get_glass_ctm_uv(8, [0, 0, 16, 8])[0][1], 4))
    assert south_quads.get((4.0, 5.5, 6.0)) == expected_left_uv0, f"Left slab south face should connect right (mask 8), got {south_quads.get((4.0, 5.5, 6.0))}"

    # Right slab South face (6, 5.0, 6): connects Left (4)
    expected_right_uv0 = (round(get_glass_ctm_uv(4, [0, 8, 16, 16])[0][0], 4), round(get_glass_ctm_uv(4, [0, 8, 16, 16])[0][1], 4))
    assert south_quads.get((6.0, 5.0, 6.0)) == expected_right_uv0, f"Right slab south face should connect left (mask 4), got {south_quads.get((6.0, 5.0, 6.0))}"


def test_glass_full_block_adjacent_to_vertical_slabs_step_ctm_and_culling():
    world = make_world()
    world.chunks[(0, 0, 0)].data.fill(0)

    # (4, 5, 5): glass_vertical_slab_north (z in [5.0, 5.5])
    # (5, 5, 5): glass (full block, z in [5.0, 6.0])
    # (6, 5, 5): glass_vertical_slab (south-facing, z in [5.5, 6.0])
    world.set_block(4, 5, 5, "glass_vertical_slab_north")
    world.set_block(5, 5, 5, "glass")
    world.set_block(6, 5, 5, "glass_vertical_slab")

    mesh = build_chunk_mesh(world, 0, 0, 0)
    assert mesh is not None and mesh.cutout is not None
    verts, inds = mesh.cutout

    # Verify contact face culling:
    # At x=6.0, z in [5.5, 6.0]: full glass touches south vertical slab -> must be culled
    # At x=6.0, z in [5.0, 5.5]: full glass touches air -> must be rendered
    # At x=5.0, z in [5.0, 5.5]: full glass touches north vertical slab -> must be culled
    # At x=5.0, z in [5.5, 6.0]: full glass touches air -> must be rendered
    has_contact_x6_south = False
    has_contact_x5_north = False
    has_exposed_x6_north = False
    has_exposed_x5_south = False

    top_quads = {}
    for q in range(len(inds) // 6):
        i0 = inds[q * 6]
        i1 = inds[q * 6 + 1]
        v = verts[i0]
        v1_pt = verts[i1]
        norm = (round(float(v[3]), 2), round(float(v[4]), 2), round(float(v[5]), 2))
        pos = (round(float(v[0]), 2), round(float(v[1]), 2), round(float(v[2]), 2))
        u, v_coord = round(float(v[6]), 4), round(float(v[7]), 4)
        z_mid = (float(v[2]) + float(v1_pt[2])) * 0.5

        if norm == (1.0, 0.0, 0.0) and pos[0] == 6.0:
            if z_mid > 5.5:
                has_contact_x6_south = True
            else:
                has_exposed_x6_north = True
        elif norm == (-1.0, 0.0, 0.0) and pos[0] == 5.0:
            if z_mid < 5.5:
                has_contact_x5_north = True
            else:
                has_exposed_x5_south = True
        elif norm == (0.0, 1.0, 0.0):
            top_quads[pos] = (u, v_coord)

    assert not has_contact_x6_south, "Contact face between full glass and south vertical slab at x=6.0 should be culled"
    assert not has_contact_x5_north, "Contact face between full glass and north vertical slab at x=5.0 should be culled"
    assert has_exposed_x6_north, "Exposed face on north side at x=6.0 should be rendered"
    assert has_exposed_x5_south, "Exposed face on south side at x=5.0 should be rendered"

    # Verify Top face CTM split on full glass block:
    # North half connects Down (2) and Left (4) = mask 6
    # South half connects Up (1) and Right (8) = mask 9
    expected_north_uv0 = (round(get_glass_ctm_uv(6, [0, 0, 16, 8])[0][0], 4), round(get_glass_ctm_uv(6, [0, 0, 16, 8])[0][1], 4))
    expected_south_uv0 = (round(get_glass_ctm_uv(9, [0, 8, 16, 16])[0][0], 4), round(get_glass_ctm_uv(9, [0, 8, 16, 16])[0][1], 4))

    assert top_quads.get((5.0, 6.0, 5.5)) == expected_north_uv0, f"Full glass north half top face should connect left+down (mask 6), got {top_quads.get((5.0, 6.0, 5.5))}"
    assert top_quads.get((5.0, 6.0, 6.0)) == expected_south_uv0, f"Full glass south half top face should connect right+up (mask 9), got {top_quads.get((5.0, 6.0, 6.0))}"

    # Left vertical slab top face: connects Right (8)
    expected_left_uv0 = (round(get_glass_ctm_uv(8, [0, 0, 16, 8])[0][0], 4), round(get_glass_ctm_uv(8, [0, 0, 16, 8])[0][1], 4))
    assert top_quads.get((4.0, 6.0, 5.5)) == expected_left_uv0, f"Left slab top face should connect right (mask 8), got {top_quads.get((4.0, 6.0, 5.5))}"

    # Right vertical slab top face: connects Left (4)
    expected_right_uv0 = (round(get_glass_ctm_uv(4, [0, 8, 16, 16])[0][0], 4), round(get_glass_ctm_uv(4, [0, 8, 16, 16])[0][1], 4))
    assert top_quads.get((6.0, 6.0, 6.0)) == expected_right_uv0, f"Right slab top face should connect left (mask 4), got {top_quads.get((6.0, 6.0, 6.0))}"


def test_glass_full_block_adjacent_to_east_west_vertical_slabs():
    world = make_world()
    world.chunks[(0, 0, 0)].data.fill(0)

    # (5, 5, 4): glass_vertical_slab_west (x in [5.0, 5.5], z in [4.0, 5.0])
    # (5, 5, 5): glass (full block)
    # (5, 5, 6): glass_vertical_slab_east (x in [5.5, 6.0], z in [6.0, 7.0])
    world.set_block(5, 5, 4, "glass_vertical_slab_west")
    world.set_block(5, 5, 5, "glass")
    world.set_block(5, 5, 6, "glass_vertical_slab_east")

    mesh = build_chunk_mesh(world, 0, 0, 0)
    assert mesh is not None and mesh.cutout is not None
    verts, inds = mesh.cutout

    # North contact face of full glass at z=5.0:
    # x in [5.0, 5.5] touches west vertical slab -> culled
    # x in [5.5, 6.0] touches air -> rendered
    # South contact face of full glass at z=6.0:
    # x in [5.5, 6.0] touches east vertical slab -> culled
    # x in [5.0, 5.5] touches air -> rendered
    has_contact_z5_west = False
    has_exposed_z5_east = False
    has_contact_z6_east = False
    has_exposed_z6_west = False

    top_quads = {}
    for q in range(len(inds) // 6):
        i0 = inds[q * 6]
        i1 = inds[q * 6 + 1]
        v = verts[i0]
        v1_pt = verts[i1]
        norm = (round(float(v[3]), 2), round(float(v[4]), 2), round(float(v[5]), 2))
        pos = (round(float(v[0]), 2), round(float(v[1]), 2), round(float(v[2]), 2))
        u, v_coord = round(float(v[6]), 4), round(float(v[7]), 4)
        x_mid = (float(v[0]) + float(v1_pt[0])) * 0.5

        if norm == (0.0, 0.0, -1.0) and pos[2] == 5.0:
            if x_mid < 5.5:
                has_contact_z5_west = True
            else:
                has_exposed_z5_east = True
        elif norm == (0.0, 0.0, 1.0) and pos[2] == 6.0:
            if x_mid > 5.5:
                has_contact_z6_east = True
            else:
                has_exposed_z6_west = True
        elif norm == (0.0, 1.0, 0.0):
            top_quads[pos] = (u, v_coord)

    assert not has_contact_z5_west, "Contact face between full glass and west vertical slab at z=5.0 should be culled"
    assert has_exposed_z5_east, "Exposed face on east side at z=5.0 should be rendered"
    assert not has_contact_z6_east, "Contact face between full glass and east vertical slab at z=6.0 should be culled"
    assert has_exposed_z6_west, "Exposed face on west side at z=6.0 should be rendered"

    # Top face of full glass block:
    # West half (u in [0, 8]): connects Up (+v, -Z) to west slab (mask 1) + Right (8) to east half = mask 9
    # East half (u in [8, 16]): connects Down (-v, +Z) to east slab (mask 2) + Left (4) to west half = mask 6
    expected_west_uv0 = (round(get_glass_ctm_uv(9, [0, 0, 8, 16])[0][0], 4), round(get_glass_ctm_uv(9, [0, 0, 8, 16])[0][1], 4))
    expected_east_uv0 = (round(get_glass_ctm_uv(6, [8, 0, 16, 16])[0][0], 4), round(get_glass_ctm_uv(6, [8, 0, 16, 16])[0][1], 4))

    assert top_quads.get((5.0, 6.0, 6.0)) == expected_west_uv0, f"Full glass west half top face should connect up+right (mask 9), got {top_quads.get((5.0, 6.0, 6.0))}"
    assert top_quads.get((5.5, 6.0, 6.0)) == expected_east_uv0, f"Full glass east half top face should connect down+left (mask 6), got {top_quads.get((5.5, 6.0, 6.0))}"





