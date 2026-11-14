import unittest

import numpy as np

from server.blocks import (
    BLOCKS,
    CREATIVE,
    get_block_ctm_uv,
    is_glass,
)
from server.items import ITEMS
from server.lighting import get_block_opacity
from server.mesh import _get_glass_face_ctm_mask, build_chunk_mesh
from server.recipes import RECIPES, craft_match, smelt_match
from server.tags import has_block_tag, has_item_tag
from server.world import Chunk, World

DYE_COLORS = (
    "white",
    "light_gray",
    "gray",
    "black",
    "red",
    "orange",
    "yellow",
    "lime",
    "green",
    "cyan",
    "light_blue",
    "blue",
    "purple",
    "magenta",
    "pink",
    "brown",
    "amber",
    "aquamarine",
    "bubblegum",
    "conifer",
    "fluorescent",
    "honey",
    "lavender",
    "maroon",
    "mint",
    "navy",
    "peach",
    "persimmon",
    "rose",
    "spring_green",
    "ultramarine",
    "wine",
)

OTHER_STONES = ("granite", "andesite", "diorite", "limestone", "marble", "sandstone")


def _single_chunk_world():
    world = World(seed=1)
    chunk = Chunk(0, 0, 0)
    chunk.data.fill(0)
    world.chunks[(0, 0, 0)] = chunk
    return world


class BuildingBlockTests(unittest.TestCase):
    def test_stone_pillars_and_large_bricks(self):
        for stone in ("stone",) + OTHER_STONES:
            pil = f"{stone}_pillar"
            lb = f"large_{stone}_bricks"
            self.assertIn(pil, BLOCKS)
            self.assertIn(f"{pil}_x", BLOCKS)
            self.assertIn(f"{pil}_z", BLOCKS)
            self.assertIn(pil, ITEMS)
            self.assertIn(pil, CREATIVE)
            self.assertEqual(BLOCKS[pil].properties.get("rotation"), "axis")

            self.assertIn(lb, BLOCKS)
            self.assertIn(lb, ITEMS)
            self.assertIn(lb, CREATIVE)

            smooth = f"smooth_{stone}"
            res_pil = craft_match((smooth, "", smooth, ""))
            self.assertEqual(res_pil[:2], (pil, 2))
            res_lb = craft_match((smooth, smooth, smooth, smooth))
            self.assertEqual(res_lb[:2], (lb, 4))

    def test_oak_and_pine_wood_and_vertical_planks(self):
        for wid, log_id in (("oak_wood", "log"), ("pine_wood", "pine_log")):
            self.assertIn(wid, BLOCKS)
            self.assertIn(f"{wid}_x", BLOCKS)
            self.assertIn(f"{wid}_z", BLOCKS)
            self.assertIn(wid, ITEMS)
            self.assertIn(wid, CREATIVE)
            # All 6 faces use log side texture
            uvs = BLOCKS[wid].face_uvs
            for i in range(1, 6):
                self.assertEqual(uvs[i], uvs[0])
            self.assertTrue(has_item_tag(wid, "logs"))
            self.assertTrue(has_block_tag(wid, "logs"))
            self.assertEqual(craft_match((log_id, log_id, log_id, log_id))[:2], (wid, 3))

        for vid, plank_id in (("vertical_planks", "planks"), ("vertical_pine_planks", "pine_planks")):
            self.assertIn(vid, BLOCKS)
            self.assertIn(vid, ITEMS)
            self.assertIn(vid, CREATIVE)
            self.assertTrue(has_item_tag(vid, "planks"))
            self.assertTrue(has_block_tag(vid, "planks"))
            res_vert = craft_match((plank_id, "", "", plank_id, "", "", plank_id, "", ""))
            self.assertEqual(res_vert[:2], (vid, 3))
            self.assertEqual(craft_match((vid, "", "", ""))[:2], (plank_id, 1))

    def test_stained_glass_registration_ctm_translucent_and_recipes(self):
        self.assertEqual(len(DYE_COLORS), 32)
        for color in DYE_COLORS:
            gid = f"{color}_stained_glass"
            self.assertIn(gid, BLOCKS)
            self.assertIn(gid, ITEMS)
            self.assertIn(gid, CREATIVE)
            blk = BLOCKS[gid]
            self.assertTrue(blk.translucent)
            self.assertTrue(blk.cutout)
            self.assertTrue(blk.solid)
            self.assertTrue(is_glass(blk))
            self.assertEqual(get_block_opacity(blk), 0)
            self.assertTrue(blk.ctm)
            self.assertEqual(len(blk.palette_ctm_uvs[0]), 16)
            ring_res = craft_match(
                (
                    "glass", "glass", "glass",
                    "glass", f"{color}_dye", "glass",
                    "glass", "glass", "glass",
                )
            )
            self.assertEqual(ring_res[:2], (gid, 8))

        # Verify translucent mesh output + same-color vs different-color CTM & culling
        world = _single_chunk_world()
        world.set_block(5, 5, 5, "red_stained_glass")
        world.set_block(6, 5, 5, "red_stained_glass")
        world.set_block(7, 5, 5, "blue_stained_glass")
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh)
        self.assertIsNone(mesh.cutout)
        self.assertIsNotNone(mesh.translucent)
        # 2 red blocks joined (10 faces) + 1 blue block (6 faces) = 16 faces * 4 = 64 verts
        self.assertEqual(len(mesh.translucent[0]), 16 * 4)

        # Red glass at (6, 5, 5) north face (-Z, normal 5): right=+X (to blue glass, NOT same color), left=-X (to red glass)
        m_red = _get_glass_face_ctm_mask(world, 6, 5, 5, 4, block=BLOCKS["red_stained_glass"])
        # South (+Z): right=+X, left=-X -> left=4 connected to (5,5,5), right=8 NOT connected to blue at (7,5,5)
        self.assertEqual(m_red & 4, 4)
        self.assertEqual(m_red & 8, 0)
        self.assertIsNotNone(get_block_ctm_uv(BLOCKS["red_stained_glass"], 0))
        self.assertNotEqual(
            get_block_ctm_uv(BLOCKS["red_stained_glass"], 0),
            get_block_ctm_uv(BLOCKS["blue_stained_glass"], 0),
        )

    def test_colored_clay_bricks_brick_blocks_and_terracotta(self):
        self.assertIn("brick", ITEMS)
        self.assertIn("bricks", BLOCKS)
        self.assertIn("terracotta", BLOCKS)
        self.assertEqual(smelt_match("clay_ball")[:2], ("brick", 1))
        self.assertEqual(craft_match(("brick", "brick", "brick", "brick"))[:2], ("bricks", 1))
        self.assertEqual(smelt_match("clay")[:2], ("terracotta", 1))

        for b_id in BLOCKS:
            self.assertNotIn("glazed_terracotta", b_id)

        for color in DYE_COLORS:
            b_item = f"{color}_brick"
            b_block = f"{color}_bricks"
            tc_block = f"{color}_terracotta"
            self.assertIn(b_item, ITEMS)
            self.assertIn(b_block, BLOCKS)
            self.assertIn(b_block, ITEMS)
            self.assertIn(tc_block, BLOCKS)
            self.assertIn(tc_block, ITEMS)

            dye = f"{color}_dye"
            self.assertEqual(
                craft_match(("brick", "brick", "brick", "brick", dye, "brick", "brick", "brick", "brick"))[:2],
                (b_item, 8),
            )
            self.assertEqual(craft_match((b_item, b_item, b_item, b_item))[:2], (b_block, 1))
            self.assertEqual(
                craft_match(("bricks", "bricks", "bricks", "bricks", dye, "bricks", "bricks", "bricks", "bricks"))[:2],
                (b_block, 8),
            )
            self.assertEqual(
                craft_match(
                    (
                        "terracotta", "terracotta", "terracotta",
                        "terracotta", dye, "terracotta",
                        "terracotta", "terracotta", "terracotta",
                    )
                )[:2],
                (tc_block, 8),
            )


class SlabsStairsWallsTests(unittest.TestCase):
    STONE_VARIANT_ROOTS = [
        ("granite", "granite"),
        ("andesite", "andesite"),
        ("diorite", "diorite"),
        ("limestone", "limestone"),
        ("marble", "marble"),
        ("smooth_granite", "smooth_granite"),
        ("smooth_andesite", "smooth_andesite"),
        ("smooth_diorite", "smooth_diorite"),
        ("smooth_limestone", "smooth_limestone"),
        ("smooth_marble", "smooth_marble"),
        ("large_stone_bricks", "large_stone_brick"),
        ("large_granite_bricks", "large_granite_brick"),
        ("large_andesite_bricks", "large_andesite_brick"),
        ("large_diorite_bricks", "large_diorite_brick"),
        ("large_limestone_bricks", "large_limestone_brick"),
        ("large_marble_bricks", "large_marble_brick"),
        ("large_sandstone_bricks", "large_sandstone_brick"),
    ]

    def test_stone_slabs_vertical_slabs_stairs_walls_and_recipes(self):
        from server.blocks import block_index
        for source_id, prefix in self.STONE_VARIANT_ROOTS:
            slab_id = f"{prefix}_slab"
            vslab_id = f"{prefix}_vertical_slab"
            stairs_id = f"{prefix}_stairs"
            wall_id = f"{prefix}_wall"

            for bid in (slab_id, vslab_id, stairs_id, wall_id):
                self.assertIn(bid, BLOCKS, f"Missing block {bid}")
                self.assertIn(bid, ITEMS, f"Missing item {bid}")

            self.assertEqual(
                craft_match((source_id, source_id, source_id, "", "", "", "", "", ""))[:2],
                (slab_id, 6),
            )
            self.assertEqual(
                craft_match((slab_id, "", "", slab_id, "", "", slab_id, "", ""))[:2],
                (vslab_id, 3),
            )
            self.assertEqual(
                craft_match(
                    (
                        source_id, "", "",
                        source_id, source_id, "",
                        source_id, source_id, source_id,
                    )
                )[:2],
                (stairs_id, 4),
            )

            self.assertEqual(
                craft_match((source_id, source_id, source_id, source_id, source_id, source_id, "", "", ""))[:2],
                (wall_id, 6),
            )

            # Lazy dynamic resolution of combined double slabs & cross-variants
            dbl_h = f"double_slab_{slab_id}"
            dbl_v = f"double_vertical_slab_z_{vslab_id}"
            off_s = f"{vslab_id}_offset_{slab_id}"
            self.assertIn(dbl_h, BLOCKS)
            self.assertIn(dbl_v, BLOCKS)
            self.assertIn(off_s, BLOCKS)
            self.assertGreater(block_index(dbl_h), 0)
            self.assertGreater(block_index(dbl_v), 0)
            self.assertGreater(block_index(off_s), 0)

    def test_stained_glass_slabs_and_vertical_slabs_translucent_and_ctm(self):
        stained_bases = sorted(
            b.id
            for b in BLOCKS.values()
            if b.id.endswith("_stained_glass") and not b.is_variant
        )
        self.assertEqual(len(stained_bases), 32)

        for glass_id in stained_bases:
            slab_id = f"{glass_id}_slab"
            vslab_id = f"{glass_id}_vertical_slab"
            for bid in (slab_id, vslab_id):
                self.assertIn(bid, BLOCKS, f"Missing stained glass block {bid}")
                self.assertIn(bid, ITEMS, f"Missing stained glass item {bid}")
                blk = BLOCKS[bid]
                self.assertTrue(blk.translucent, f"{bid} should be translucent")
                self.assertTrue(blk.ctm, f"{bid} should enable CTM")
                self.assertTrue(blk.properties.get("glass"), f"{bid} should have glass=true")
                self.assertEqual(blk.properties.get("ctm_group"), glass_id)

            self.assertEqual(
                craft_match((glass_id, glass_id, glass_id, "", "", "", "", "", ""))[:2],
                (slab_id, 6),
            )
            self.assertEqual(
                craft_match((slab_id, "", "", slab_id, "", "", slab_id, "", ""))[:2],
                (vslab_id, 3),
            )

        # Verify translucent mesh emission for slabs, vertical slabs, and mixed doubles
        world = _single_chunk_world()
        world.set_block(4, 4, 4, "red_stained_glass_slab")
        world.set_block(6, 4, 4, "lavender_stained_glass_vertical_slab")
        world.set_block(8, 4, 4, "double_slab_red_stained_glass_slab_blue_stained_glass_slab")
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh.translucent)
        self.assertGreater(len(mesh.translucent[0]), 0)
        self.assertIsNone(mesh.cutout)

    def test_vertical_planks_slabs_and_stairs(self):
        for source_id in ("vertical_planks", "vertical_pine_planks"):
            slab_id = f"{source_id}_slab"
            vslab_id = f"{source_id}_vertical_slab"
            stairs_id = f"{source_id}_stairs"
            for bid in (slab_id, vslab_id, stairs_id):
                self.assertIn(bid, BLOCKS, f"Missing {bid}")
                self.assertIn(bid, ITEMS, f"Missing item {bid}")

            self.assertEqual(
                craft_match((source_id, source_id, source_id, "", "", "", "", "", ""))[:2],
                (slab_id, 6),
            )
            self.assertEqual(
                craft_match((slab_id, "", "", slab_id, "", "", slab_id, "", ""))[:2],
                (vslab_id, 3),
            )
            self.assertEqual(
                craft_match(
                    (
                        source_id, "", "",
                        source_id, source_id, "",
                        source_id, source_id, source_id,
                    )
                )[:2],
                (stairs_id, 4),
            )

    def test_palette_corner_stairs_and_wall_arms_preserve_sub_tile_uvs(self):
        from server.blocks import block_palette_index, get_stair_elements, get_wall_elements

        base_stairs = BLOCKS["granite_stairs"]
        self.assertGreater(len(base_stairs.palette_face_uvs), 1)
        for var_id in (
            "granite_stairs_south",
            "granite_stairs_east",
            "granite_stairs_west",
            "granite_stairs_north_top",
        ):
            var_blk = BLOCKS[var_id]
            self.assertEqual(len(var_blk.palette_face_uvs), len(base_stairs.palette_face_uvs))
            self.assertTrue(all("face_uv_specs" in el for el in var_blk.elements))

        # Test isolated corner stair mesh UVs against tile dimensions
        for shape_neighbor_pos, neighbor_id in (
            ((4, 4, 5), "granite_stairs_east"),  # inner_right corner
            ((4, 4, 3), "granite_stairs_east"),  # outer_right corner
        ):
            world = _single_chunk_world()
            world.set_block(4, 4, 4, "granite_stairs")
            els = get_stair_elements(world, 4, 4, 4, base_stairs)
            self.assertEqual(len(els), 2)

            world.set_block(*shape_neighbor_pos, neighbor_id)
            corner_els = get_stair_elements(world, 4, 4, 4, base_stairs)
            self.assertTrue(all("face_uv_specs" in el for el in corner_els))

            # Verify lower half side faces map half-height [8..16], not [0..16]
            for face_idx in (0, 1, 4, 5):
                self.assertEqual(corner_els[0]["face_uv_specs"][face_idx][1], 8.0)
                self.assertEqual(corner_els[0]["face_uv_specs"][face_idx][3], 16.0)

            # Verify upper corner quad vertices in built mesh use 8x8 half-tile UV span
            p_idx = block_palette_index(4, 4, 4, world.seed, len(base_stairs.palette_face_uvs))
            u0, v0, u1, v1 = base_stairs.palette_texture_rects[(p_idx, 2)]
            tile_w = u1 - u0
            tile_h = v1 - v0

            # Clear neighbor only after getting elements? Keep empty around isolated corner
            w_isolated = _single_chunk_world()
            # Keep only target block and neighbor at (4,4,3) (outside y>0.5 of (4,4,4))
            w_isolated.set_block(4, 4, 4, "granite_stairs")
            w_isolated.set_block(*shape_neighbor_pos, neighbor_id)
            mesh = build_chunk_mesh(w_isolated, 0, 0, 0)
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            # Group into 4-vertex quads and filter by +Y upper face strictly inside block (4, 4, 4)
            quads = verts.reshape(-1, 4, 9)
            centers = quads[:, :, :3].mean(axis=1)
            normals = quads[:, 0, 3:6]
            mask = (
                (np.abs(normals[:, 1] - 1.0) < 1e-4)
                & (np.abs(centers[:, 1] - 5.0) < 1e-4)
                & (centers[:, 0] > 4.0)
                & (centers[:, 0] < 5.0)
                & (centers[:, 2] > 4.0)
                & (centers[:, 2] < 5.0)
            )
            top_quads = quads[mask]
            self.assertGreaterEqual(len(top_quads), 1)
            # Each 8x8 quadrant quad has U span = 0.5 * tile_w and V span = 0.5 * tile_h
            for quad in top_quads:
                self.assertAlmostEqual(float(quad[:, 6].max() - quad[:, 6].min()), 0.5 * tile_w, places=5)
                self.assertAlmostEqual(float(quad[:, 7].max() - quad[:, 7].min()), 0.5 * tile_h, places=5)

        # Palette wall connecting arms should also carry sub-tile face_uv_specs
        wall_world = _single_chunk_world()
        wall_world.set_block(6, 4, 6, "granite_wall")
        wall_world.set_block(6, 4, 7, "granite_wall")
        w_els = get_wall_elements(wall_world, 6, 4, 6, BLOCKS["granite_wall"])
        self.assertGreater(len(w_els), 1)
        self.assertTrue(all("face_uv_specs" in el for el in w_els))

    def test_offsetting_stained_glass_with_vertical_slab_preserves_color_and_uvs(self):
        from server.blocks import get_double_vertical_slab, is_vertical_slab
        from server.sub_cell import SUB_CELL_BLOCK, calculate_horizontal_offset_placement, full_block_for

        # Every non-base vertical slab should resolve to a full cube block
        for b in list(BLOCKS.values()):
            if not b.is_variant and is_vertical_slab(b):
                fb = full_block_for(b.id)
                self.assertIn(fb, BLOCKS)
                self.assertTrue(BLOCKS[fb].full_cube, f"{b.id} resolved to non-full block {fb}")

        red_tiles = [BLOCKS["red_stained_glass"].palette_ctm_uvs[0][m] for m in range(16)]
        red_u = [(min(uv[0] for uv in t), max(uv[0] for uv in t)) for t in red_tiles]
        red_v = [(min(uv[1] for uv in t), max(uv[1] for uv in t)) for t in red_tiles]

        for vid, expected_mat in (("stone_brick_vertical_slab", "stone_bricks"), ("stone_vertical_slab", "stone")):
            world = _single_chunk_world()
            world.set_block(4, 4, 4, vid)
            hit = {"x": 4, "y": 4, "z": 4, "nx": 0, "ny": 0, "nz": -1, "local_x": 0.5, "local_y": 0.5, "local_z": 0.5}
            res = calculate_horizontal_offset_placement(world, hit, "red_stained_glass", 0.0)
            self.assertIsNotNone(res)
            for cx, cy, cz, sub in res:
                world.set_block(cx, cy, cz, SUB_CELL_BLOCK)
                world.set_sub_cell(cx, cy, cz, sub)

            mesh = build_chunk_mesh(world, 0, 0, 0)
            self.assertIsNone(mesh.cutout)
            self.assertIsNotNone(mesh.translucent)

            tquads = np.asarray(mesh.translucent[0], dtype=np.float32).reshape(-1, 4, 9)
            for quad in tquads:
                qu0, qu1 = float(quad[:, 6].min()), float(quad[:, 6].max())
                qv0, qv1 = float(quad[:, 7].min()), float(quad[:, 7].max())
                self.assertTrue(
                    any(qu0 >= u0 - 1e-5 and qu1 <= u1 + 1e-5 and qv0 >= v0 - 1e-5 and qv1 <= v1 + 1e-5 for (u0, u1), (v0, v1) in zip(red_u, red_v)),
                    f"{vid}: stained glass quad fell outside red_stained_glass CTM atlas tiles",
                )

            oquads = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 4, 9)
            fb_uv = BLOCKS[expected_mat].face_uvs[2]
            tw = abs(fb_uv[1][0] - fb_uv[0][0])
            th = abs(fb_uv[2][1] - fb_uv[0][1])
            for quad in oquads:
                du = float(quad[:, 6].max() - quad[:, 6].min())
                dv = float(quad[:, 7].max() - quad[:, 7].min())
                self.assertAlmostEqual(du, 0.5 * tw, places=4)
                self.assertAlmostEqual(dv, 0.5 * th, places=4)

        # Mixed double vertical slab preserves per-element stained glass CTM
        mix_world = _single_chunk_world()
        dvs = get_double_vertical_slab("red_stained_glass_vertical_slab", "stone_vertical_slab_north")
        mix_world.set_block(4, 4, 4, dvs)
        mix_mesh = build_chunk_mesh(mix_world, 0, 0, 0)
        self.assertIsNotNone(mix_mesh.translucent)
        for quad in np.asarray(mix_mesh.translucent[0], dtype=np.float32).reshape(-1, 4, 9):
            qu0, qu1 = float(quad[:, 6].min()), float(quad[:, 6].max())
            qv0, qv1 = float(quad[:, 7].min()), float(quad[:, 7].max())
            self.assertTrue(
                any(qu0 >= u0 - 1e-5 and qu1 <= u1 + 1e-5 and qv0 >= v0 - 1e-5 and qv1 <= v1 + 1e-5 for (u0, u1), (v0, v1) in zip(red_u, red_v))
            )

    def test_stained_glass_slab_mixing_culling_and_ctm(self):
        from server.blocks import get_capped_offset_block, get_double_slab, get_double_vertical_slab, get_offset_block
        from server.sub_cell import SUB_CELL_BLOCK, SubCell, calculate_horizontal_offset_placement

        tiles = BLOCKS["yellow_stained_glass"].palette_ctm_uvs[0]

        def translucent_quads(world):
            mesh = build_chunk_mesh(world, 0, 0, 0)
            self.assertIsNone(mesh.cutout)
            if mesh.translucent is None:
                return np.zeros((0, 4, 9), dtype=np.float32)
            return np.asarray(mesh.translucent[0], dtype=np.float32).reshape(-1, 4, 9)

        def tile_indices(quad):
            qu0, qu1 = float(quad[:, 6].min()), float(quad[:, 6].max())
            qv0, qv1 = float(quad[:, 7].min()), float(quad[:, 7].max())
            out = []
            for idx, t in enumerate(tiles):
                tu0, tu1 = min(uv[0] for uv in t), max(uv[0] for uv in t)
                tv0, tv1 = min(uv[1] for uv in t), max(uv[1] for uv in t)
                if qu0 >= tu0 - 1e-5 and qu1 <= tu1 + 1e-5 and qv0 >= tv0 - 1e-5 and qv1 <= tv1 + 1e-5:
                    out.append(idx)
            return out

        # 1. Full stained glass + mixed double_slab culls internal seam at x=5.0
        w_ds = _single_chunk_world()
        w_ds.set_block(4, 4, 4, "yellow_stained_glass")
        w_ds.set_block(5, 4, 4, get_double_slab("yellow_stained_glass_slab", "stone_brick_slab_top"))
        q_ds = translucent_quads(w_ds)
        seam_x5 = [q for q in q_ds if 4.99 <= float(q[:, 0].min()) <= 5.01 and 4.99 <= float(q[:, 0].max()) <= 5.01]
        self.assertEqual(len(seam_x5), 0)

        # 2. Full stained glass below mixed SubCell (yellow bottom + stone top-north)
        w_sub_v = _single_chunk_world()
        w_sub_v.set_block(4, 4, 4, "yellow_stained_glass")
        sc_b = SubCell()
        for ox in (0, 1):
            for oz in (0, 1):
                sc_b.set_octant(ox, 0, oz, "yellow_stained_glass")
                if oz == 0:
                    sc_b.set_octant(ox, 1, oz, "stone_bricks")
        w_sub_v.set_block(4, 5, 4, SUB_CELL_BLOCK)
        w_sub_v.set_sub_cell(4, 5, 4, sc_b)
        q_sub_v = translucent_quads(w_sub_v)
        seam_y5 = [q for q in q_sub_v if 4.99 <= float(q[:, 1].min()) <= 5.01 and 4.99 <= float(q[:, 1].max()) <= 5.01]
        self.assertEqual(len(seam_y5), 0)
        z_pos_masks = [tile_indices(q)[0] for q in q_sub_v if round(float(q[0, 5]), 1) == 1.0]
        self.assertIn(1, z_pos_masks)
        self.assertIn(2, z_pos_masks)

        # 3. Full stained glass + horizontally offset SubCell glass on vertical slab
        w_off = _single_chunk_world()
        w_off.set_block(4, 4, 4, "yellow_stained_glass")
        w_off.set_block(5, 4, 4, "stone_brick_vertical_slab")
        hit = {"x": 5, "y": 4, "z": 4, "nx": 0, "ny": 0, "nz": -1, "local_x": 0.5, "local_y": 0.5, "local_z": 0.5}
        for cx, cy, cz, sub in calculate_horizontal_offset_placement(w_off, hit, "yellow_stained_glass", 0.0):
            w_off.set_block(cx, cy, cz, SUB_CELL_BLOCK)
            w_off.set_sub_cell(cx, cy, cz, sub)
        q_off = translucent_quads(w_off)
        x5_quads = [q for q in q_off if 4.99 <= float(q[:, 0].min()) <= 5.01 and 4.99 <= float(q[:, 0].max()) <= 5.01]
        # Only exposed z in [3.5, 4.0] remains on x=5.0; internal z in [4.0, 5.0] is culled
        self.assertTrue(len(x5_quads) > 0)
        for q in x5_quads:
            self.assertLessEqual(float(q[:, 2].max()), 4.01)

        # 4. Full stained glass abutting single SubCell octant culls only that quadrant
        w_quad = _single_chunk_world()
        w_quad.set_block(4, 4, 4, "yellow_stained_glass")
        sc_q = SubCell()
        sc_q.set_octant(0, 0, 0, "yellow_stained_glass")
        w_quad.set_block(5, 4, 4, SUB_CELL_BLOCK)
        w_quad.set_sub_cell(5, 4, 4, sc_q)
        x5_pos = [
            q for q in translucent_quads(w_quad)
            if 4.99 <= float(q[:, 0].min()) <= 5.01 and 4.99 <= float(q[:, 0].max()) <= 5.01 and float(q[0, 3]) > 0.5
        ]
        self.assertEqual(len(x5_pos), 3)

        # 5. Vertically stacked mixed double_vertical_slab culls y=5.0 seam and connects CTM
        w_dvs = _single_chunk_world()
        dvs = get_double_vertical_slab("yellow_stained_glass_vertical_slab", "stone_brick_vertical_slab_north")
        w_dvs.set_block(4, 4, 4, dvs)
        w_dvs.set_block(4, 5, 4, dvs)
        q_dvs = translucent_quads(w_dvs)
        self.assertEqual(
            [q for q in q_dvs if 4.99 <= float(q[:, 1].min()) <= 5.01 and 4.99 <= float(q[:, 1].max()) <= 5.01],
            [],
        )
        east_masks = {tile_indices(q)[0] for q in q_dvs if round(float(q[0, 3]), 1) == 1.0}
        self.assertEqual(east_masks, {1, 2})

        # 6. Stacked offset + capped offset glass culls y=4.5 seam and connects CTM
        w_cap = _single_chunk_world()
        fb_off = get_offset_block("yellow_stained_glass")
        capped = get_capped_offset_block(fb_off, "stone_brick_slab_top")
        self.assertIsNotNone(capped)
        w_cap.set_block(4, 4, 4, fb_off)
        w_cap.set_block(4, 5, 4, capped)
        q_cap = translucent_quads(w_cap)
        self.assertEqual(
            [q for q in q_cap if 4.49 <= float(q[:, 1].min()) <= 4.51 and 4.49 <= float(q[:, 1].max()) <= 4.51],
            [],
        )
        cap_z_masks = {tile_indices(q)[0] for q in q_cap if round(float(q[0, 5]), 1) == 1.0}
        self.assertEqual(cap_z_masks, {1, 2})

        # 7. SubCell L-shape (bottom glass slab + glass vertical slab) uses consistent tile-0 subregions on +Z
        w_l = _single_chunk_world()
        sc_l = SubCell()
        for ox in (0, 1):
            for oz in (0, 1):
                sc_l.set_octant(ox, 0, oz, "yellow_stained_glass")
            sc_l.set_octant(ox, 1, 1, "yellow_stained_glass")
        w_l.set_block(4, 5, 4, SUB_CELL_BLOCK)
        w_l.set_sub_cell(4, 5, 4, sc_l)
        t0 = tiles[0]
        tu0, tu1 = min(u[0] for u in t0), max(u[0] for u in t0)
        tv0, tv1 = min(u[1] for u in t0), max(u[1] for u in t0)
        z_quads = [q for q in translucent_quads(w_l) if round(float(q[0, 5]), 1) == 1.0]
        self.assertEqual(len(z_quads), 4)
        for q in z_quads:
            self.assertEqual(tile_indices(q), [0])
            self.assertAlmostEqual(float(q[:, 6].max() - q[:, 6].min()), 0.5 * (tu1 - tu0), places=5)
            self.assertAlmostEqual(float(q[:, 7].max() - q[:, 7].min()), 0.5 * (tv1 - tv0), places=5)

        # 8. Different-color mixed double_slab preserves its shared y=4.5 seam
        w_diff = _single_chunk_world()
        w_diff.set_block(4, 4, 4, get_double_slab("yellow_stained_glass_slab", "blue_stained_glass_slab_top"))
        seam_diff = [
            q for q in translucent_quads(w_diff)
            if 4.49 <= float(q[:, 1].min()) <= 4.51 and 4.49 <= float(q[:, 1].max()) <= 4.51
        ]
        self.assertEqual(len(seam_diff), 2)

        # 9. Offset glass next to full glass splits and culls overlapping y=[4.0, 4.5] seam on both sides
        from server.blocks import get_vertical_slab_offset_block

        w_og_full = _single_chunk_world()
        w_og_full.set_block(4, 4, 4, fb_off)
        w_og_full.set_block(5, 4, 4, "yellow_stained_glass")
        x5_og_full = [
            q for q in translucent_quads(w_og_full)
            if 4.99 <= float(q[:, 0].min()) <= 5.01 and 4.99 <= float(q[:, 0].max()) <= 5.01
        ]
        self.assertEqual(len(x5_og_full), 2)
        y_ranges = sorted((round(float(q[:, 1].min()), 2), round(float(q[:, 1].max()), 2)) for q in x5_og_full)
        self.assertEqual(y_ranges, [(3.5, 4.0), (4.5, 5.0)])

        # 10. Offset glass next to bottom glass slab culls y=[4.0, 4.5] overlap and connects CTM
        w_og_slab = _single_chunk_world()
        w_og_slab.set_block(4, 4, 4, fb_off)
        w_og_slab.set_block(5, 4, 4, "yellow_stained_glass_slab")
        q_og_slab = translucent_quads(w_og_slab)
        x5_og_slab = [
            q for q in q_og_slab
            if 4.99 <= float(q[:, 0].min()) <= 5.01 and 4.99 <= float(q[:, 0].max()) <= 5.01
        ]
        self.assertEqual(len(x5_og_slab), 1)
        self.assertAlmostEqual(float(x5_og_slab[0][:, 1].min()), 3.5, places=4)
        self.assertAlmostEqual(float(x5_og_slab[0][:, 1].max()), 4.0, places=4)
        z_masks_og_slab = sorted(tile_indices(q)[0] for q in q_og_slab if round(float(q[0, 5]), 1) == 1.0)
        self.assertEqual(z_masks_og_slab, [1, 4, 10])

        # 11. Offset glass next to offset vertical slab culls covered z=[4.0, 4.5] half
        vs_off_n = get_vertical_slab_offset_block("yellow_stained_glass_vertical_slab_north")
        w_og_vso = _single_chunk_world()
        w_og_vso.set_block(4, 4, 4, fb_off)
        w_og_vso.set_block(5, 4, 4, vs_off_n)
        x5_og_vso = [
            q for q in translucent_quads(w_og_vso)
            if 4.99 <= float(q[:, 0].min()) <= 5.01 and 4.99 <= float(q[:, 0].max()) <= 5.01
        ]
        self.assertEqual(len(x5_og_vso), 1)
        self.assertAlmostEqual(float(x5_og_vso[0][:, 2].min()), 4.5, places=4)
        self.assertAlmostEqual(float(x5_og_vso[0][:, 2].max()), 5.0, places=4)

    def test_mesh_element_cache_isolation_and_door_batch_upload(self):
        from unittest.mock import MagicMock
        from client.app import Game

        # Chunk element cache must not leak across world mutations between rebuilds
        w = _single_chunk_world()
        w.set_block(4, 4, 4, "yellow_stained_glass")
        m1 = build_chunk_mesh(w, 0, 0, 0)
        q1 = np.asarray(m1.translucent[0], dtype=np.float32).reshape(-1, 4, 9)
        self.assertEqual(len(q1), 6)
        w.set_block(5, 4, 4, "yellow_stained_glass")
        m2 = build_chunk_mesh(w, 0, 0, 0)
        q2 = np.asarray(m2.translucent[0], dtype=np.float32).reshape(-1, 4, 9)
        self.assertEqual(len(q2), 10)

        # Toggling a double door inside one chunk triggers a single chunk upload instead of 4
        game = Game.__new__(Game)
        game.world = _single_chunk_world()
        game.world.dirty = set()
        game.network_client = None
        game.sound_manager = None
        game.world.set_block(5, 3, 5, "stone")
        game.world.set_block(6, 3, 5, "stone")
        game.world.set_block(5, 4, 5, "wooden_door_lower_south_right")
        game.world.set_block(5, 5, 5, "wooden_door_upper_south_right")
        game.world.set_block(6, 4, 5, "wooden_door_lower_south")
        game.world.set_block(6, 5, 5, "wooden_door_upper_south")
        game._upload = MagicMock()
        game._toggle_door(5, 4, 5)
        self.assertEqual(game._upload.call_count, 1)

    def test_dynamic_composite_glass_ctm_and_color_preservation(self):
        from server.blocks import (
            GLASS_CTM_UVS,
            get_block_ctm_uv,
            get_capped_offset_block,
            get_capped_vertical_slab_offset,
            get_offset_block,
            get_vertical_slab_offset_block,
        )

        fb_off = get_offset_block("yellow_stained_glass")
        cap = BLOCKS[get_capped_offset_block(fb_off, "stone_brick_slab_top")]
        self.assertTrue(cap.translucent)
        self.assertIsNotNone(cap.palette_ctm_rects)
        self.assertIsNotNone(cap.palette_ctm_uvs)

        v_off = get_vertical_slab_offset_block("yellow_stained_glass_vertical_slab")
        v_cap = BLOCKS[get_capped_vertical_slab_offset(v_off, "stone_brick_slab_top")]
        self.assertTrue(v_cap.translucent)
        self.assertIsNotNone(v_cap.palette_ctm_rects)
        self.assertIsNotNone(v_cap.palette_ctm_uvs)

        # Dynamic composite glass blocks must never return clear glass CTM
        yellow_base_uv = BLOCKS["yellow_stained_glass"].palette_ctm_uvs[0][0]
        clear_glass_uv = GLASS_CTM_UVS[0]
        self.assertNotEqual(yellow_base_uv, clear_glass_uv)

        for b in (cap, v_cap, BLOCKS[fb_off], BLOCKS[v_off]):
            uv = get_block_ctm_uv(b, 0)
            self.assertNotEqual(uv, clear_glass_uv, f"{b.id} returned clear glass UV")
            self.assertEqual(uv, yellow_base_uv, f"{b.id} did not preserve yellow stained glass CTM")
