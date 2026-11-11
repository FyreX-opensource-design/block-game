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
