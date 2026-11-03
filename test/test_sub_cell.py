import unittest

from server.sub_cell import OCTANT_COORDS, SubCell, octant_index


class TestSubCell(unittest.TestCase):
    def test_octant_indexing(self):
        for idx, (ox, oy, oz) in enumerate(OCTANT_COORDS):
            self.assertEqual(octant_index(ox, oy, oz), idx)

    def test_horizontal_slabs(self):
        bottom = SubCell.from_horizontal_slab("stone", half="bottom")
        self.assertFalse(bottom.is_empty())
        self.assertFalse(bottom.is_full())
        self.assertEqual(bottom.count_occupied(), 4)
        for ox in (0, 1):
            for oz in (0, 1):
                self.assertEqual(bottom.get_octant(ox, 0, oz), "stone")
                self.assertEqual(bottom.get_octant(ox, 1, oz), "")

        top = SubCell.from_horizontal_slab("stone", half="top")
        self.assertEqual(top.count_occupied(), 4)
        for ox in (0, 1):
            for oz in (0, 1):
                self.assertEqual(top.get_octant(ox, 0, oz), "")
                self.assertEqual(top.get_octant(ox, 1, oz), "stone")

    def test_vertical_slabs(self):
        north = SubCell.from_vertical_slab("wood", facing="north")
        self.assertEqual(north.count_occupied(), 4)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(north.get_octant(ox, oy, 0), "wood")
                self.assertEqual(north.get_octant(ox, oy, 1), "")

        west = SubCell.from_vertical_slab("wood", facing="west")
        self.assertEqual(west.count_occupied(), 4)
        for oy in (0, 1):
            for oz in (0, 1):
                self.assertEqual(west.get_octant(0, oy, oz), "wood")
                self.assertEqual(west.get_octant(1, oy, oz), "")

    def test_mixed_slabs(self):
        cell = SubCell.from_horizontal_slab("stone", half="bottom")
        for ox in (0, 1):
            for oz in (0, 1):
                cell.set_octant(ox, 1, oz, "planks")
        self.assertEqual(cell.count_occupied(), 8)
        self.assertFalse(cell.is_full())  # Different materials!
        self.assertEqual(cell.get_octant(0, 0, 0), "stone")
        self.assertEqual(cell.get_octant(0, 1, 0), "planks")

    def test_serialization(self):
        cell = SubCell.from_horizontal_slab("sandstone", half="bottom")
        data = cell.to_dict()
        restored = SubCell.from_dict(data)
        self.assertEqual(restored.octants, cell.octants)
        self.assertEqual(restored.waterlogged, cell.waterlogged)

    def test_bounding_boxes(self):
        cell = SubCell.from_horizontal_slab("stone", half="bottom")
        boxes = cell.bounding_boxes(10.0, 20.0, 30.0)
        self.assertEqual(len(boxes), 4)
        for bx0, by0, bz0, bx1, by1, bz1 in boxes:
            self.assertAlmostEqual(by0, 20.0)
            self.assertAlmostEqual(by1, 20.5)
            self.assertAlmostEqual(bx1 - bx0, 0.5)
            self.assertAlmostEqual(bz1 - bz0, 0.5)


class TestWorldSubCell(unittest.TestCase):
    def setUp(self):
        from server.world import World

        self.world = World(seed=1)
        self.world.load_chunk(0, 5, 0)
        self.world.chunks[(0, 5, 0)].data.fill(0)

    def test_world_set_get_sub_cell(self):
        cell = SubCell.from_horizontal_slab("stone", half="bottom")
        self.assertTrue(self.world.set_sub_cell(5, 81, 5, cell))
        self.assertEqual(self.world.get_block(5, 81, 5), "sub_cell")
        retrieved = self.world.get_sub_cell(5, 81, 5)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.count_occupied(), 4)
        self.assertTrue(self.world.is_solid(5, 81, 5))

    def test_world_sub_cell_full_remains_sub_cell(self):
        # A fully filled sub-cell remains a sub-cell to preserve sub-voxel identities
        cell = SubCell(["stone"] * 8)
        self.assertTrue(self.world.set_sub_cell(5, 81, 5, cell))
        self.assertEqual(self.world.get_block(5, 81, 5), "sub_cell")
        self.assertIsNotNone(self.world.get_sub_cell(5, 81, 5))

    def test_world_sub_cell_collision_and_raycast(self):
        cell = SubCell.from_horizontal_slab("stone", half="bottom")
        self.world.set_sub_cell(5, 81, 5, cell)

        # Collision boxes
        boxes = self.world.get_collision_boxes(5, 81, 5)
        self.assertEqual(len(boxes), 4)

        # AABB collision in bottom half (y=81.0..81.4)
        self.assertTrue(self.world.collides_box(5.1, 81.0, 5.1, 5.4, 81.3, 5.4))
        # Top half (y=81.6..81.9) is empty, so no collision!
        self.assertFalse(self.world.collides_box(5.1, 81.6, 5.1, 5.4, 81.9, 5.4))

        # Raycast into the bottom slab from above
        hit = self.world.raycast(5.2, 85.0, 5.2, 0.0, -1.0, 0.0, max_dist=10.0)
        self.assertIsNotNone(hit)
        self.assertEqual((hit["x"], hit["y"], hit["z"]), (5, 81, 5))
        self.assertAlmostEqual(hit["hit_y"], 81.5)
        self.assertEqual((hit["nx"], hit["ny"], hit["nz"]), (0, 1, 0))
        self.assertIn("octant", hit)

    def test_overwriting_sub_cell_cleans_up(self):
        cell = SubCell.from_horizontal_slab("stone", half="bottom")
        self.world.set_sub_cell(5, 81, 5, cell)
        self.assertIsNotNone(self.world.get_sub_cell(5, 81, 5))

        # Setting regular block clears sub_cell dict entry
        self.world.set_block(5, 81, 5, "dirt")
        self.assertEqual(self.world.get_block(5, 81, 5), "dirt")
        self.assertIsNone(self.world.get_sub_cell(5, 81, 5))

    def test_sub_cell_meshing(self):
        from server.mesh import build_chunk_mesh

        cell = SubCell.from_horizontal_slab("stone", half="bottom")
        self.world.set_sub_cell(5, 81, 5, cell)

        mesh = build_chunk_mesh(self.world, 0, 5, 0)
        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.indices), 0)
        # 16 quads * 4 vertices = 64 vertices for an isolated bottom slab
        self.assertEqual(len(mesh.vertices), 64)
        self.assertEqual(len(mesh.indices), 96)

        # Now place a solid block underneath at y=80
        self.world.set_block(5, 80, 5, "stone")
        mesh2 = build_chunk_mesh(self.world, 0, 5, 0)
        # The 4 bottom faces of the slab are culled by the solid stone below (12 quads = 48 verts)
        # Stone block at y=80 has 6 exposed faces (since sub_cell is a partial block): 6 * 4 = 24 verts
        # Total = 48 + 24 = 72 vertices
        self.assertEqual(len(mesh2.vertices), 72)

    def test_save_and_load_sub_cells(self):
        import tempfile
        from pathlib import Path
        from server.storage import ChunkStore

        with tempfile.TemporaryDirectory() as tmp:
            store = ChunkStore(Path(tmp) / "chunks")
            self.world.store = store
            cell = SubCell.from_horizontal_slab("sandstone", half="bottom")
            self.world.set_sub_cell(5, 81, 5, cell)
            self.world.flush()

            # Evict from memory and reload from disk
            self.world.evict_chunk((0, 5, 0))
            self.assertNotIn((0, 5, 0), self.world.chunks)

            loaded = self.world.load_chunk(0, 5, 0)
            self.assertIsNotNone(loaded)
            self.assertEqual(self.world.get_block(5, 81, 5), "sub_cell")
            retrieved = self.world.get_sub_cell(5, 81, 5)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.get_octant(0, 0, 0), "sandstone")
            self.assertEqual(retrieved.get_octant(0, 1, 0), "")

    def test_horizontal_and_vertical_slab_mixing_in_sub_cell(self):
        from server.mesh import build_chunk_mesh

        # Create a cell with bottom horizontal slab (stone) and half vertical slab (planks, facing south)
        cell = SubCell()
        # Bottom horizontal slab
        for ox in (0, 1):
            for oz in (0, 1):
                cell.set_octant(ox, 0, oz, "stone")
        # Half vertical slab on top South
        for ox in (0, 1):
            cell.set_octant(ox, 1, 1, "planks")

        self.world.set_sub_cell(5, 81, 5, cell)
        mesh = build_chunk_mesh(self.world, 0, 5, 0)
        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh.vertices), 0)

        # Raycast tests:
        # Hitting top South (planks vertical slab) from above at z=5.8
        hit_planks = self.world.raycast(5.5, 85.0, 5.8, 0.0, -1.0, 0.0, 10.0)
        self.assertIsNotNone(hit_planks)
        self.assertAlmostEqual(hit_planks["hit_y"], 82.0)  # Top of cell
        self.assertEqual(hit_planks["octant"], (1, 1, 1) if hit_planks["octant"][0] == 1 else (0, 1, 1))

        # Hitting top North (stone horizontal slab) from above at z=5.2
        hit_stone = self.world.raycast(5.5, 85.0, 5.2, 0.0, -1.0, 0.0, 10.0)
        self.assertIsNotNone(hit_stone)
        self.assertAlmostEqual(hit_stone["hit_y"], 81.5)  # Top of horizontal slab at 81.5!
        self.assertEqual(hit_stone["octant"][1], 0)  # oy=0 (the stone slab below!)

    def test_placement_mixing(self):
        from server.sub_cell import calculate_sub_cell_placement

        # Start with a horizontal stone slab
        self.world.set_block(5, 81, 5, "stone_slab")

        # Click on top of the bottom slab: ny=1, hit_y=81.5
        hit = {
            "x": 5, "y": 81, "z": 5,
            "nx": 0, "ny": 1, "nz": 0,
            "hit_x": 5.5, "hit_y": 81.5, "hit_z": 5.5,
        }

        # 1. Place a sandstone slab into the top half
        res = calculate_sub_cell_placement(self.world, hit, "sandstone_slab")
        self.assertIsNotNone(res)
        cx, cy, cz, sub = res
        self.assertEqual((cx, cy, cz), (5, 81, 5))
        self.assertEqual(sub.get_octant(0, 0, 0), "stone")
        self.assertEqual(sub.get_octant(0, 1, 0), "sandstone")

        # 2. Place a vertical slab into an existing top slab (filling bottom half)
        self.world.set_block(6, 81, 6, "stone_slab_top")
        hit_top_slab = {
            "x": 6, "y": 81, "z": 6,
            "nx": 0, "ny": -1, "nz": 0,
            "hit_x": 6.5, "hit_y": 81.5, "hit_z": 6.5,
        }
        # player_yaw=0.0 faces North (oz=0)
        res_v = calculate_sub_cell_placement(self.world, hit_top_slab, "cobble_vertical_slab", player_yaw=0.0)
        self.assertIsNotNone(res_v)
        cx, cy, cz, sub_v = res_v
        self.assertEqual((cx, cy, cz), (6, 81, 6))
        self.assertEqual(sub_v.get_octant(0, 1, 0), "stone")
        self.assertEqual(sub_v.get_octant(0, 0, 0), "cobble")  # North half on bottom

    def test_horizontal_offset_placement_and_capping(self):
        from server.sub_cell import (
            calculate_horizontal_offset_placement,
            calculate_sub_cell_placement,
        )

        # Place vertical slab facing North at (5, 80, 5) -> occupies z in [5.0, 5.5]
        self.world.set_block(5, 80, 5, "cobble_vertical_slab_north")

        # 1. Aim at back face of vertical slab (at z=5.5, normal pointing South nz=1)
        hit_back = {
            "x": 5, "y": 80, "z": 5,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 5.5,
        }

        # Place full block "planks" against it -> horizontally offset from 5.5 to 6.5!
        res = calculate_horizontal_offset_placement(self.world, hit_back, "planks")
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 2)
        (c1x, c1y, c1z, sub1), (c2x, c2y, c2z, sub2) = res
        self.assertEqual((c1x, c1y, c1z), (5, 80, 5))
        self.assertEqual((c2x, c2y, c2z), (5, 80, 6))

        # Check octants in cell 1:
        self.assertEqual(sub1.get_octant(0, 0, 0), "cobble")  # Original vertical slab
        self.assertEqual(sub1.get_octant(0, 0, 1), "planks")  # Front half of offset planks
        # Check octants in cell 2:
        self.assertEqual(sub2.get_octant(0, 0, 0), "planks")  # Back half of offset planks
        self.assertEqual(sub2.get_octant(0, 0, 1), "")  # Open space

        # Apply to world
        self.world.set_sub_cell(c1x, c1y, c1z, sub1)
        self.world.set_sub_cell(c2x, c2y, c2z, sub2)

        # Raycast into the exposed South face of the offset planks at z=6.5
        hit_planks_south = self.world.raycast(5.5, 80.5, 8.0, 0.0, 0.0, -1.0, 5.0)
        self.assertIsNotNone(hit_planks_south)
        self.assertEqual((hit_planks_south["x"], hit_planks_south["y"], hit_planks_south["z"]), (5, 80, 6))
        self.assertAlmostEqual(hit_planks_south["hit_z"], 6.5)
        self.assertEqual(hit_planks_south["octant"], (1, 1, 0) if hit_planks_south["octant"][0] == 1 else (0, 0, 0) or hit_planks_south["octant"])

        # 2. Place a second full block "dirt" against the exposed South face -> offset from 6.5 to 7.5
        hit_dirt = {
            "x": 5, "y": 80, "z": 6,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 6.5,
            "octant": (0, 0, 0),
        }
        res2 = calculate_horizontal_offset_placement(self.world, hit_dirt, "dirt")
        self.assertIsNotNone(res2)
        (d1x, d1y, d1z, dsub1), (d2x, d2y, d2z, dsub2) = res2
        self.assertEqual((d1x, d1y, d1z), (5, 80, 6))
        self.assertEqual((d2x, d2y, d2z), (5, 80, 7))
        self.assertEqual(dsub1.get_octant(0, 0, 0), "planks")
        self.assertEqual(dsub1.get_octant(0, 0, 1), "dirt")
        self.assertEqual(dsub2.get_octant(0, 0, 0), "dirt")
        self.assertEqual(dsub2.get_octant(0, 0, 1), "")

        self.world.set_sub_cell(d1x, d1y, d1z, dsub1)
        self.world.set_sub_cell(d2x, d2y, d2z, dsub2)

        # 3. Cap the wall by placing a vertical slab facing North into the open South half of (5, 80, 7)
        hit_cap = {
            "x": 5, "y": 80, "z": 7,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 7.5,
            "octant": (0, 0, 0),
        }
        res_cap = calculate_sub_cell_placement(self.world, hit_cap, "cobble_vertical_slab", player_yaw=0.0)
        self.assertIsNotNone(res_cap)
        cap_x, cap_y, cap_z, cap_sub = res_cap
        self.assertEqual((cap_x, cap_y, cap_z), (5, 80, 7))
        self.assertEqual(cap_sub.get_octant(0, 0, 0), "dirt")
        self.assertEqual(cap_sub.get_octant(0, 0, 1), "cobble")  # Capped at z=8.0!

    def test_sub_cell_opacity_and_lighting(self):
        from server.blocks import BLOCKS
        from server.lighting import get_block_opacity

        sub_block = BLOCKS.get("sub_cell")
        self.assertIsNotNone(sub_block)
        # Sub-cell must have opacity 0 so daylight passes through open octants
        self.assertEqual(get_block_opacity(sub_block), 0)

    def test_raycast_offset_box_and_breaking(self):
        from server.blocks import AIR
        from server.sub_cell import calculate_horizontal_offset_placement

        # Setup: Vertical slab facing North at (5, 80, 5)
        self.world.set_block(5, 80, 5, "stone_vertical_slab_north")

        # Place full stone block horizontally offset against South back face
        hit_back = {
            "x": 5, "y": 80, "z": 5,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 5.5,
        }
        res = calculate_horizontal_offset_placement(self.world, hit_back, "stone")
        self.assertIsNotNone(res)
        for cx, cy, cz, sub in res:
            self.world.set_sub_cell(cx, cy, cz, sub)

        # Raycast into the offset stone block from South: ray at z=7.0 pointing -Z
        hit = self.world.raycast(5.5, 80.5, 7.0, 0.0, 0.0, -1.0, max_dist=5.0)
        self.assertIsNotNone(hit)
        # Bounding box should wrap around the offset stone block
        self.assertIn("box", hit)
        bx0, by0, bz0, bx1, by1, bz1 = hit["box"]
        self.assertAlmostEqual(bz1 - bz0, 1.0)  # Full 1.0m depth for the offset block!

        # Breaking the offset stone block:
        # Create a mock app or invoke the breaking logic directly
        from unittest.mock import MagicMock
        from client.app import Game

        from client.inventory import make_stack
        app = MagicMock(spec=Game)
        app.world = self.world
        app.target = hit
        app.gamemode = "survival"
        app.selected = 0
        app.hotbar = [make_stack("wood_pickaxe", 1)]
        app.network_client = None
        app._upload_block_change = MagicMock()
        app._spawn_break_drops = MagicMock()
        app._reset_breaking = MagicMock()

        # Call Game._break_block with app as self
        Game._break_block(app)

        # Verify drops: should drop full stone drop ("cobble"), NOT slab!
        app._spawn_break_drops.assert_called_once()
        drops = app._spawn_break_drops.call_args[0][0]
        self.assertEqual(drops, [("cobble", 1)])

        # Verify world state:
        # (5, 80, 6) should be completely empty (AIR)
        self.assertEqual(self.world.get_block(5, 80, 6), AIR)
        self.assertIsNone(self.world.get_sub_cell(5, 80, 6))

        # (5, 80, 5) should retain ONLY the stone vertical slab (4 octants at oz=0)
        sub5 = self.world.get_sub_cell(5, 80, 5)
        self.assertIsNotNone(sub5)
        self.assertEqual(sub5.count_occupied(), 4)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(sub5.get_octant(ox, oy, 0), "stone")
                self.assertEqual(sub5.get_octant(ox, oy, 1), "")

    def test_full_8_octant_sub_cell_breaks_as_full_block(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from client.inventory import make_stack
        from server.blocks import AIR

        # Create a SubCell where all 8 octants are smooth_sandstone at y=81 (within loaded chunk 0, 5, 0)
        cell = SubCell(["smooth_sandstone"] * 8)
        self.world.set_sub_cell(5, 81, 6, cell)

        # Raycast into it: must return full 1.0m cube box and sub_full
        hit = self.world.raycast(5.5, 81.5, 8.0, 0.0, 0.0, -1.0, max_dist=5.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["box"], (5.0, 81.0, 6.0, 6.0, 82.0, 7.0))
        self.assertEqual(hit["slab_half"], "sub_full")

        app = MagicMock(spec=Game)
        app.world = self.world
        app.target = hit
        app.gamemode = "survival"
        app.selected = 0
        app.hotbar = [make_stack("wood_pickaxe", 1)]
        app.network_client = None
        app._upload_block_change = MagicMock()
        app._spawn_break_drops = MagicMock()
        app._reset_breaking = MagicMock()

        Game._break_block(app)

        # Verify drops: full block drop, not slab!
        app._spawn_break_drops.assert_called_once()
        drops = app._spawn_break_drops.call_args[0][0]
        self.assertEqual(drops, [("smooth_sandstone", 1)])

        # Verify world state: cell is completely cleared to AIR
        self.assertEqual(self.world.get_block(5, 81, 6), AIR)
        self.assertIsNone(self.world.get_sub_cell(5, 81, 6))

    def test_offset_placement_preserves_rotation(self):
        from server.sub_cell import calculate_horizontal_offset_placement

        # Setup: Vertical slab facing North at (5, 80, 5)
        self.world.set_block(5, 80, 5, "stone_vertical_slab_north")

        # Place furnace_north horizontally offset against South back face
        hit_back = {
            "x": 5, "y": 80, "z": 5,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 5.5,
        }
        res = calculate_horizontal_offset_placement(self.world, hit_back, "furnace_north")
        self.assertIsNotNone(res)
        for cx, cy, cz, sub in res:
            self.world.set_sub_cell(cx, cy, cz, sub)

        # Check that furnace_north is preserved in both sub-cells
        sub1 = self.world.get_sub_cell(5, 80, 5)
        self.assertIsNotNone(sub1)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(sub1.get_octant(ox, oy, 1), "furnace_north")

        sub2 = self.world.get_sub_cell(5, 80, 6)
        self.assertIsNotNone(sub2)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(sub2.get_octant(ox, oy, 0), "furnace_north")

    def test_offset_crafting_table_and_furnace_interaction(self):
        from unittest.mock import MagicMock
        from client.app import Game

        # Setup: Crafting table placed horizontally offset at (5, 80, 5)-(5, 80, 6)
        sub1 = SubCell()
        sub2 = SubCell()
        for ox in (0, 1):
            for oy in (0, 1):
                sub1.set_octant(ox, oy, 1, "crafting_table")
                sub2.set_octant(ox, oy, 0, "crafting_table")
        self.world.set_sub_cell(5, 80, 5, sub1)
        self.world.set_sub_cell(5, 80, 6, sub2)

        # Raycast into the crafting table
        hit = self.world.raycast(5.5, 80.5, 7.0, 0.0, 0.0, -1.0, max_dist=5.0)
        self.assertIsNotNone(hit)

        app = MagicMock(spec=Game)
        app.world = self.world
        app.target = hit
        app._handle_mob_interaction = MagicMock(return_value=False)
        app._handle_bucket_interaction = MagicMock(return_value=False)
        app._handle_block_hit_interaction = MagicMock(return_value=False)
        app._handle_food_interaction = MagicMock(return_value=False)
        app._set_inventory = MagicMock()
        app._aim = MagicMock()

        # Call right click: should open crafting table!
        Game._on_right_click(app, modifiers=0)
        app._set_inventory.assert_called_once_with(True, container_type="crafting_table", container_block="crafting_table")

    def test_offset_block_uv_continuity(self):
        from server.blocks import BLOCKS
        from server.mesh import octant_face_uvs

        block = BLOCKS["smooth_sandstone"]
        # Normal 2 is Top (+Y)
        # Front half (cell A, oz=1) with off_z=0.0:
        uvs_front = octant_face_uvs(block, 2, 0, 1, 1, off_z=0.0)
        # Back half (cell B, oz=0) with off_z=0.5:
        uvs_back = octant_face_uvs(block, 2, 0, 1, 0, off_z=0.5)

        # In CORNERS[2]:
        # k=0: [0, 1, 1] (Z=1.0 of the half)
        # k=1: [1, 1, 1] (Z=1.0 of the half)
        # k=2: [1, 1, 0] (Z=0.0 of the half)
        # k=3: [0, 1, 0] (Z=0.0 of the half)
        # For top face: s = X, t = 1.0 - Z
        # For front half (off_z=0.0): Z spans 0.0 to 0.5 => t spans 1.0 to 0.5.
        # At k=0,1: Z = 0.5 => t = 0.5.
        # At k=2,3: Z = 0.0 => t = 1.0.
        # For back half (off_z=0.5): Z spans 0.5 to 1.0 => t spans 0.5 to 0.0.
        # At k=0,1: Z = 1.0 => t = 0.0.
        # At k=2,3: Z = 0.5 => t = 0.5.
        # Notice that at the shared boundary (k=0,1 of front half and k=2,3 of back half), t == 0.5 for BOTH!
        t_front_boundary = uvs_front[0][1]
        t_back_boundary = uvs_back[2][1]
        self.assertAlmostEqual(t_front_boundary, t_back_boundary)

    def test_two_offset_blocks_sharing_cell_do_not_split_both(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from client.inventory import make_stack
        from server.sub_cell import calculate_horizontal_offset_placement

        # 1. Setup: Vertical slab facing North at (5, 80, 5) (occupies oz=0)
        self.world.set_block(5, 80, 5, "stone_vertical_slab_north")

        # 2. Place first offset planks block against back face (nz=1)
        hit1 = {
            "x": 5, "y": 80, "z": 5,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 5.5,
        }
        res1 = calculate_horizontal_offset_placement(self.world, hit1, "planks")
        self.assertIsNotNone(res1)
        for cx, cy, cz, sub in res1:
            self.world.set_sub_cell(cx, cy, cz, sub)

        # 3. Place second offset planks block against back face of first offset block (nz=1 at z=6)
        # Raycast into back of first offset block (at z=6.5)
        hit2 = {
            "x": 5, "y": 80, "z": 6,
            "nx": 0, "ny": 0, "nz": 1,
            "hit_x": 5.5, "hit_y": 80.5, "hit_z": 6.5,
            "octant": (0, 0, 0),
        }
        res2 = calculate_horizontal_offset_placement(self.world, hit2, "planks")
        self.assertIsNotNone(res2)
        for cx, cy, cz, sub in res2:
            self.world.set_sub_cell(cx, cy, cz, sub)

        # Verify Cell (5, 80, 6) now has ALL 8 octants as planks!
        mid_cell = self.world.get_sub_cell(5, 80, 6)
        self.assertIsNotNone(mid_cell)
        self.assertEqual(mid_cell.count_occupied(), 8)
        self.assertTrue(all(mid_cell.octants[i] == "planks" for i in range(8)))

        # 4. Raycast into the middle cell from the top, aiming at the first offset block half (z = 6.25, oz=0)
        # Ray from above: origin (5.5, 83.0, 6.25) pointing -Y
        hit = self.world.raycast(5.5, 83.0, 6.25, 0.0, -1.0, 0.0, max_dist=5.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["x"], 5)
        self.assertEqual(hit["y"], 80)
        self.assertEqual(hit["z"], 6)
        # The octant must resolve to oz=0 (belonging to Offset Block 1)
        self.assertEqual(hit.get("octant")[2], 0)

        # 5. Break the block
        app = MagicMock(spec=Game)
        app.world = self.world
        app.target = hit
        app.gamemode = "survival"
        app.selected = 0
        app.hotbar = [make_stack("wood_pickaxe", 1)]
        app.network_client = None
        app._upload_block_change = MagicMock()
        app._spawn_break_drops = MagicMock()
        app._reset_breaking = MagicMock()

        Game._break_block(app)

        # Drops: must drop 1 full planks!
        app._spawn_break_drops.assert_called_once()
        drops = app._spawn_break_drops.call_args[0][0]
        self.assertEqual(drops, [("planks", 1)])

        # World State check:
        # 1. Cell (5, 80, 5): vertical slab at oz=0 must remain intact, oz=1 cleared
        sub5 = self.world.get_sub_cell(5, 80, 5)
        self.assertIsNotNone(sub5)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(sub5.get_octant(ox, oy, 0), "stone")
                self.assertEqual(sub5.get_octant(ox, oy, 1), "")

        # 2. Cell (5, 80, 6): oz=0 was cleared, BUT oz=1 MUST REMAIN INTACT as planks!
        sub6 = self.world.get_sub_cell(5, 80, 6)
        self.assertIsNotNone(sub6)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(sub6.get_octant(ox, oy, 0), "")
                self.assertEqual(sub6.get_octant(ox, oy, 1), "planks")

        # 3. Cell (5, 80, 7): oz=0 MUST REMAIN INTACT as planks!
        sub7 = self.world.get_sub_cell(5, 80, 7)
        self.assertIsNotNone(sub7)
        for ox in (0, 1):
            for oy in (0, 1):
                self.assertEqual(sub7.get_octant(ox, oy, 0), "planks")
                self.assertEqual(sub7.get_octant(ox, oy, 1), "")


if __name__ == "__main__":
    unittest.main()
