import unittest
import math

from server.blocks import (
    AIR,
    BLOCKS,
    STONE,
    get_double_slab,
    get_slab_info,
    is_slab,
    is_waterlogged_slab,
    placement_block,
)
from server.entity import ItemEntity
from server.fluid import (
    WATER_SOURCE,
    get_fluid_level,
    is_water,
    is_water_source,
)
from server.player import Player
from server.world import World


class TestSlabs(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=123)
        self.cx, self.cy, self.cz = 0, 5, 0
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

    def test_slab_registration_and_properties(self):
        self.assertTrue(is_slab("stone_slab"))
        self.assertTrue(is_slab("stone_slab_top"))
        self.assertTrue(is_slab("stone_slab_waterlogged"))
        self.assertTrue(is_slab("stone_slab_top_waterlogged"))
        self.assertTrue(is_slab("cobble_slab"))
        self.assertTrue(is_slab("planks_slab"))
        self.assertTrue(is_slab("double_slab_stone_slab_planks_slab"))

        info_bot = get_slab_info("stone_slab")
        self.assertIsNotNone(info_bot)
        self.assertEqual(info_bot["half"], "bottom")
        self.assertFalse(info_bot["double"])
        self.assertFalse(info_bot["waterlogged"])
        self.assertEqual(info_bot["base_id"], "stone_slab")

        info_top = get_slab_info("stone_slab_top")
        self.assertIsNotNone(info_top)
        self.assertEqual(info_top["half"], "top")
        self.assertFalse(info_top["double"])

        info_wl = get_slab_info("stone_slab_waterlogged")
        self.assertIsNotNone(info_wl)
        self.assertTrue(info_wl["waterlogged"])
        self.assertTrue(is_waterlogged_slab("stone_slab_waterlogged"))
        self.assertTrue(is_water("stone_slab_waterlogged"))

        info_double = get_slab_info("double_slab_stone_slab_planks_slab")
        self.assertIsNotNone(info_double)
        self.assertTrue(info_double["double"])
        self.assertEqual(info_double["bottom"], "stone_slab")
        self.assertEqual(info_double["top"], "planks_slab")

    def test_sandstone_and_smooth_sandstone_slabs_register(self):
        self.assertTrue(is_slab("sandstone_slab"))
        self.assertTrue(is_slab("smooth_sandstone_slab"))
        self.assertTrue(is_slab("sandstone_slab_top"))
        self.assertTrue(is_slab("smooth_sandstone_slab_top"))

        sandstone_info = get_slab_info("sandstone_slab")
        self.assertIsNotNone(sandstone_info)
        self.assertTrue(sandstone_info["is_slab"])
        self.assertFalse(sandstone_info["double"])
        self.assertEqual(sandstone_info["half"], "bottom")
        self.assertEqual(sandstone_info["base_id"], "sandstone_slab")
        self.assertEqual(sandstone_info["top"], "sandstone_slab")

        smooth_info = get_slab_info("smooth_sandstone_slab")
        self.assertIsNotNone(smooth_info)
        self.assertEqual(smooth_info["base_id"], "smooth_sandstone_slab")

        top_info = get_slab_info("smooth_sandstone_slab_top")
        self.assertIsNotNone(top_info)
        self.assertEqual(top_info["half"], "top")

    def test_collision_boxes(self):
        self.world.set_block(1, 80, 1, "stone_slab")
        boxes = self.world.get_collision_boxes(1, 80, 1)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (1.0, 80.0, 1.0, 2.0, 80.5, 2.0))

        self.world.set_block(2, 80, 2, "stone_slab_top")
        top_boxes = self.world.get_collision_boxes(2, 80, 2)
        self.assertEqual(len(top_boxes), 1)
        self.assertEqual(top_boxes[0], (2.0, 80.5, 2.0, 3.0, 81.0, 3.0))

        self.world.set_block(3, 80, 3, "double_slab_stone_slab_planks_slab")
        double_boxes = self.world.get_collision_boxes(3, 80, 3)
        self.assertEqual(len(double_boxes), 1)
        self.assertEqual(double_boxes[0], (3.0, 80.0, 3.0, 4.0, 81.0, 4.0))

    def test_player_standing_on_slab(self):
        # Bottom slab at y=80
        self.world.set_block(1, 80, 1, "stone_slab")
        player = Player(self.world)
        # Position at y=80.5 (on top of bottom slab)
        player.pos = [1.5, 80.5, 1.5]
        self.assertFalse(player.collides_at(*player.pos))

        # Position slightly intersecting slab
        player.pos[1] = 80.45
        self.assertTrue(player.collides_at(*player.pos))

    def test_item_entity_landing_on_slab(self):
        self.world.set_block(1, 80, 1, "stone_slab")
        item = ItemEntity("dirt", pos=[1.5, 82.0, 1.5], vel=[0.0, -5.0, 0.0])
        self.world.entities.append(item)

        for _ in range(20):
            item.update(0.05, self.world)

        # Landed on bottom slab at y=80.5
        self.assertAlmostEqual(item.pos[1], 80.5, places=2)
        self.assertTrue(item.on_ground)

    def test_raycast_through_open_half_of_slab(self):
        # Bottom slab at x=2, y=80, z=0 (occupies y: 80.0 to 80.5)
        self.world.set_block(2, 80, 0, "stone_slab")
        # Full stone block behind it at x=4, y=80, z=0
        self.world.set_block(4, 80, 0, STONE)

        # Ray aiming at upper half (y=80.75): should pass through open top half of slab and hit stone block
        hit = self.world.raycast(0.0, 80.75, 0.5, 1.0, 0.0, 0.0, 10.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["x"], 4)

        # Ray aiming at lower half (y=80.25): should hit the bottom slab
        hit_slab = self.world.raycast(0.0, 80.25, 0.5, 1.0, 0.0, 0.0, 10.0)
        self.assertIsNotNone(hit_slab)
        self.assertEqual(hit_slab["x"], 2)
        self.assertEqual(hit_slab["slab_half"], "bottom")

    def test_raycast_double_slab_top_and_bottom_detection(self):
        self.world.set_block(2, 80, 0, "double_slab_stone_slab_planks_slab")

        # Aim at top half
        hit_top = self.world.raycast(0.0, 80.75, 0.5, 1.0, 0.0, 0.0, 10.0)
        self.assertIsNotNone(hit_top)
        self.assertEqual(hit_top["x"], 2)
        self.assertEqual(hit_top["slab_half"], "top")

        # Aim at bottom half
        hit_bot = self.world.raycast(0.0, 80.25, 0.5, 1.0, 0.0, 0.0, 10.0)
        self.assertIsNotNone(hit_bot)
        self.assertEqual(hit_bot["x"], 2)
        self.assertEqual(hit_bot["slab_half"], "bottom")

    def test_breaking_double_slab_halves(self):
        from client.app import Game
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world):
                self.world = world
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack("wood_pickaxe", 1)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._break_block = Game._break_block
        MockGame._aim = lambda self: None

        game = MockGame(self.world)

        # 1. Break top half of double slab
        self.world.set_block(5, 80, 5, "double_slab_stone_slab_planks_slab")
        game.target = {"x": 5, "y": 80, "z": 5, "slab_half": "top", "local_y": 0.75}
        game._break_block()
        # Remaining block should be bottom slab (stone_slab)
        self.assertEqual(self.world.get_block(5, 80, 5), "stone_slab")
        # Should have dropped planks_slab item
        planks_drops = [e for e in self.world.entities if isinstance(e, ItemEntity) and e.item == "planks_slab"]
        self.assertGreaterEqual(len(planks_drops), 1)

        # 2. Break bottom half of double slab
        self.world.set_block(6, 80, 6, "double_slab_stone_slab_planks_slab")
        game.target = {"x": 6, "y": 80, "z": 6, "slab_half": "bottom", "local_y": 0.25}
        game._break_block()
        # Remaining block should be top slab (planks_slab_top)
        self.assertEqual(self.world.get_block(6, 80, 6), "planks_slab_top")
        stone_drops = [e for e in self.world.entities if isinstance(e, ItemEntity) and e.item == "stone_slab"]
        self.assertGreaterEqual(len(stone_drops), 1)

    def test_waterlogged_slab_breaking_leaves_water(self):
        from client.app import Game
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world):
                self.world = world
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack("wood_pickaxe", 1)]
                self.selected = 0
                self.target = {"x": 2, "y": 80, "z": 2, "slab_half": "bottom", "local_y": 0.25}

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._break_block = Game._break_block
        MockGame._aim = lambda self: None

        # 1. Source waterlogged slab leaves water source
        self.world.set_block(2, 80, 2, "stone_slab_waterlogged")
        game = MockGame(self.world)
        game._break_block()
        self.assertEqual(self.world.get_block(2, 80, 2), "water")

        # 2. Flowing waterlogged bottom slab leaves flowing water with same level
        self.world.set_block(2, 80, 2, "stone_slab_waterlogged_4")
        game = MockGame(self.world)
        game._break_block()
        self.assertEqual(self.world.get_block(2, 80, 2), "water_4")

        # 3. Flowing waterlogged top slab leaves flowing water with same level
        self.world.set_block(2, 80, 2, "stone_slab_top_waterlogged_2")
        game = MockGame(self.world)
        game._break_block()
        self.assertEqual(self.world.get_block(2, 80, 2), "water_2")

        # 4. Dry slab leaves air
        self.world.set_block(2, 80, 2, "stone_slab")
        game = MockGame(self.world)
        game._break_block()
        self.assertEqual(self.world.get_block(2, 80, 2), AIR)

    def test_bucket_waterlogging(self):
        from client.app import Game
        from server.stacks import make_stack, stack_id

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.player = Player(self.world)
                self.eye = (2.5, 85.0, 2.5)
                self.forward = (0.0, -1.0, 0.0)
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack(held_item, 1)]
                self.storage = []
                self.selected = 0

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._handle_bucket_interaction = Game._handle_bucket_interaction

        # Place dry bottom slab
        self.world.set_block(2, 80, 2, "stone_slab")

        # 1. Right click with water bucket waterlogs it
        waterlog_game = MockGame(self.world, "water_bucket")
        self.assertTrue(waterlog_game._handle_bucket_interaction())
        self.assertEqual(self.world.get_block(2, 80, 2), "stone_slab_waterlogged")
        self.assertEqual(stack_id(waterlog_game.hotbar[0]), "iron_bucket")

        # 2. Right click with empty bucket unlogs it
        empty_game = MockGame(self.world, "iron_bucket")
        self.assertTrue(empty_game._handle_bucket_interaction())
        self.assertEqual(self.world.get_block(2, 80, 2), "stone_slab")
        self.assertEqual(stack_id(empty_game.hotbar[0]), "water_bucket")

    def test_fluid_flows_over_bottom_slab_only_if_over_half_height(self):
        from server.fluid import compute_new_fluid_state
        # Floor at y=79
        for x in range(0, 5):
            self.world.set_block(x, 79, 0, STONE)

        # Place bottom slab at x=2, y=80
        self.world.set_block(2, 80, 0, "stone_slab")

        # Case 1: Low water level (level 4, exactly half height or less) at x=1
        self.world.set_block(1, 80, 0, "water_4")
        # Does NOT flow over bottom slab
        self.assertEqual(compute_new_fluid_state(self.world, 2, 80, 0), "stone_slab")

        # Case 2: Water level > 4 (level 6, over half height) at x=1
        self.world.set_block(1, 80, 0, "water_6")
        # Flows over bottom slab with level 5
        self.assertEqual(compute_new_fluid_state(self.world, 2, 80, 0), "stone_slab_waterlogged_5")

        # Recedes when water is removed
        self.world.set_block(2, 80, 0, "stone_slab_waterlogged_5")
        self.world.set_block(1, 80, 0, AIR)
        self.assertEqual(compute_new_fluid_state(self.world, 2, 80, 0), "stone_slab")

    def test_fluid_flows_under_top_slab(self):
        from server.fluid import compute_new_fluid_state
        # Floor at y=79
        for x in range(0, 5):
            self.world.set_block(x, 79, 0, STONE)

        # Top slab at x=2, y=80
        self.world.set_block(2, 80, 0, "stone_slab_top")

        # Low level water at x=1, y=80 (level 2)
        self.world.set_block(1, 80, 0, "water_2")

        # Flows under top slab with level 1
        self.assertEqual(compute_new_fluid_state(self.world, 2, 80, 0), "stone_slab_top_waterlogged_1")

        # Recedes when water is removed
        self.world.set_block(2, 80, 0, "stone_slab_top_waterlogged_1")
        self.world.set_block(1, 80, 0, AIR)
        self.assertEqual(compute_new_fluid_state(self.world, 2, 80, 0), "stone_slab_top")

    def test_placing_slabs_into_double_slabs(self):
        from client.app import Game
        from server.stacks import make_stack, stack_count

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.player = Player(self.world)
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack(held_item, 2)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._place_block = Game._place_block
        MockGame._aim = lambda self: None

        # Existing bottom stone_slab at (3, 80, 3)
        self.world.set_block(3, 80, 3, "stone_slab")

        # Hold planks_slab and click top face (ny=1)
        game = MockGame(self.world, "planks_slab")
        game.target = {"x": 3, "y": 80, "z": 3, "nx": 0, "ny": 1, "nz": 0, "local_y": 0.5}
        game._place_block()

        # Should combine into mixed double slab!
        self.assertEqual(self.world.get_block(3, 80, 3), "double_slab_stone_slab_planks_slab")
        self.assertEqual(stack_count(game.hotbar[0]), 1)

    def test_slab_neighbor_face_not_culled(self):
        from server.blocks import DIRT
        from server.mesh import build_chunk_mesh

        # Clear chunk and set dirt at (1, 80, 0) and top slab at (0, 80, 0)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)
        self.world.set_block(1, 80, 0, DIRT)
        self.world.set_block(0, 80, 0, "stone_slab_top")

        mesh = build_chunk_mesh(self.world, self.cx, self.cy, self.cz)
        self.assertIsNotNone(mesh)

        # Dirt at (1, 80, 0) must NOT cull its west face (-X) facing the slab
        has_dirt_west = False
        for i in range(0, len(mesh.vertices), 4):
            v = mesh.vertices[i]
            norm = (round(float(v[3])), round(float(v[4])), round(float(v[5])))
            pos = (round(float(v[0])), round(float(v[1])), round(float(v[2])))
            if norm == (-1, 0, 0) and pos[0] == 1:
                has_dirt_west = True
                break
        self.assertTrue(has_dirt_west, "Dirt face facing open half of slab was culled")

    def test_mixed_double_slab_mesh_both_textures(self):
        from server.mesh import build_chunk_mesh

        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)
        dslab = get_double_slab("stone_slab", "planks_slab")
        self.world.set_block(0, 80, 0, dslab)

        mesh = build_chunk_mesh(self.world, self.cx, self.cy, self.cz)
        self.assertIsNotNone(mesh)

        # Double slab must render 10 quads (5 for bottom half, 5 for top half, no internal face)
        self.assertEqual(len(mesh.vertices), 40)

        # Collect distinct UVs across bottom (y=0) and top (y=0.5) quads
        bottom_uvs = set()
        top_uvs = set()
        for i in range(0, len(mesh.vertices), 4):
            v = mesh.vertices[i]
            uv = (round(float(v[6]), 3), round(float(v[7]), 3))
            if round(float(v[1]), 2) == 0.0:
                bottom_uvs.add(uv)
            else:
                top_uvs.add(uv)

        self.assertTrue(len(bottom_uvs) > 0)
        self.assertTrue(len(top_uvs) > 0)
        # The textures used for bottom and top halves must not be identical
        self.assertNotEqual(bottom_uvs, top_uvs)

    def test_slab_lighting_opacity_and_underside_sample(self):
        from server.lighting import get_block_opacity
        from server.blocks import BLOCKS
        from server.mesh import build_chunk_mesh

        # Half slabs must have opacity 0 so light passes through
        self.assertEqual(get_block_opacity(BLOCKS["stone_slab"]), 0)
        self.assertEqual(get_block_opacity(BLOCKS["stone_slab_top"]), 0)
        # Double slabs must block light (opacity 15)
        self.assertEqual(get_block_opacity(BLOCKS["double_slab_stone_slab"]), 15)

        # Place top slab in open air at (1, 80, 1)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)
        self.world.set_block(1, 80, 1, "stone_slab_top")

        mesh = build_chunk_mesh(self.world, self.cx, self.cy, self.cz)
        self.assertIsNotNone(mesh)

        # The underside face of the top slab (normal=(0, -1, 0) at y=0.5) must sample full skylight (1.0)
        underside_light = None
        for i in range(len(mesh.vertices)):
            v = mesh.vertices[i]
            norm = (round(float(v[3])), round(float(v[4])), round(float(v[5])))
            pos = (round(float(v[0]), 2), round(float(v[1]), 2), round(float(v[2]), 2))
            if norm == (0, -1, 0) and pos[1] == 0.5:
                underside_light = mesh.lights[i][0]
                break
        self.assertIsNotNone(underside_light)
        self.assertGreaterEqual(underside_light, 0.9)

    def test_slab_held_item_mesh_half_height(self):
        from client.player_render import held_item_mesh
        import numpy as np

        identity = np.eye(4, dtype=np.float32)
        full_mesh, _ = held_item_mesh("stone", identity)
        slab_mesh, _ = held_item_mesh("stone_slab", identity)

        full_verts = full_mesh[0].reshape(-1, 8)
        slab_verts = slab_mesh[0].reshape(-1, 8)

        full_height = full_verts[:, 1].max() - full_verts[:, 1].min()
        slab_height = slab_verts[:, 1].max() - slab_verts[:, 1].min()

        self.assertAlmostEqual(full_height, 0.5, places=2)
        # Slab item mesh should be half the height of a full block mesh
        self.assertAlmostEqual(slab_height, 0.25, places=2)

    def test_raycast_double_slab_section_box(self):
        # Mixed double slab at (0, 80, 0): bottom = stone_slab, top = planks_slab
        dslab = get_double_slab("stone_slab", "planks_slab")
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)
        self.world.set_block(0, 80, 0, dslab)

        # Aim at bottom half (y=80.25)
        hit_bottom = self.world.raycast(0.5, 80.25, 3.0, 0, 0, -1, max_dist=5.0)
        self.assertIsNotNone(hit_bottom)
        self.assertEqual(hit_bottom["slab_half"], "bottom")
        self.assertEqual(hit_bottom["box"], (0.0, 80.0, 0.0, 1.0, 80.5, 1.0))

        # Aim at top half (y=80.75)
        hit_top = self.world.raycast(0.5, 80.75, 3.0, 0, 0, -1, max_dist=5.0)
        self.assertIsNotNone(hit_top)
        self.assertEqual(hit_top["slab_half"], "top")
        self.assertEqual(hit_top["box"], (0.0, 80.5, 0.0, 1.0, 81.0, 1.0))

    def test_breaking_mixed_double_slab_mining_properties(self):
        from client.app import Game
        from server.stacks import make_stack
        from server.items import ITEMS

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.gamemode = "survival"
                self.hotbar = [make_stack(held_item, 1)]
                self.selected = 0
                self.target = None
                self._breaking_block = None
                self._break_time = 0.0
                self._break_required = 0.0
                self._break_progress = 0.0

            def _reset_breaking(self):
                self._breaking_block = None
                self._break_time = 0.0
                self._break_required = 0.0
                self._break_progress = 0.0

            def _break_block(self):
                pass

        MockGame._start_breaking = Game._start_breaking

        dslab = get_double_slab("stone_slab", "planks_slab")
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)
        self.world.set_block(0, 80, 0, dslab)

        # Case 1: Pickaxe vs bottom half (stone) vs top half (wood)
        game_pick = MockGame(self.world, "wood_pickaxe")

        # Aim at bottom (stone)
        target_bottom = {"x": 0, "y": 80, "z": 0, "slab_half": "bottom", "local_y": 0.25}
        game_pick._start_breaking(target_bottom)
        time_stone_with_pick = game_pick._break_required

        # Aim at top (wood)
        target_top = {"x": 0, "y": 80, "z": 0, "slab_half": "top", "local_y": 0.75}
        game_pick._start_breaking(target_top)
        time_wood_with_pick = game_pick._break_required

        # Pickaxe is effective on stone slab but NOT wood slab, so stone breaks much faster with pickaxe!
        self.assertLess(time_stone_with_pick, time_wood_with_pick)

    def test_waterlogged_slab_fluid_mesh_uvs_and_height(self):
        from server.mesh import build_chunk_mesh
        import numpy as np

        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)
        # Place waterlogged bottom slab at (2, 80, 2) with level 6 (height = 12/16 = 0.75)
        self.world.set_block(2, 80, 2, "cobble_slab_waterlogged_6")
        mesh = build_chunk_mesh(self.world, self.cx, self.cy, self.cz)
        self.assertIsNotNone(mesh)
        self.assertIsNotNone(mesh.translucent)

        # Translucent mesh should have water vertices
        t_verts = mesh.translucent[0]
        self.assertGreater(len(t_verts), 0)
        water_uv = BLOCKS["water"].face_uvs[2]
        cobble_uv = BLOCKS["cobble_slab"].face_uvs[2]

        # Top face should use water UV, not cobblestone UV!
        # Top face normal is (0, 1, 0)
        top_quads = [
            t_verts[i:i+4] for i in range(0, len(t_verts), 4)
            if np.allclose(t_verts[i:i+4, 3:6], [0, 1, 0])
        ]
        self.assertGreater(len(top_quads), 0)
        top_quad = top_quads[0]
        # Height of top quad should be 0.75 (chunk-local y=0 + 0.75), not full block 1.0
        self.assertAlmostEqual(float(top_quad[0, 1]), 0.75, places=2)
        # UV should match water, not cobble
        uv_diff_water = np.max(np.abs(top_quad[:, 6:8] - np.array(water_uv)))
        uv_diff_cobble = np.max(np.abs(top_quad[:, 6:8] - np.array(cobble_uv)))
        self.assertAlmostEqual(uv_diff_water, 0.0, places=4)
        self.assertGreater(uv_diff_cobble, 0.05)

    def test_offset_full_blocks_on_slabs_and_capping_slab(self):
        from client.app import Game
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.player = Player(self.world)
                self.player.pos = [10.5, 80.0, 10.5]
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack(held_item, 10)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._place_block = Game._place_block
        MockGame._aim = lambda self: None

        # 1. Place bottom slab at (4, 80, 4) -> occupies [80.0, 80.5]
        self.world.set_block(4, 80, 4, "stone_slab")

        # 2. Place full block "planks" on top of the bottom slab at (4, 80, 4)
        game = MockGame(self.world, "planks")
        game.target = self.world.raycast(4.5, 82.0, 4.5, 0, -1, 0, max_dist=5.0)
        self.assertEqual((game.target["x"], game.target["y"], game.target["z"]), (4, 80, 4))
        game._place_block()

        # Should place "planks_offset" at (4, 81, 4), occupying [80.5, 81.5]
        self.assertEqual(self.world.get_block(4, 81, 4), "planks_offset")
        self.assertEqual(self.world.get_collision_boxes(4, 81, 4), [(4.0, 80.5, 4.0, 5.0, 81.5, 5.0)])

        # 3. Place another full block "cobble" on top of the offset planks at (4, 81, 4)
        game.hotbar = [make_stack("cobble", 10)]
        game.target = self.world.raycast(4.5, 83.0, 4.5, 0, -1, 0, max_dist=5.0)
        self.assertEqual((game.target["x"], game.target["y"], game.target["z"]), (4, 81, 4))
        self.assertEqual(game.target["box"], (4.0, 80.5, 4.0, 5.0, 81.5, 5.0))
        game._place_block()

        # Should place "cobble_offset" at (4, 82, 4), occupying [81.5, 82.5]
        self.assertEqual(self.world.get_block(4, 82, 4), "cobble_offset")
        self.assertEqual(self.world.get_collision_boxes(4, 82, 4), [(4.0, 81.5, 4.0, 5.0, 82.5, 5.0)])

        # 4. Place a slab "stone_slab" on top of the offset cobble at (4, 82, 4)
        game.hotbar = [make_stack("stone_slab", 10)]
        game.target = self.world.raycast(4.5, 84.0, 4.5, 0, -1, 0, max_dist=5.0)
        self.assertEqual((game.target["x"], game.target["y"], game.target["z"]), (4, 82, 4))
        game._place_block()

        # Should cap (4, 82, 4) into "cobble_offset_stone_slab", occupying [81.5, 83.0]
        self.assertEqual(self.world.get_block(4, 82, 4), "cobble_offset_stone_slab")
        self.assertEqual(self.world.get_collision_boxes(4, 82, 4), [(4.0, 81.5, 4.0, 5.0, 83.0, 5.0)])

        # 5. Place another full block "dirt" on top of the capped offset block at (4, 82, 4)
        game.hotbar = [make_stack("dirt", 10)]
        game.target = self.world.raycast(4.5, 85.0, 4.5, 0, -1, 0, max_dist=5.0)
        self.assertEqual((game.target["x"], game.target["y"], game.target["z"]), (4, 82, 4))
        game._place_block()

        # Since the offset stack was capped by a slab, the new full block is normal grid-aligned "dirt" at (4, 83, 4)!
        self.assertEqual(self.world.get_block(4, 83, 4), "dirt")
        self.assertEqual(self.world.get_collision_boxes(4, 83, 4), [(4.0, 83.0, 4.0, 5.0, 84.0, 5.0)])

    def test_breaking_capped_offset_block_parts(self):
        from client.app import Game
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world):
                self.world = world
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack("wood_pickaxe", 1)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._break_block = Game._break_block
        MockGame._aim = lambda self: None

        game = MockGame(self.world)

        # 1. Break top slab cap of capped offset block -> leaves uncapped offset block
        self.world.set_block(5, 81, 5, "planks_offset_stone_slab")
        hit_top = self.world.raycast(5.5, 81.75, 8.0, 0, 0, -1, max_dist=5.0)
        self.assertIsNotNone(hit_top)
        self.assertEqual(hit_top["slab_half"], "top")
        self.assertEqual(hit_top["box"], (5.0, 81.5, 5.0, 6.0, 82.0, 6.0))
        game.target = hit_top
        game._break_block()
        self.assertEqual(self.world.get_block(5, 81, 5), "planks_offset")

        # 2. Break offset block part of capped offset block -> leaves top slab
        self.world.set_block(6, 81, 6, "planks_offset_stone_slab")
        hit_off = self.world.raycast(6.5, 81.0, 9.0, 0, 0, -1, max_dist=5.0)
        self.assertIsNotNone(hit_off)
        self.assertEqual(hit_off["slab_half"], "offset")
        self.assertEqual(hit_off["box"], (6.0, 80.5, 6.0, 7.0, 81.5, 7.0))
        game.target = hit_off
        game._break_block()
        self.assertEqual(self.world.get_block(6, 81, 6), "stone_slab_top")

    def test_cannot_place_offset_block_into_torch_half_block_lower(self):
        from client.app import Game
        from server.blocks import can_place_block
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.player = Player(self.world)
                self.player.pos = [0.0, 90.0, 0.0]
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack(held_item, 10)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._place_block = Game._place_block
        MockGame._aim = lambda self: None

        # Place a bottom slab at (4, 80, 4) and an offset block on top at (4, 81, 4) ([80.5, 81.5])
        self.world.set_block(4, 80, 4, "stone_slab")
        self.world.set_block(4, 81, 4, "planks_offset")

        # Place a floor block at (5, 79, 4) and a torch at (5, 80, 4) (half a block lower than planks_offset at y=81)
        self.world.set_block(5, 79, 4, STONE)
        self.world.set_block(5, 80, 4, "torch")

        # Aim at the +x side of the offset block at (4, 81, 4)
        hit_side = self.world.raycast(7.0, 81.0, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_side)
        self.assertEqual((hit_side["x"], hit_side["y"], hit_side["z"]), (4, 81, 4))
        self.assertEqual((hit_side["nx"], hit_side["ny"], hit_side["nz"]), (1, 0, 0))
        self.assertEqual(hit_side["slab_half"], "offset")

        # Attempting to place planks against the side of (4, 81, 4) into (5, 81, 4) must be blocked by the torch at (5, 80, 4)
        game = MockGame(self.world, "planks")
        game.target = hit_side
        game._place_block()
        self.assertEqual(self.world.get_block(5, 81, 4), AIR)
        self.assertEqual(self.world.get_block(5, 80, 4), "torch")

        # Also cannot place a torch at (3, 80, 4) underneath an existing offset block at (3, 81, 4)
        self.world.set_block(3, 79, 4, STONE)
        self.world.set_block(3, 81, 4, "planks_offset")
        self.assertFalse(can_place_block(self.world, 3, 80, 4, "torch"))

    def test_place_slab_on_bottom_and_top_half_of_offset_block(self):
        from client.app import Game
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.player = Player(self.world)
                self.player.pos = [0.0, 90.0, 0.0]
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack(held_item, 10)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._place_block = Game._place_block
        MockGame._aim = lambda self: None

        # Offset block at (4, 81, 4) occupies y in [80.5, 81.5]
        self.world.set_block(4, 80, 4, "stone_slab")
        self.world.set_block(4, 81, 4, "planks_offset")

        game = MockGame(self.world, "stone_slab")

        # 1. Aim at the bottom half [80.5, 81.0) of the offset block at (4, 81, 4) from +x side (hy = 80.75)
        hit_bottom = self.world.raycast(7.0, 80.75, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_bottom)
        self.assertEqual((hit_bottom["x"], hit_bottom["y"], hit_bottom["z"]), (4, 81, 4))
        self.assertLess(hit_bottom["local_y"], 0.0)
        game.target = hit_bottom
        game._place_block()

        # Should place a top slab at (5, 80, 4) occupying [80.5, 81.0], matching the bottom half of the offset block!
        self.assertEqual(self.world.get_block(5, 80, 4), "stone_slab_top")
        self.assertEqual(self.world.get_block(5, 81, 4), AIR)

        # 2. Aim at the top half [81.0, 81.5) of the offset block at (4, 81, 4) from +x side (hy = 81.25)
        hit_top = self.world.raycast(7.0, 81.25, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_top)
        self.assertEqual((hit_top["x"], hit_top["y"], hit_top["z"]), (4, 81, 4))
        self.assertGreaterEqual(hit_top["local_y"], 0.0)
        game.target = hit_top
        game._place_block()

        # Should place a bottom slab at (5, 81, 4) occupying [81.0, 81.5], matching the top half of the offset block!
        self.assertEqual(self.world.get_block(5, 81, 4), "stone_slab")

    def test_torch_on_top_of_bottom_slab_at_half_height(self):
        from client.app import Game
        from server.stacks import make_stack

        class MockGame:
            def __init__(self, world, held_item):
                self.world = world
                self.player = Player(self.world)
                self.player.pos = [0.0, 90.0, 0.0]
                self.gamemode = "survival"
                self.network_client = None
                self.hotbar = [make_stack(held_item, 10)]
                self.selected = 0
                self.target = None

            def _upload_block_change(self, x, y, z):
                pass

        MockGame._place_block = Game._place_block
        MockGame._break_block = Game._break_block
        MockGame._aim = lambda self: None

        # Place bottom slab at (4, 80, 4)
        self.world.set_block(4, 80, 4, "stone_slab")

        # 1. Place a torch on top of the bottom slab at (4, 80, 4)
        game = MockGame(self.world, "torch")
        hit_slab = self.world.raycast(4.5, 82.0, 4.5, 0, -1, 0, max_dist=5.0)
        self.assertIsNotNone(hit_slab)
        self.assertEqual((hit_slab["x"], hit_slab["y"], hit_slab["z"]), (4, 80, 4))
        self.assertEqual(hit_slab["ny"], 1)
        game.target = hit_slab
        game._place_block()

        # Should place "torch_offset" at (4, 81, 4), resting on y=80.5 and emitting light
        self.assertEqual(self.world.get_block(4, 81, 4), "torch_offset")
        self.assertEqual(self.world.get_blocklight(4, 81, 4), (14, 11, 4, 0, 10))

        # 2. Raycasting horizontally at y=80.75 (inside voxel y=80 above the bottom slab) hits the offset torch at (4, 81, 4)
        hit_torch = self.world.raycast(6.0, 80.75, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_torch)
        self.assertEqual((hit_torch["x"], hit_torch["y"], hit_torch["z"]), (4, 81, 4))
        self.assertAlmostEqual(hit_torch["box"][1], 80.5)

        # 3. Cannot combine the bottom slab at (4, 80, 4) into a double slab while torch_offset is on top
        slab_game = MockGame(self.world, "stone_slab")
        slab_game.target = hit_slab
        slab_game._place_block()
        self.assertEqual(self.world.get_block(4, 80, 4), "stone_slab")

        # 4. Place a top-half slab above the offset torch at (4, 81, 4) by clicking the upper half of adjacent planks at (3, 81, 4)
        self.world.set_block(3, 81, 4, "planks")
        hit_adj_upper = self.world.raycast(6.0, 81.75, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_adj_upper)
        self.assertEqual((hit_adj_upper["x"], hit_adj_upper["y"], hit_adj_upper["z"]), (3, 81, 4))
        self.assertGreaterEqual(hit_adj_upper["local_y"], 0.5)
        slab_game.target = hit_adj_upper
        slab_game._place_block()

        # (4, 81, 4) becomes "torch_offset_stone_slab", keeping torch light and top-slab collision box
        self.assertEqual(self.world.get_block(4, 81, 4), "torch_offset_stone_slab")
        self.assertEqual(self.world.get_blocklight(4, 81, 4), (14, 11, 4, 0, 10))
        self.assertEqual(self.world.get_collision_boxes(4, 81, 4), [(4.0, 81.5, 4.0, 5.0, 82.0, 5.0)])

        # 5. Aiming at the top slab part of "torch_offset_stone_slab" and breaking it leaves "torch_offset"
        hit_top_part = self.world.raycast(6.0, 81.75, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_top_part)
        self.assertEqual((hit_top_part["x"], hit_top_part["y"], hit_top_part["z"]), (4, 81, 4))
        self.assertEqual(hit_top_part["slab_half"], "top")
        slab_game.target = hit_top_part
        slab_game._break_block()
        self.assertEqual(self.world.get_block(4, 81, 4), "torch_offset")

        # 6. Breaking the supporting bottom slab at (4, 80, 4) pops off torch_offset and drops "torch"
        self.world.set_block(4, 80, 4, AIR)
        self.assertEqual(self.world.get_block(4, 81, 4), AIR)
        dropped_items = [e.item for e in self.world.entities]
        self.assertIn("torch", dropped_items)

        # 7. Offset full block at (4, 81, 4) with an offset torch on top at (4, 82, 4) can have an offset full block placed above it at (4, 83, 4)
        self.world.set_block(4, 80, 4, "stone_slab")
        self.world.set_block(4, 81, 4, "planks_offset")
        self.world.set_block(4, 82, 4, "purple_torch_offset")
        self.world.set_block(3, 83, 4, "planks_offset")

        planks_game = MockGame(self.world, "planks")
        hit_side_offset = self.world.raycast(6.0, 83.0, 4.5, -1, 0, 0, max_dist=5.0)
        self.assertIsNotNone(hit_side_offset)
        self.assertEqual((hit_side_offset["x"], hit_side_offset["y"], hit_side_offset["z"]), (3, 83, 4))
        planks_game.target = hit_side_offset
        planks_game._place_block()

        self.assertEqual(self.world.get_block(4, 83, 4), "planks_offset")
        self.assertEqual(self.world.get_block(4, 82, 4), "purple_torch_offset")

    def test_cutout_offset_block_does_not_cull_slab_or_adjacent_offset_faces(self):
        from server.mesh import build_chunk_mesh

        # Place two bottom slabs at (4, 80, 4) and (5, 80, 4)
        self.world.set_block(4, 80, 4, "stone_slab")
        self.world.set_block(5, 80, 4, "stone_slab")

        # Place cutout offset block (leaves_offset) on (4, 80, 4) and opaque offset block (planks_offset) on (5, 80, 4)
        self.world.set_block(4, 81, 4, "leaves_offset")
        self.world.set_block(5, 81, 4, "planks_offset")

        mesh = build_chunk_mesh(self.world, 0, 5, 0)
        self.assertIsNotNone(mesh)
        verts = mesh.vertices.reshape(-1, 9)

        # 1. Top face of stone_slab at (4, 80, 4) (y = 0.5 in chunk cy=5, x in [4, 5], z in [4, 5], normal = (0, 1, 0)) must NOT be culled by leaves_offset
        slab_top_verts = verts[
            (verts[:, 0] >= 4.0)
            & (verts[:, 0] <= 5.0)
            & (verts[:, 1] == 0.5)
            & (verts[:, 2] >= 4.0)
            & (verts[:, 2] <= 5.0)
            & (verts[:, 4] == 1.0)
        ]
        self.assertEqual(len(slab_top_verts), 4)

        # 2. West face (-X, normal = (-1, 0, 0) at x = 5.0) of planks_offset at (5, 81, 4) (top corners at y = 1.5) must NOT be culled by leaves_offset at (4, 81, 4)
        planks_west_top_verts = verts[
            (verts[:, 0] == 5.0)
            & (verts[:, 1] == 1.5)
            & (verts[:, 2] >= 4.0)
            & (verts[:, 2] <= 5.0)
            & (verts[:, 3] == -1.0)
        ]
        self.assertEqual(len(planks_west_top_verts), 2)


if __name__ == "__main__":
    unittest.main()
