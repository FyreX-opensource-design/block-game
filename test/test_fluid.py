import unittest
import math

from server.blocks import AIR, BLOCKS, STONE, WATER
from server.fluid import (
    FLUID_LEVELS,
    WATER_FALLING,
    WATER_LEVEL_BLOCKS,
    WATER_SOURCE,
    FluidSystem,
    compute_new_fluid_state,
    find_dropoff_directions,
    get_fluid_level,
    is_water,
    is_water_source,
)
from server.mesh import build_chunk_mesh
from server.player import Player
from server.world import World


class TestFluidPhysics(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        # Clear chunk (0, 5, 0)
        self.cx, self.cy, self.cz = 0, 5, 0
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

    def test_fluid_levels_and_definitions(self):
        self.assertEqual(get_fluid_level("water"), 8)
        self.assertEqual(get_fluid_level("water_falling"), 8)
        self.assertEqual(get_fluid_level("water_7"), 7)
        self.assertEqual(get_fluid_level("water_1"), 1)
        self.assertEqual(get_fluid_level(AIR), 0)
        self.assertTrue(is_water("water"))
        self.assertTrue(is_water("water_falling"))
        self.assertTrue(is_water("water_3"))
        self.assertFalse(is_water(STONE))
        self.assertTrue(is_water_source("water"))
        self.assertFalse(is_water_source("water_falling"))
        self.assertFalse(is_water_source("water_7"))

    def test_water_falls_downward(self):
        # Create a floor high up in chunk (0, 5, 0): y=80..95
        # Set stone at y=80
        for x in range(3):
            for z in range(3):
                self.world.set_block(x, 80, z, STONE)

        # Place water at (1, 85, 1)
        self.world.set_block(1, 85, 1, WATER)

        # Update fluid system until it hits floor
        for _ in range(10):
            self.world.update_fluids(0.2)

        # At (1, 84, 1), (1, 83, 1), (1, 82, 1), (1, 81, 1), it should be water_falling
        for y in range(81, 85):
            self.assertEqual(
                self.world.get_block(1, y, 1),
                WATER_FALLING,
                f"Expected water_falling at y={y}, got {self.world.get_block(1, y, 1)}",
            )

    def test_water_spreads_horizontally_up_to_8_blocks(self):
        # Build a long stone platform from x=1..12 at y=80, z=5
        for x in range(1, 15):
            self.world.set_block(x, 80, 5, STONE)
            # Walls on both sides of the trench so it only flows in +X direction
            self.world.set_block(x, 81, 4, STONE)
            self.world.set_block(x, 81, 6, STONE)
        # End wall at x=14
        self.world.set_block(14, 81, 5, STONE)
        # Back wall at x=0
        self.world.set_block(0, 81, 5, STONE)

        # Place water source at x=1, y=81, z=5
        self.world.set_block(1, 81, 5, WATER)

        # Tick until flow stabilizes
        for _ in range(30):
            self.world.update_fluids(0.2)

        # Water source is at x=1 (level 8)
        self.assertEqual(self.world.get_block(1, 81, 5), WATER)
        # Flowing levels from x=2 to x=8 should decrease from water_7 down to water_1
        self.assertEqual(self.world.get_block(2, 81, 5), "water_7")
        self.assertEqual(self.world.get_block(3, 81, 5), "water_6")
        self.assertEqual(self.world.get_block(4, 81, 5), "water_5")
        self.assertEqual(self.world.get_block(5, 81, 5), "water_4")
        self.assertEqual(self.world.get_block(6, 81, 5), "water_3")
        self.assertEqual(self.world.get_block(7, 81, 5), "water_2")
        self.assertEqual(self.world.get_block(8, 81, 5), "water_1")
        # Beyond 8 blocks (total distance 8 from source), it must NOT flow further (level 0 = air)
        self.assertEqual(self.world.get_block(9, 81, 5), AIR)
        self.assertEqual(self.world.get_block(10, 81, 5), AIR)

    def test_dropoff_pathfinding(self):
        # Flat surface at y=80 with a hole at (3, 80, 5)
        for x in range(1, 8):
            for z in range(1, 8):
                self.world.set_block(x, 80, z, STONE)

        # Carve a dropoff hole at (3, 80, 5)
        self.world.set_block(3, 80, 5, AIR)

        # Test dropoff directions from (3, 81, 7)
        # Looking toward the hole at z=5 (-Z)
        dirs = find_dropoff_directions(self.world, 3, 81, 6)
        self.assertIn((0, -1), dirs)
        # Only the direction toward the dropoff should be favored
        self.assertEqual(dirs, ((0, -1),))

    def test_infinite_water_source_creation(self):
        # 3-block trench with solid stone below: (1, 80, 2), (2, 80, 2), (3, 80, 2)
        for x in range(1, 4):
            self.world.set_block(x, 80, 2, STONE)
            self.world.set_block(x, 81, 1, STONE)
            self.world.set_block(x, 81, 3, STONE)
        self.world.set_block(0, 81, 2, STONE)
        self.world.set_block(4, 81, 2, STONE)

        # Place sources at ends: (1, 81, 2) and (3, 81, 2)
        self.world.set_block(1, 81, 2, WATER)
        self.world.set_block(3, 81, 2, WATER)

        # Tick fluid system
        for _ in range(5):
            self.world.update_fluids(0.2)

        # The middle block (2, 81, 2) should turn into a full WATER source!
        self.assertEqual(self.world.get_block(2, 81, 2), WATER)

    def test_water_dissipation_when_source_removed(self):
        for x in range(1, 6):
            self.world.set_block(x, 80, 5, STONE)
            self.world.set_block(x, 81, 4, STONE)
            self.world.set_block(x, 81, 6, STONE)
        self.world.set_block(0, 81, 5, STONE)
        self.world.set_block(1, 81, 5, WATER)
        for _ in range(10):
            self.world.update_fluids(0.2)
        self.assertEqual(self.world.get_block(2, 81, 5), "water_7")

        # Remove source
        self.world.set_block(1, 81, 5, AIR)
        for _ in range(15):
            self.world.update_fluids(0.2)

        # All flowing water should have dried up
        for x in range(1, 6):
            self.assertEqual(self.world.get_block(x, 81, 5), AIR)

    def test_unloaded_chunks_do_not_flow(self):
        # Fluid at the edge of loaded chunk cx=0, cy=5, cz=0 (x=15)
        # Chunk (1, 5, 0) is NOT loaded
        self.assertNotIn((1, 5, 0), self.world.chunks)
        self.world.set_block(15, 81, 5, WATER)
        for _ in range(5):
            self.world.update_fluids(0.2)
        # Water cannot have flown into unloaded chunk
        self.assertNotIn((1, 5, 0), self.world.chunks)


class TestPlayerSwimming(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.cx, self.cy, self.cz = 0, 5, 0
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

        # Floor at y=80
        for x in range(4):
            for z in range(4):
                self.world.set_block(x, 80, z, STONE)

        # Pool of water at y=81, 82, 83, 84
        for y in range(81, 85):
            for x in range(4):
                for z in range(4):
                    self.world.set_block(x, y, z, WATER)

        self.player = Player(self.world)
        self.player.pos = [1.5, 82.0, 1.5]

    def test_in_water_detection(self):
        self.assertTrue(self.player.in_water())
        # Move player above water pool (y=86)
        self.player.pos[1] = 86.0
        self.assertFalse(self.player.in_water())

    def test_swim_up_with_jump(self):
        self.player.input.jump = True
        initial_y = self.player.pos[1]
        self.player.update(0.1)
        self.assertGreater(self.player.vel[1], 0.0)
        self.assertGreater(self.player.pos[1], initial_y)

    def test_swim_down_with_down(self):
        self.player.input.down = True
        initial_y = self.player.pos[1]
        self.player.update(0.1)
        self.assertLess(self.player.vel[1], 0.0)
        self.assertLess(self.player.pos[1], initial_y)

    def test_pitch_based_swimming(self):
        # Look down (pitch > 0 in this engine: positive pitch is looking up or down?)
        # Let's check look_direction: y component is sin(pitch)
        self.player.input.forward = True
        self.player.pitch = -0.5  # Looking down
        self.player.update(0.1)
        self.assertLess(self.player.vel[1], 0.0)

        # Look up
        self.player.pitch = 0.5  # Looking up
        self.player.update(0.1)
        self.player.update(0.1)
        self.assertGreater(self.player.vel[1], 0.0)

    def test_cushioned_falling_into_water(self):
        # Player falling with high downward velocity
        self.player.vel[1] = -30.0
        self.player.update(0.05)
        # Water drag slows vertical velocity significantly
        self.assertGreater(self.player.vel[1], -5.0)


class TestFluidMeshing(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.cx, self.cy, self.cz = 0, 5, 0
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

    def test_water_mesh_generation(self):
        # 2x2 water pool on stone floor
        for x in range(2):
            for z in range(2):
                self.world.set_block(x, 80, z, STONE)
                self.world.set_block(x, 81, z, WATER)

        mesh = build_chunk_mesh(self.world, self.cx, self.cy, self.cz)
        self.assertIsNotNone(mesh)
        self.assertIsNotNone(mesh.translucent)
        vertices, indices = mesh.translucent

        # Top faces should be generated
        # Total top faces for 2x2 water pool should be 4 quads = 16 vertices, plus exposed outer sides
        # But interior shared faces between adjacent water blocks must NOT be generated
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)

    def test_flowing_step_faces_generated(self):
        # Trench with water source -> water_7 -> water_6
        self.world.set_block(1, 80, 1, STONE)
        self.world.set_block(2, 80, 1, STONE)
        self.world.set_block(3, 80, 1, STONE)

        self.world.set_block(1, 81, 1, "water")
        self.world.set_block(2, 81, 1, "water_7")
        self.world.set_block(3, 81, 1, "water_6")

        mesh = build_chunk_mesh(self.world, self.cx, self.cy, self.cz)
        self.assertIsNotNone(mesh.translucent)
        verts, _ = mesh.translucent

        step_at_2 = False
        step_at_3 = False
        for i in range(0, len(verts), 4):
            norm = verts[i, 3:6]
            pts = verts[i : i + 4, 0:3]
            # Normal [1, 0, 0] at x=2 connecting water (h=2.0) down to water_7 (h=1.875)
            if (
                abs(norm[0] - 1.0) < 1e-3
                and abs(pts[0, 0] - 2.0) < 1e-3
                and abs(pts[0, 1] - 1.875) < 1e-3
                and abs(pts[2, 1] - 2.0) < 1e-3
            ):
                step_at_2 = True
            # Normal [1, 0, 0] at x=3 connecting water_7 (h=1.875) down to water_6 (h=1.75)
            if (
                abs(norm[0] - 1.0) < 1e-3
                and abs(pts[0, 0] - 3.0) < 1e-3
                and abs(pts[0, 1] - 1.75) < 1e-3
                and abs(pts[2, 1] - 1.875) < 1e-3
            ):
                step_at_3 = True

        self.assertTrue(step_at_2, "Missing vertical step face between water and water_7")
        self.assertTrue(step_at_3, "Missing vertical step face between water_7 and water_6")

    def test_water_texture_translucency(self):
        from server.images import read_png
        from pathlib import Path
        img = read_png(Path("assets/textures/water.png"))
        avg_alpha = float(img[:, :, 3].mean())
        # Alpha should be translucent (between 100 and 200 out of 255)
        self.assertGreater(avg_alpha, 100)
        self.assertLess(avg_alpha, 200)

    def test_bucket_interaction_creative_and_survival(self):
        from server.player import Player
        from server.stacks import make_stack, stack_id
        from client.app import Game

        class MockApp:
            def __init__(self, world, bucket_id, gamemode="creative"):
                self.world = world
                self.player = Player(self.world)
                self.eye = (0.5, 85.5, 0.5)
                self.forward = (0.0, -1.0, 0.0)
                self.selected = 0
                self.hotbar = [make_stack(bucket_id, 1)]
                self.storage = []
                self.gamemode = gamemode
                self.network_client = None

            def _upload_block_change(self, x, y, z):
                pass

        MockApp._handle_bucket_interaction = Game._handle_bucket_interaction

        # Put stone floor at y=80
        self.world.set_block(0, 80, 0, STONE)

        # 1. Creative placement with clay_water_bucket
        app_creative = MockApp(self.world, "clay_water_bucket", "creative")
        self.assertTrue(app_creative._handle_bucket_interaction())
        self.assertEqual(self.world.get_block(0, 81, 0), "water")
        # In creative, bucket is retained
        self.assertEqual(stack_id(app_creative.hotbar[0]), "clay_water_bucket")

        # 2. Survival placement with clay_water_bucket
        self.world.set_block(0, 81, 0, AIR)
        app_survival = MockApp(self.world, "clay_water_bucket", "survival")
        self.assertTrue(app_survival._handle_bucket_interaction())
        self.assertEqual(self.world.get_block(0, 81, 0), "water")
        # In survival, bucket becomes empty clay_bucket
        self.assertEqual(stack_id(app_survival.hotbar[0]), "clay_bucket")

        # 3. Pickup water source block with empty clay_bucket
        self.assertTrue(app_survival._handle_bucket_interaction())
        self.assertEqual(self.world.get_block(0, 81, 0), AIR)
        self.assertEqual(stack_id(app_survival.hotbar[0]), "clay_water_bucket")

        # 4. Trying to pick up flowing water should fail
        self.world.set_block(0, 81, 0, "water_7")
        app_flow = MockApp(self.world, "clay_bucket", "survival")
        self.assertFalse(app_flow._handle_bucket_interaction())
        self.assertEqual(stack_id(app_flow.hotbar[0]), "clay_bucket")

    def test_fluids_break_fluid_breakable_plants_and_drop_loot(self):
        # 1. Block trait checks
        for plant in ("tall_grass", "dandelion", "fern", "rose"):
            self.assertTrue(getattr(BLOCKS[plant], "fluid_breakable", False), f"{plant} should have fluid_breakable=True")
        self.assertFalse(getattr(BLOCKS[STONE], "fluid_breakable", False))

        # 2. Water flowing horizontally into tall grass breaks it and drops tall_grass
        for x in range(1, 5):
            self.world.set_block(x, 80, 2, STONE)
            self.world.set_block(x, 81, 1, STONE)
            self.world.set_block(x, 81, 3, STONE)
        self.world.set_block(1, 81, 2, STONE)
        self.world.set_block(3, 81, 2, "tall_grass")
        self.world.set_block(2, 81, 2, "water")

        initial_count = len(self.world.entities)
        self.world.update_fluids(0.2)

        # Space where tall grass was is now water
        self.assertTrue(is_water(self.world.get_block(3, 81, 2)))
        # A dropped item entity was spawned
        self.assertEqual(len(self.world.entities), initial_count + 1)
        self.assertEqual(self.world.entities[-1].item, "tall_grass")

        # 3. Placing water directly into dandelion breaks it and drops dandelion
        self.world.set_block(5, 80, 5, STONE)
        self.world.set_block(5, 81, 5, "dandelion")
        self.world.set_block(5, 81, 5, "water")
        self.assertEqual(self.world.get_block(5, 81, 5), "water")
        self.assertEqual(self.world.entities[-1].item, "dandelion")

    def test_bucket_replaces_plant_and_drops_loot(self):
        from server.player import Player
        from server.stacks import make_stack, stack_id
        from client.app import Game

        class MockApp:
            def __init__(self, world, bucket_id, gamemode="survival"):
                self.world = world
                self.player = Player(self.world)
                self.eye = (0.5, 83.5, 0.5)
                self.forward = (0.0, -1.0, 0.0)
                self.selected = 0
                self.hotbar = [make_stack(bucket_id, 1)]
                self.storage = []
                self.gamemode = gamemode
                self.network_client = None

            def _upload_block_change(self, x, y, z):
                pass

        MockApp._handle_bucket_interaction = Game._handle_bucket_interaction

        # Soil at y=80, tall_grass at y=81
        self.world.set_block(0, 80, 0, "grass")
        self.world.set_block(0, 81, 0, "tall_grass")
        self.assertEqual(self.world.get_block(0, 81, 0), "tall_grass")
        initial_entities = len(self.world.entities)

        # Aim at tall_grass with water_bucket
        app = MockApp(self.world, "water_bucket", "survival")
        self.assertTrue(app._handle_bucket_interaction())

        # Water must replace tall_grass at y=81 (NOT be placed at y=82 on top of it)
        self.assertEqual(self.world.get_block(0, 81, 0), "water")
        self.assertEqual(self.world.get_block(0, 82, 0), AIR)

        # Plant loot must have dropped
        self.assertEqual(len(self.world.entities), initial_entities + 1)
        self.assertEqual(self.world.entities[-1].item, "tall_grass")

        # In survival, bucket becomes empty iron_bucket
        self.assertEqual(stack_id(app.hotbar[0]), "iron_bucket")

        # Test with flower (rose) using clay_water_bucket
        self.world.set_block(2, 80, 2, "grass")
        self.world.set_block(2, 81, 2, "rose")
        app_rose = MockApp(self.world, "clay_water_bucket", "survival")
        app_rose.eye = (2.5, 83.5, 2.5)

        self.assertTrue(app_rose._handle_bucket_interaction())
        self.assertEqual(self.world.get_block(2, 81, 2), "water")
        self.assertEqual(self.world.get_block(2, 82, 2), AIR)
        self.assertEqual(self.world.entities[-1].item, "rose")
        self.assertEqual(stack_id(app_rose.hotbar[0]), "clay_bucket")

    def test_water_flowing_into_ocean_does_not_spread_over_surface(self):
        # Stone cliff / shore at y=80: x=0..2, z=5
        for x in range(0, 3):
            self.world.set_block(x, 80, 5, STONE)
            # Walls so water on land flows towards the ocean at +X
            self.world.set_block(x, 81, 4, STONE)
            self.world.set_block(x, 81, 6, STONE)
        self.world.set_block(0, 81, 5, STONE)  # Back wall

        # Ocean water source at y=80: x=3..12, z=3..7
        for ox in range(3, 13):
            for oz in range(3, 8):
                self.world.set_block(ox, 80, oz, WATER)

        # Place water on shore at x=1, y=81, z=5
        self.world.set_block(1, 81, 5, WATER)

        # Update fluids until stable
        for _ in range(25):
            self.world.update_fluids(0.2)

        # Water source on shore
        self.assertEqual(self.world.get_block(1, 81, 5), WATER)
        # Flowing water at edge of shore (solid below)
        self.assertEqual(self.world.get_block(2, 81, 5), "water_7")

        # Water falls into ocean at (3, 81, 5)
        # But beyond the cliff dropoff point, it must NOT spread horizontally across the ocean surface
        for ox in range(4, 13):
            self.assertEqual(
                self.world.get_block(ox, 81, 5),
                AIR,
                f"Ocean surface at ({ox}, 81, 5) should be AIR, got {self.world.get_block(ox, 81, 5)}",
            )
        # Also must not spread horizontally in +Z or -Z on top of ocean
        self.assertEqual(self.world.get_block(3, 81, 4), AIR)
        self.assertEqual(self.world.get_block(3, 81, 6), AIR)

    def test_waterfall_into_ocean_does_not_spread_horizontally(self):
        # Ocean water at y=80 from x=0..6, z=0..6
        for x in range(0, 7):
            for z in range(0, 7):
                self.world.set_block(x, 80, z, WATER)

        # High waterfall column at (3, y, 3) starting from y=85
        self.world.set_block(3, 85, 3, WATER)

        for _ in range(15):
            self.world.update_fluids(0.2)

        # Column should fall down
        for y in range(81, 85):
            self.assertEqual(self.world.get_block(3, y, 3), WATER_FALLING)

        # Ocean at y=80 should remain WATER source
        self.assertEqual(self.world.get_block(3, 80, 3), WATER)

        # Sideways blocks at y=81 directly above ocean surface must NOT have fluid spread
        self.assertEqual(self.world.get_block(2, 81, 3), AIR)
        self.assertEqual(self.world.get_block(4, 81, 3), AIR)
        self.assertEqual(self.world.get_block(3, 81, 2), AIR)
        self.assertEqual(self.world.get_block(3, 81, 4), AIR)

    def test_take_dirty_distance_prioritization(self):
        self.world.dirty.clear()
        self.world.dirty.update([(10, 0, 10), (0, 0, 0), (5, 0, 5), (1, 0, 1)])
        # Near player at (0, 0, 0)
        taken = self.world.take_dirty(limit=2, player_pos=(0.0, 0.0, 0.0))
        self.assertEqual(len(taken), 2)
        self.assertEqual(taken[0], (0, 0, 0))
        self.assertEqual(taken[1], (1, 0, 1))


class TestFluidFlowPushing(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=42)
        self.cx, self.cy, self.cz = 0, 5, 0
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

        # Build a stone canal along X at y=80, z=5
        for x in range(1, 15):
            self.world.set_block(x, 80, 5, STONE)
            self.world.set_block(x, 81, 4, STONE)
            self.world.set_block(x, 81, 6, STONE)
        self.world.set_block(0, 81, 5, STONE)
        self.world.set_block(14, 81, 5, STONE)

        # Water source at x=1, y=81, z=5 flowing in +X direction
        self.world.set_block(1, 81, 5, WATER)
        for _ in range(25):
            self.world.update_fluids(0.2)

    def test_fluid_flow_vectors(self):
        from server.fluid import get_entity_fluid_flow, get_fluid_flow

        # In canal, water flows towards +X
        fx, fy, fz = get_fluid_flow(self.world, 3, 81, 5)
        self.assertAlmostEqual(fx, 1.0, places=2)
        self.assertAlmostEqual(fy, 0.0, places=2)
        self.assertAlmostEqual(fz, 0.0, places=2)

        # Entity bounding box inside stream
        efx, efy, efz = get_entity_fluid_flow(self.world, 3, 81, 5, 3, 81, 5)
        self.assertAlmostEqual(efx, 1.0, places=2)

    def test_player_pushed_by_stream(self):
        player = Player(self.world)
        player.pos = [3.5, 81.0, 5.5]
        # Player has no movement input
        player.input.forward = False
        player.input.back = False
        player.input.left = False
        player.input.right = False

        initial_x = player.pos[0]
        for _ in range(5):
            player.update(0.05)

        self.assertGreater(player.vel[0], 0.5)
        self.assertGreater(player.pos[0], initial_x)

    def test_item_entity_pushed_by_stream(self):
        from server.entity import ItemEntity

        item = ItemEntity("dirt", pos=[3.5, 81.0, 5.5], vel=[0.0, 0.0, 0.0])
        self.world.entities.append(item)

        initial_x = item.pos[0]
        for _ in range(10):
            item.update(0.05, self.world)

        self.assertGreater(item.vel[0], 0.2)
        self.assertGreater(item.pos[0], initial_x)

    def test_item_buoyancy_in_water(self):
        from server.entity import ItemEntity

        # Submerged item in deep water floats upward
        for y in range(81, 85):
            self.world.set_block(8, y, 8, WATER)
        self.world.set_block(8, 80, 8, STONE)

        item = ItemEntity("diamond", pos=[8.5, 81.2, 8.5], vel=[0.0, 0.0, 0.0])
        self.world.entities.append(item)

        initial_y = item.pos[1]
        for _ in range(10):
            item.update(0.05, self.world)

        self.assertGreater(item.vel[1], 0.0)
        self.assertGreater(item.pos[1], initial_y)

    def test_mob_pushed_by_stream(self):
        from server.entity import Pig

        pig = Pig(pos=[3.5, 81.0, 5.5], vel=[0.0, 0.0, 0.0])
        pig.desired_speed = 0.0
        self.world.entities.append(pig)

        initial_x = pig.pos[0]
        for _ in range(10):
            pig.update(0.05, self.world)

        self.assertGreater(pig.vel[0], 0.2)
        self.assertGreater(pig.pos[0], initial_x)

    def test_waterfall_pulls_entities_down(self):
        from server.entity import ItemEntity, Pig

        # Setup vertical waterfall at (10, y, 10)
        for y in range(81, 87):
            self.world.set_block(10, y, 10, WATER_FALLING)
        self.world.set_block(10, 80, 10, STONE)

        # Player in waterfall
        player = Player(self.world)
        player.pos = [10.5, 84.0, 10.5]
        player.vel = [0.0, 0.0, 0.0]
        player.update(0.05)
        self.assertLess(player.vel[1], -0.2)

        # Item in waterfall
        item = ItemEntity("dirt", pos=[10.5, 84.0, 10.5], vel=[0.0, 0.0, 0.0])
        item.update(0.05, self.world)
        self.assertLess(item.vel[1], -0.2)

        # Mob in waterfall
        pig = Pig(pos=[10.5, 84.0, 10.5], vel=[0.0, 0.0, 0.0])
        pig.desired_speed = 0.0
        pig.update(0.05, self.world)
        self.assertLess(pig.vel[1], -0.2)


class TestFluidDefinitionsAndInteractions(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.cx, self.cy, self.cz = 0, 5, 0
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

    def test_fluid_asset_definitions_loaded(self):
        from server.fluid import FLUIDS, FluidDefinition

        self.assertIn("water", FLUIDS)
        self.assertIn("salt_water", FLUIDS)
        self.assertIn("brackish_water", FLUIDS)
        for fid in ("water", "salt_water", "brackish_water"):
            fdef = FLUIDS[fid]
            self.assertIsInstance(fdef, FluidDefinition)
            self.assertEqual(fdef.flow_distance, 7)
            self.assertEqual(fdef.speed, 1.0)

    def test_custom_flow_distance_and_speed(self):
        from server.fluid import FLUIDS, FluidDefinition

        orig = FLUIDS["salt_water"]
        try:
            # Set salt_water flow_distance=3 and speed=0.5 (tick_interval = 0.30s)
            FLUIDS["salt_water"] = FluidDefinition("salt_water", block="salt_water", flow_distance=3, speed=0.5)
            for x in range(1, 10):
                self.world.set_block(x, 80, 5, STONE)
                self.world.set_block(x, 81, 4, STONE)
                self.world.set_block(x, 81, 6, STONE)
            self.world.set_block(0, 81, 5, STONE)
            self.world.set_block(9, 81, 5, STONE)

            self.world.set_block(1, 81, 5, "salt_water")
            # At dt=0.16 (< 0.30 interval for speed=0.5), salt_water has not ticked yet
            self.world.update_fluids(0.16)
            self.assertEqual(self.world.get_block(2, 81, 5), AIR)

            # Another dt=0.16 reaches 0.32 >= 0.30, so salt_water spreads 1 step to level 3
            self.world.update_fluids(0.16)
            self.assertEqual(self.world.get_block(2, 81, 5), "salt_water_3")

            # Tick further until stabilized
            for _ in range(15):
                self.world.update_fluids(0.35)

            self.assertEqual(self.world.get_block(1, 81, 5), "salt_water")
            self.assertEqual(self.world.get_block(2, 81, 5), "salt_water_3")
            self.assertEqual(self.world.get_block(3, 81, 5), "salt_water_2")
            self.assertEqual(self.world.get_block(4, 81, 5), "salt_water_1")
            self.assertEqual(self.world.get_block(5, 81, 5), AIR)
        finally:
            FLUIDS["salt_water"] = orig

    def test_both_flowing_water_and_salt_water_do_not_create_brackish_source(self):
        # Build a trench from x=1 to x=7 at y=81, z=5
        for x in range(1, 8):
            self.world.set_block(x, 80, 5, STONE)
            self.world.set_block(x, 81, 4, STONE)
            self.world.set_block(x, 81, 6, STONE)
        self.world.set_block(0, 81, 5, STONE)
        self.world.set_block(8, 81, 5, STONE)

        # Fresh water at x=1, salt water at x=7: their flowing streams meet in the middle (x=4)
        self.world.set_block(1, 81, 5, "water")
        self.world.set_block(7, 81, 5, "salt_water")

        for _ in range(15):
            self.world.update_fluids(0.2)

        trench_blocks = [self.world.get_block(x, 81, 5) for x in range(1, 8)]
        # Neither stream reached the other's source block, so NO brackish_water source is created
        self.assertNotIn("brackish_water", trench_blocks)
        self.assertEqual(self.world.get_block(1, 81, 5), "water")
        self.assertEqual(self.world.get_block(7, 81, 5), "salt_water")

    def test_flowing_fluids_meet_with_custom_flow_result_recipe(self):
        from server.recipes import FLUID_MEET, RECIPES, Recipe
        import server.recipes as recipes_mod

        custom_recipe = Recipe(
            "custom_flow_meet",
            False,
            (),
            (),
            "stone",
            1,
            FLUID_MEET,
            fluids=("water", "brackish_water"),
            source_result="dirt",
        )
        orig_recipes = recipes_mod.RECIPES
        try:
            recipes_mod.RECIPES = orig_recipes + (custom_recipe,)
            for x in range(1, 6):
                self.world.set_block(x, 80, 5, STONE)
                self.world.set_block(x, 81, 4, STONE)
                self.world.set_block(x, 81, 6, STONE)
            self.world.set_block(0, 81, 5, STONE)
            self.world.set_block(6, 81, 5, STONE)

            # Place water at x=1 and brackish_water at x=5 so flowing streams meet at x=3
            self.world.set_block(1, 81, 5, "water")
            self.world.set_block(5, 81, 5, "brackish_water")
            for _ in range(5):
                self.world.update_fluids(0.2)

            trench_blocks = [self.world.get_block(x, 81, 5) for x in range(1, 6)]
            self.assertIn("stone", trench_blocks)
        finally:
            recipes_mod.RECIPES = orig_recipes

    def test_flowing_fluid_hitting_source_converts_source_block(self):
        # Build a short trench from x=1 to x=3 at y=81, z=5
        for x in range(1, 4):
            self.world.set_block(x, 80, 5, STONE)
            self.world.set_block(x, 81, 4, STONE)
            self.world.set_block(x, 81, 6, STONE)
        self.world.set_block(0, 81, 5, STONE)
        self.world.set_block(4, 81, 5, STONE)

        # Fresh water at x=1 flows into x=2, and salt_water source is at x=3
        self.world.set_block(1, 81, 5, "water")
        self.world.update_fluids(0.2)
        self.assertEqual(self.world.get_block(2, 81, 5), "water_7")

        # Now place salt_water source at x=3 next to flowing water_7 at x=2
        self.world.set_block(3, 81, 5, "salt_water")
        self.world.update_fluids(0.2)

        # The salt_water source block at x=3 turns into brackish_water!
        self.assertEqual(self.world.get_block(3, 81, 5), "brackish_water")


if __name__ == "__main__":
    unittest.main()
