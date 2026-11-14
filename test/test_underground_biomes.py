import unittest
import numpy as np

from server.blocks import BLOCKS, block_from_index, block_index
from server.biomes import BIOMES, biome_at
from server.underground_biomes import (
    UndergroundBiome,
    load_underground_biomes,
    underground_biome_at,
    default_stone_strata,
)
from server.world import World, CHUNK_SIZE


class TestUndergroundBiomes(unittest.TestCase):
    def test_load_underground_biomes(self):
        biomes = load_underground_biomes()
        self.assertGreaterEqual(len(biomes), 2)
        biome_ids = {b.id for b in biomes}
        self.assertIn("limestone_caverns", biome_ids)
        self.assertIn("crystal_caverns", biome_ids)

        limestone = next(b for b in biomes if b.id == "limestone_caverns")
        self.assertEqual(limestone.name, "Limestone Caverns")
        self.assertEqual(limestone.default_stone, "limestone")
        self.assertTrue(limestone.enabled)
        self.assertGreater(len(limestone.features), 0)
        self.assertIn("limestone_iron_ore", limestone.ores)

        crystal = next(b for b in biomes if b.id == "crystal_caverns")
        self.assertEqual(crystal.name, "Crystal Caverns")
        self.assertEqual(crystal.default_stone, "marble")
        self.assertTrue(crystal.enabled)
        self.assertGreater(len(crystal.features), 0)
        self.assertIn("marble_diamond_ore", crystal.ores)

    def test_underground_biome_resolution(self):
        biomes = load_underground_biomes()
        # Out of bounds check: above max_y or below min_y returns None
        res_high = underground_biome_at(100, 120, 100, 42, biomes)
        self.assertIsNone(res_high)

        # Sampling within valid ranges should find active regions across the world
        found_limestone = False
        found_crystal = False
        for x in range(0, 1000, 40):
            for z in range(0, 1000, 40):
                ub_mid = underground_biome_at(x, 25, z, 42, biomes)
                if ub_mid and ub_mid.id == "limestone_caverns":
                    found_limestone = True
                ub_deep = underground_biome_at(x, 15, z, 42, biomes)
                if ub_deep and ub_deep.id == "crystal_caverns":
                    found_crystal = True
                if found_limestone and found_crystal:
                    break
            if found_limestone and found_crystal:
                break
        self.assertTrue(found_limestone, "Limestone caverns should be found in world generation")
        self.assertTrue(found_crystal, "Crystal caverns should be found in world generation")

    def test_biome_at_3d_integration(self):
        biomes = BIOMES
        ubs = load_underground_biomes()
        # Surface query (y=None) returns 2D surface biome
        surface_b = biome_at(0, 0, 42, biomes)
        self.assertIsNotNone(surface_b)
        self.assertFalse(hasattr(surface_b, "cave_fill"))

        # Surface query with high y (above surface) returns surface biome
        high_b = biome_at(0, 0, 42, biomes, y=90, underground_biomes=ubs)
        self.assertIsNotNone(high_b)

        # Deep query with active underground biome returns UndergroundBiome
        # Find coordinates where underground biome is active
        found_3d = False
        for x in range(0, 600, 30):
            for z in range(0, 600, 30):
                res = biome_at(x, z, 42, biomes, y=20, underground_biomes=ubs)
                if isinstance(res, UndergroundBiome):
                    found_3d = True
                    self.assertIn(res.id, ("limestone_caverns", "crystal_caverns"))
                    break
            if found_3d:
                break
        self.assertTrue(found_3d, "3D biome query should return an UndergroundBiome instance")

    def test_default_stone_strata(self):
        # High elevation should yield andesite
        st_high = default_stone_strata(50, seed=123)
        self.assertIn(st_high, ("andesite", "stone"))

        # Deep elevation should yield marble
        st_low = default_stone_strata(5, seed=123)
        self.assertIn(st_low, ("marble", "stone"))

    def test_cave_features_and_strata_in_chunk(self):
        # Generate underground chunks and verify geological diversity
        world = World(seed=12345)
        stone_types_found = set()
        features_found = set()

        for cx in range(-2, 3):
            for cz in range(-2, 3):
                for cy in (0, 1): # underground chunks (y=0..31)
                    chunk = world.load_chunk(cx, cy, cz)
                    present_ids = np.unique(chunk.data)
                    for blk_id in present_ids:
                        b = block_from_index(blk_id)
                        if not b:
                            continue
                        if b.id in ("granite", "andesite", "diorite", "limestone", "marble"):
                            stone_types_found.add(b.id)
                        if b.id in (
                            "limestone_stalactite",
                            "limestone_stalagmite",
                            "amethyst_crystal",
                            "ruby_crystal",
                            "sapphire_crystal",
                            "emerald_crystal",
                            "diamond_crystal",
                        ):
                            features_found.add(b.id)

        # Verify geological strata stones were generated in underground chunks
        self.assertGreaterEqual(len(stone_types_found), 3, f"Expected varied rock strata, found {stone_types_found}")
        # Verify underground cave features were generated
        self.assertGreaterEqual(len(features_found), 1, f"Expected cave features to generate, found {features_found}")

    def test_block_properties_of_new_features(self):
        # Verify stalactites and stalagmites
        stalactite = BLOCKS["limestone_stalactite"]
        self.assertTrue(stalactite.solid)
        self.assertEqual(stalactite.cutout, True)
        self.assertEqual(stalactite.tool, "pickaxe")

        stalagmite = BLOCKS["limestone_stalagmite"]
        self.assertTrue(stalagmite.solid)
        self.assertEqual(stalagmite.cutout, True)
        self.assertEqual(stalagmite.tool, "pickaxe")

        # Verify crystal cluster blocks have light emission and proper cutout
        for gem in ("amethyst", "ruby", "sapphire", "emerald", "diamond"):
            c_name = f"{gem}_crystal"
            crystal = BLOCKS[c_name]
            self.assertFalse(crystal.solid)
            self.assertEqual(crystal.cutout, True)
            self.assertIsNotNone(crystal.emission, f"{c_name} should emit light")
            self.assertGreaterEqual(len(crystal.emission), 3)

    def test_underground_biome_cave_generation(self):
        # Biome defining custom cave_generation carves caves even if world.caves is empty
        custom_ub = UndergroundBiome(
            biome_id="test_cave_biome",
            name="Test Cave Biome",
            min_y=10,
            max_y=30,
            threshold=-1.0,  # Always active everywhere in this y range
            default_stone="limestone",
            cave_generation={
                "style": "cavern",
                "threshold": -1.0,  # Carves wide open caves
                "scale": 0.05,
            },
        )
        world = World(seed=999, caves=[], underground_biomes=[custom_ub])
        chunk = world.load_chunk(0, 1, 0) # y: 16..31
        # Check that blocks between y=16..26 inside the custom biome were carved to AIR
        air_count = np.sum(chunk.data == 0)
        self.assertGreater(air_count, 100, "Cave generation in underground biome should carve out caves")

    def test_underground_biome_stone_types_and_layers(self):
        # Custom biome with stone layers
        layered_ub = UndergroundBiome(
            biome_id="test_layered_biome",
            min_y=16,
            max_y=31,
            threshold=-1.0,
            default_stone="limestone",
            stone_layers=[
                {"min_y": 16, "max_y": 23, "stone": "granite"},
                {"min_y": 24, "max_y": 31, "stone": "marble"},
            ],
        )
        world = World(seed=123, caves=[], underground_biomes=[layered_ub])
        chunk = world.load_chunk(0, 1, 0)
        granite_idx = block_index("granite")
        marble_idx = block_index("marble")
        # Bottom half of chunk (y=16..23) should have granite
        self.assertIn(granite_idx, chunk.data[0:8])
        # Top half of chunk (y=24..31) should have marble
        self.assertIn(marble_idx, chunk.data[8:16])

    def test_underground_biome_ores(self):
        # Custom biome defining specific ores
        ore_ub = UndergroundBiome(
            biome_id="test_ore_biome",
            min_y=16,
            max_y=31,
            threshold=-1.0,
            default_stone="limestone",
            ores=[
                {
                    "block": "limestone_iron_ore",
                    "target": "limestone",
                    "scale": 0.1,
                    "threshold": -0.5, # High frequency for testing
                }
            ],
        )
        world = World(seed=456, caves=[], underground_biomes=[ore_ub])
        chunk = world.load_chunk(0, 1, 0)
        iron_idx = block_index("limestone_iron_ore")
        self.assertIn(iron_idx, chunk.data)

    def test_underground_biome_mob_spawns(self):
        from server.spawner import MobSpawner
        class DummyPlayer:
            def __init__(self, pos):
                self.pos = list(pos)

        # 1. Biome with empty mob_spawns disables mob spawning
        peaceful_ub = UndergroundBiome(
            biome_id="peaceful_cave",
            min_y=10,
            max_y=30,
            threshold=-1.0,
            default_stone="stone",
            mob_spawns=[],
        )
        world = World(seed=789, caves=[], underground_biomes=[peaceful_ub])
        world.load_chunk(0, 1, 0)
        world.set_block(5, 21, 5, "air")
        world.set_block(5, 22, 5, "air")
        spawner = MobSpawner(world)
        p = DummyPlayer([5, 21, 5])
        spawner.try_spawn([p])
        self.assertEqual(len(world.entities), 0, "Peaceful biome with empty mob_spawns should not spawn mobs")

        # 2. Biome with custom mob spawn definition
        active_ub = UndergroundBiome(
            biome_id="zombie_cavern",
            min_y=10,
            max_y=30,
            threshold=-1.0,
            default_stone="stone",
            mob_spawns=[{"mob": "zombie", "weight": 100, "max_light": 15}],
        )
        active_world = World(seed=789, caves=[], underground_biomes=[active_ub])
        active_world.load_chunk(0, 1, 0)
        active_world.set_block(5, 21, 5, "air")
        active_world.set_block(5, 22, 5, "air")
        active_spawner = MobSpawner(active_world)
        self.assertTrue(active_spawner.can_spawn_mob_at(5, 20, 5, mob_type="zombie", max_light=15))

    def test_world_get_biome(self):
        world = World(seed=42)
        # 2D surface lookup
        surface_b = world.get_biome(100, 100)
        self.assertIsNotNone(surface_b)
        # 3D lookup
        b_high = world.get_biome(100, 100, 100)
        self.assertEqual(b_high.id, surface_b.id)

    def test_dripstone_connections(self):
        from server.blocks import get_dripstone_elements
        from server.mesh import build_chunk_mesh

        world = World(seed=123)
        world.load_chunk(0, 1, 0)
        world.chunks[(0, 1, 0)].data.fill(0)

        # 1. 2-block stalactite (hanging from ceiling at y=23)
        world.set_block(5, 23, 5, "stone")
        world.set_block(5, 22, 5, "limestone_stalactite")
        world.set_block(5, 21, 5, "limestone_stalactite")

        el_base = get_dripstone_elements(world, 5, 22, 5, BLOCKS["limestone_stalactite"])
        el_tip = get_dripstone_elements(world, 5, 21, 5, BLOCKS["limestone_stalactite"])
        uv_base = el_base[0]["faces"][6]
        uv_tip = el_tip[0]["faces"][6]
        self.assertEqual(uv_base, BLOCKS["limestone_stalactite"].dripstone_variants["base"][0]["faces"][6])
        self.assertEqual(uv_tip, BLOCKS["limestone_stalactite"].dripstone_variants["tip"][0]["faces"][6])
        self.assertNotEqual(uv_base, uv_tip)

        # 2. 3-block stalactite (y=20 added below)
        world.set_block(5, 20, 5, "limestone_stalactite")
        el_mid = get_dripstone_elements(world, 5, 21, 5, BLOCKS["limestone_stalactite"])
        el_tip3 = get_dripstone_elements(world, 5, 20, 5, BLOCKS["limestone_stalactite"])
        uv_mid = el_mid[0]["faces"][6]
        uv_tip3 = el_tip3[0]["faces"][6]
        self.assertEqual(uv_mid, BLOCKS["limestone_stalactite"].dripstone_variants["middle"][0]["faces"][6])
        self.assertEqual(uv_tip3, BLOCKS["limestone_stalactite"].dripstone_variants["tip"][0]["faces"][6])

        # 3. 2-block stalagmite (growing from floor at y=16)
        world.set_block(8, 16, 8, "stone")
        world.set_block(8, 17, 8, "limestone_stalagmite")
        world.set_block(8, 18, 8, "limestone_stalagmite")

        sm_base = get_dripstone_elements(world, 8, 17, 8, BLOCKS["limestone_stalagmite"])
        sm_tip = get_dripstone_elements(world, 8, 18, 8, BLOCKS["limestone_stalagmite"])
        self.assertEqual(sm_base[0]["faces"][6], BLOCKS["limestone_stalagmite"].dripstone_variants["base"][0]["faces"][6])
        self.assertEqual(sm_tip[0]["faces"][6], BLOCKS["limestone_stalagmite"].dripstone_variants["tip"][0]["faces"][6])

        # 4. Verify chunk mesh builds without error
        mesh = build_chunk_mesh(world, 0, 1, 0)
        self.assertIsNotNone(mesh)
        self.assertIsNotNone(mesh.cutout)
        self.assertGreater(len(mesh.cutout[1]), 0)

    def test_dripstone_collision_boxes(self):
        from server.blocks import dripstone_collision_boxes

        world = World(seed=123)
        world.load_chunk(0, 0, 0)
        world.chunks[(0, 0, 0)].data.fill(0)

        # Single standalone stalagmite
        world.set_block(2, 5, 2, "limestone_stalagmite")
        boxes = dripstone_collision_boxes(world, 2, 5, 2)
        self.assertEqual(len(boxes), 1)
        bx0, by0, bz0, bx1, by1, bz1 = boxes[0]
        self.assertAlmostEqual(bx0, 2.0 + 4.0 / 16.0)
        self.assertAlmostEqual(by0, 5.0)
        self.assertAlmostEqual(by1, 5.0 + 12.0 / 16.0)

        # 2-block stalagmite: base at y=5, tip at y=6
        world.set_block(2, 6, 2, "limestone_stalagmite")
        b_base = dripstone_collision_boxes(world, 2, 5, 2)
        b_tip = dripstone_collision_boxes(world, 2, 6, 2)
        self.assertAlmostEqual(b_base[0][3], 2.0 + 14.0 / 16.0)
        self.assertAlmostEqual(b_base[0][4], 6.0)
        self.assertAlmostEqual(b_tip[0][4], 6.0 + 11.0 / 16.0)

        # Single standalone stalactite hanging at y=8
        world.set_block(4, 8, 4, "limestone_stalactite")
        st_box = dripstone_collision_boxes(world, 4, 8, 4)
        self.assertAlmostEqual(st_box[0][1], 8.0 + 4.0 / 16.0)
        self.assertAlmostEqual(st_box[0][4], 9.0)

    def test_player_cannot_walk_through_dripstone(self):
        from server.player import Player

        world = World(seed=123)
        world.load_chunk(0, 0, 0)
        world.chunks[(0, 0, 0)].data.fill(0)
        for x in range(2, 8):
            for z in range(2, 8):
                world.set_block(x, 4, z, "stone")

        # Stalagmite at (5, 5, 5)
        world.set_block(5, 5, 5, "limestone_stalagmite")
        p = Player(world)
        p.pos = [4.0, 5.0, 5.5]
        p.on_ground = True
        collided = p.sweep(0, 1.5)
        self.assertTrue(collided)
        self.assertLess(p.pos[0], 5.25)

        # Stalactite at head height (y=6)
        world.set_block(5, 7, 5, "stone")
        world.set_block(5, 6, 5, "limestone_stalactite")
        p.pos = [4.0, 5.0, 5.5]
        collided_head = p.sweep(0, 1.5)
        self.assertTrue(collided_head)

    def test_player_landing_on_stalagmite_damage(self):
        from server.player import Player

        world = World(seed=123)
        world.load_chunk(0, 0, 0)
        world.chunks[(0, 0, 0)].data.fill(0)
        world.set_block(5, 4, 5, "stone")
        world.set_block(5, 5, 5, "limestone_stalagmite")

        p = Player(world)
        # Drop 1.5 blocks onto stalagmite
        p.pos = [5.5, 7.25, 5.5]
        p.update_health(0.05, grounded=False)
        p.pos = [5.5, 5.75, 5.5]
        events = p.update_health(0.05, grounded=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].damage_type.id, "stalagmite")
        self.assertEqual(events[0].amount, 2.0)
        self.assertEqual(events[0].message, "was impaled on a stalagmite")

    def test_ai_avoids_stalagmite_hazard(self):
        from server.ai import is_position_hazardous

        world = World(seed=123)
        world.load_chunk(0, 0, 0)
        world.chunks[(0, 0, 0)].data.fill(0)
        world.set_block(5, 4, 5, "stone")
        world.set_block(5, 5, 5, "limestone_stalagmite")

        # Stepping into cell with stalagmite is hazardous
        self.assertTrue(is_position_hazardous(world, 5.5, 5.0, 5.5))
        # Dropping onto stalagmite from above is hazardous
        self.assertTrue(is_position_hazardous(world, 5.5, 7.0, 5.5))

    def test_dripstone_lighting_opacity(self):
        from server.lighting import get_block_opacity, _get_luts
        from server.mesh import build_chunk_mesh

        # Verify stalactites and stalagmites have 0 light opacity
        for b_name in (
            "limestone_stalactite",
            "limestone_stalagmite",
            "limestone_stalactite_tip",
            "limestone_stalagmite_tip",
        ):
            b = BLOCKS[b_name]
            self.assertEqual(get_block_opacity(b), 0)

        op_lut, _ = _get_luts()
        st_idx = block_index("limestone_stalactite")
        sm_idx = block_index("limestone_stalagmite")
        self.assertEqual(op_lut[st_idx], 0)
        self.assertEqual(op_lut[sm_idx], 0)

        # In a lit room, stalagmites receive light and do not extinguish it
        world = World(seed=123)
        world.load_chunk(0, 0, 0)
        world.chunks[(0, 0, 0)].data.fill(0)
        for x in range(16):
            for z in range(16):
                world.set_block(x, 0, z, "stone")
                world.set_block(x, 8, z, "stone")

        world.lighting.set_blocklight(2, 2, 2, 14, 11, 4, 0, 10)
        world.set_block(5, 1, 5, "limestone_stalagmite")
        world.set_block(5, 7, 5, "limestone_stalactite")

        world.lighting.compute_chunk_lighting(0, 0, 0)
        ch = world.chunks[(0, 0, 0)]
        # Stalagmite and stalactite must have non-zero blocklight from nearby torch
        self.assertGreater(ch.blocklight[1, 5, 5, 0], 0)
        self.assertGreater(ch.blocklight[7, 5, 5, 0], 0)

        # Mesh cutout lights array should have non-zero light values
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh.cutout_lights)
        self.assertGreater(mesh.cutout_lights.max(), 0.0)



if __name__ == "__main__":
    unittest.main()
