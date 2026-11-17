import unittest

import numpy as np

from server.blocks import BLOCKS, CREATIVE, block_index
from server.features import ConfiguredFeature
from server.fluid import FLUIDS, get_bucket_fluid, get_filled_bucket, is_lava, is_water, submerged_fog
from server.mesh import build_chunk_mesh
from server.fire import FIRE_SIMULATION, FIRE_TYPES, body_flame, burn_seconds, fire_type, flame_kind, is_flammable
from server.tags import has_block_tag
from server.player import Player
from server.world import World


class LavaTests(unittest.TestCase):
    def test_lava_flows_less_than_water_and_milk(self):
        self.assertLess(FLUIDS["lava"].flow_distance, FLUIDS["milk"].flow_distance)
        self.assertLess(FLUIDS["milk"].flow_distance, FLUIDS["water"].flow_distance)
        self.assertEqual(FLUIDS["blue_lava"].flow_distance, FLUIDS["lava"].flow_distance)
        self.assertFalse(FLUIDS["lava"].infinite_source)
        self.assertTrue(is_lava("lava"))
        self.assertTrue(is_lava("lava_falling"))
        self.assertTrue(is_lava("blue_lava_3"))
        self.assertTrue(is_lava("black_lava"))
        self.assertFalse(is_lava("water"))
        self.assertFalse(is_water("lava"))
        self.assertTrue(is_water("water"))

    def test_fire_catches_a_plant_and_then_goes_out(self):
        world = World(seed=1, caves=[])
        world.load_chunk(0, 0, 0)
        world.set_block(0, 11, 0, "lava")
        world.set_block(1, 11, 0, "tall_grass")
        self.assertTrue(is_flammable("tall_grass"))
        world.update_fire(0.5)
        self.assertEqual(world.get_block(1, 11, 0), "fire")
        for _ in range(30):
            world.update_fire(0.5)
        self.assertEqual(world.get_block(1, 11, 0), "air")
        self.assertEqual(world.fire.flames, {})

    def test_blue_lava_lights_blue_fire(self):
        world = World(seed=2, caves=[])
        world.load_chunk(0, 0, 0)
        world.set_block(4, 11, 4, "blue_lava")
        world.set_block(5, 11, 4, "fern")
        world.update_fire(0.5)
        self.assertEqual(world.get_block(5, 11, 4), "blue_fire")

    def test_wood_burns_away_and_flames_cling_to_it(self):
        self.assertIsNone(burn_seconds("stone"))
        self.assertEqual(burn_seconds("tall_grass"), 0.0)
        self.assertGreater(burn_seconds("log"), 0.0)
        self.assertGreater(burn_seconds("planks"), 0.0)
        self.assertTrue(is_flammable("log"))
        self.assertTrue(has_block_tag("log", "burnable"))
        self.assertTrue(has_block_tag("planks", "burnable"))
        self.assertFalse(has_block_tag("stone", "burnable"))

        world = World(seed=8, caves=[])
        world.load_chunk(0, 0, 0)
        for y in range(9, 14):
            for z in range(3, 7):
                for x in range(3, 7):
                    world.set_block(x, y, z, "air")
        world.set_block(4, 10, 4, "fire")
        alone = build_chunk_mesh(world, 0, 0, 0)
        world.set_block(5, 10, 4, "log")
        world.set_block(4, 11, 4, "planks")
        clung = build_chunk_mesh(world, 0, 0, 0)
        self.assertGreater(len(clung.cutout[0]), len(alone.cutout[0]))
        self.assertTrue(np.any(clung.cutout_colors[:, 0] > 1.5))
        pts = clung.cutout[0]
        near = pts[
            (pts[:, 0] > 3.5)
            & (pts[:, 0] < 5.5)
            & (pts[:, 1] > 9.5)
            & (pts[:, 1] < 11.5)
            & (pts[:, 2] > 3.5)
            & (pts[:, 2] < 5.5)
        ]
        # The standing cross meets the cell corner. Face sheets do not:
        # the log's side is a vertical plane, and the planks' underside is flat.
        corner = np.all(np.abs(near[:, :3] - np.array([4.0, 10.0, 4.0])) < 0.05, axis=1)
        self.assertFalse(np.any(corner))
        self.assertTrue(np.any(np.abs(near[:, 0] - (4.0 + 15.2 / 16.0)) < 0.02))
        self.assertTrue(np.any(np.abs(near[:, 1] - (10.0 + 15.2 / 16.0)) < 0.02))
        side = near[np.abs(near[:, 0] - (4.0 + 15.2 / 16.0)) < 0.02]
        self.assertGreater(float(side[:, 1].max() - side[:, 1].min()), 0.9)
        top = near[np.abs(near[:, 1] - (10.0 + 15.2 / 16.0)) < 0.02]
        self.assertLess(float(top[:, 1].max() - top[:, 1].min()), 0.05)

        world.set_block(4, 10, 4, "air")
        world.set_block(4, 11, 4, "air")
        world.set_block(5, 10, 4, "air")
        world.set_block(1, 12, 0, "air")
        world.set_block(2, 11, 0, "stone")
        world.set_block(1, 11, 1, "stone")
        world.set_block(1, 11, 0, "log")
        world.set_block(0, 11, 0, "lava")
        world.update_fire(0.5)
        self.assertEqual(world.get_block(1, 12, 0), "fire")
        self.assertEqual(world.get_block(1, 11, 0), "log")
        for _ in range(40):
            world.update_fire(0.5)
        self.assertNotEqual(world.get_block(1, 11, 0), "log")

        world.set_block(6, 11, 6, "blue_fire")
        world.set_block(3, 11, 3, "fire")
        player = Player(world)
        player.gamemode = "survival"
        player.pos = [6.5, 11.0, 6.5]
        player.update_health(0.05)
        self.assertEqual(player.fire_kind, "blue_fire")
        self.assertEqual(body_flame(world, player.pos, player.radius, player.height), "blue_fire")
        player.pos = [3.5, 11.0, 3.5]
        player.update_health(0.05)
        self.assertEqual(player.fire_kind, "fire")

        mob = world.spawn_mob("pig", pos=(6.5, 11.0, 6.5))
        before = mob.health
        mob.update(0.05, world)
        self.assertEqual(mob.fire_kind, "blue_fire")
        self.assertLess(mob.health, before)

    def test_fire_types_come_from_assets(self):
        import json
        from server.blocks import ASSETS

        ordinary = json.loads((ASSETS / "fire" / "fire.json").read_text(encoding="utf-8"))
        blue = json.loads((ASSETS / "fire" / "blue_fire.json").read_text(encoding="utf-8"))
        limits = json.loads((ASSETS / "fire.json").read_text(encoding="utf-8"))
        self.assertEqual(set(FIRE_TYPES), {"fire", "blue_fire"})
        self.assertEqual(fire_type("fire").burn_window, ordinary["burn_window"])
        self.assertEqual(fire_type("fire").spread_chance, ordinary["spread_chance"])
        self.assertEqual(fire_type("fire").extinguish_chance, ordinary["extinguish_chance"])
        self.assertEqual(tuple(fire_type("fire").life), tuple(ordinary["life"]))
        self.assertEqual(fire_type("blue_fire").cooldown, blue["cooldown"])
        self.assertGreater(fire_type("blue_fire").priority, fire_type("fire").priority)
        self.assertEqual(FIRE_SIMULATION.tick_interval, limits["tick_interval"])
        self.assertEqual(FIRE_SIMULATION.max_new_flames, limits["max_new_flames"])
        self.assertEqual(FIRE_SIMULATION.flame_cap, limits["flame_cap"])
        self.assertEqual(flame_kind("fire"), "fire")
        self.assertEqual(flame_kind("blue_lava"), "blue_fire")
        self.assertEqual(flame_kind("stone"), "")

    def test_a_forest_fire_goes_out_with_trees_left(self):
        died_out = 0
        for seed in range(4):
            world = World(seed=20 + seed, caves=[])
            world.load_chunk(0, 0, 0)
            for y in range(8, 16):
                for z in range(2, 14):
                    for x in range(2, 14):
                        world.set_block(x, y, z, "log" if y < 11 else "leaves")
            world.set_block(7, 12, 7, "fire")
            for _ in range(40):
                world.update_fire(0.5)
            if world.fire.flames:
                continue
            leaves = 0
            for y in range(11, 16):
                for z in range(2, 14):
                    for x in range(2, 14):
                        if world.get_block(x, y, z) == "leaves":
                            leaves += 1
            # One cell started as fire (719 leaves). A patch burns, the rest stays.
            if 500 < leaves < 719:
                died_out += 1
        self.assertGreaterEqual(died_out, 3)

    def test_a_burned_out_flame_does_not_leave_its_light(self):
        world = World(seed=11, caves=[])
        for cx, cy, cz in ((0, 2, 0), (1, 2, 0), (0, 3, 0), (0, 2, 1)):
            world.load_chunk(cx, cy, cz)
        for y in range(38, 44):
            for z in range(6, 12):
                for x in range(10, 22):
                    world.set_block(x, y, z, "air")
        world.set_block(15, 40, 8, "fire")
        self.assertGreater(world.get_blocklight(15, 40, 8)[0], 8)
        self.assertGreater(world.get_blocklight(17, 40, 8)[0], 0)
        lighting = world.lighting
        lighting.begin_batch()
        world._fire_light_batch = True
        try:
            world.set_block(15, 40, 8, "air")
        finally:
            world._fire_light_batch = False
            lighting.end_batch()
        self.assertEqual(world.get_block(15, 40, 8), "air")
        self.assertEqual(world.get_blocklight(15, 40, 8)[0], 0)
        self.assertEqual(world.get_blocklight(17, 40, 8)[0], 0)
        self.assertEqual(world.get_blocklight(15, 42, 8)[0], 0)

    def test_items_are_destroyed_and_mobs_burn(self):
        world = World(seed=3, caves=[])
        world.load_chunk(0, 0, 0)
        world.set_block(2, 10, 2, "lava")
        item = world.spawn_item("stone", pos=(2.5, 10.2, 2.5))
        item.update(0.05, world)
        self.assertEqual(item.count, 0)

        mob = world.spawn_mob("pig", pos=(2.5, 10.2, 2.5))
        before = mob.health
        mob.update(0.05, world)
        self.assertLess(mob.health, before)

        player = Player(world)
        player.gamemode = "survival"
        player.pos = [2.5, 10.2, 2.5]
        before = player.health
        player.update_health(0.5)
        self.assertLess(player.health, before)

    def test_an_underground_cave_floor_can_hold_a_lava_pool(self):
        world = World(seed=4, caves=[])
        data = np.zeros((16, 16, 16), dtype=np.uint16)
        data[:] = block_index("stone")
        data[8, 8, 8] = 0
        data[9, 8, 8] = 0
        feat = ConfiguredFeature(
            "lava_pool",
            "pool",
            {
                "fluid": "lava",
                "underground_chance": 1,
                "radius_min": 1,
                "radius_max": 1,
                "depth": 1,
                "bed_blocks": ["stone"],
            },
        )
        world._place_underground_pool(data, 0, 32, 0, feat, lambda lx, lz: 80)
        self.assertTrue(np.any(data == block_index("lava")))
        self.assertEqual(int(data[8, 8, 8]), 0)

    def test_jumping_swims_up_through_lava(self):
        world = World(seed=5, caves=[])
        world.load_chunk(0, 0, 0)
        for y in range(10, 16):
            world.set_block(2, y, 2, "air")
        world.set_block(2, 10, 2, "lava")
        world.set_block(2, 11, 2, "lava")
        player = Player(world)
        player.pos = [2.5, 10.2, 2.5]
        player.vel = [0.0, -2.0, 0.0]
        player.input.jump = True
        player.update(0.1)
        self.assertGreater(player.pos[1], 10.2)

    def test_each_fluid_has_its_own_underwater_fog(self):
        water = submerged_fog("water")
        milk = submerged_fog("milk")
        lava = submerged_fog("lava_falling")
        blue = submerged_fog("blue_lava")
        self.assertIsNone(submerged_fog("stone"))
        self.assertIsNone(submerged_fog("air"))
        self.assertLess(water[2], 22.0)
        self.assertLess(lava[2], water[2])
        self.assertNotEqual(water[0], milk[0])
        self.assertNotEqual(lava[0], blue[0])

    def test_lava_is_carried_in_an_iron_bucket(self):
        self.assertEqual(get_filled_bucket("iron_bucket", "lava"), "lava_bucket")
        self.assertEqual(get_filled_bucket("iron_bucket", "blue_lava"), "blue_lava_bucket")
        self.assertEqual(get_filled_bucket("iron_bucket", "black_lava"), "black_lava_bucket")
        self.assertIsNone(get_filled_bucket("wood_bucket", "lava"))
        self.assertIsNone(get_filled_bucket("clay_bucket", "lava"))
        self.assertEqual(get_bucket_fluid("lava_bucket"), "lava")
        self.assertNotIn("lava", CREATIVE)
        self.assertNotIn("black_fire", BLOCKS)
        self.assertEqual(BLOCKS["black_lava"].properties["lava"], "fire")
        self.assertEqual(len(BLOCKS["fire"].anim_rects), 4)
        self.assertEqual(len(BLOCKS["blue_fire"].anim_rects), 4)

    def test_lava_is_drawn_as_a_solid_surface(self):
        world = World(seed=6, caves=[])
        world.load_chunk(0, 0, 0)
        for y in range(8, 13):
            for z in range(2, 6):
                for x in range(2, 6):
                    world.set_block(x, y, z, "air")
        world.set_block(3, 10, 3, "lava")
        world.set_block(4, 10, 3, "fire")
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh.cutout_colors)
        self.assertTrue(np.any(mesh.cutout_colors[:, 0] > 1.5))
        if mesh.translucent_colors is not None:
            self.assertFalse(np.any(mesh.translucent_colors[:, 0] > 1.5))

    def test_flowing_lava_does_not_relight_every_loaded_chunk(self):
        world = World(seed=7, caves=[])
        for cx in range(-2, 3):
            for cz in range(-2, 3):
                for cy in range(0, 3):
                    world.load_chunk(cx, cy, cz)
        for x in range(0, 12):
            for z in range(0, 12):
                world.set_block(x, 20, z, "stone")
        world.set_block(4, 21, 4, "lava")
        calls = []
        original = world.lighting.compute_chunk_lighting

        def _count(cx, cy, cz, spread=True):
            calls.append((cx, cy, cz))
            return original(cx, cy, cz, spread=spread)

        world.lighting.compute_chunk_lighting = _count
        world.update_fluids(0.5)
        world.update_fluids(0.5)
        # Spreading used to recompute every loaded section many times over.
        self.assertLess(len(calls), 40)
        repeats = {}
        for key in calls:
            repeats[key] = repeats.get(key, 0) + 1
        self.assertLess(max(repeats.values()) if repeats else 0, 6)
        self.assertGreater(world.get_blocklight(5, 21, 4)[0], 0)
        for x in range(0, 16):
            for z in range(0, 16):
                for y in range(18, 26):
                    block_id = world.get_block(x, y, z)
                    if block_id and "lava" in str(block_id):
                        world.set_block(x, y, z, "air")
        self.assertEqual(world.get_blocklight(4, 21, 4), (0, 0, 0, 0, 0))
        self.assertEqual(world.get_blocklight(5, 21, 4), (0, 0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
