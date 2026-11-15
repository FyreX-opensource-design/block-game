import unittest

import numpy as np

from server.blocks import BLOCKS, CREATIVE, block_index
from server.features import ConfiguredFeature
from server.fluid import FLUIDS, get_bucket_fluid, get_filled_bucket, is_lava, is_water, submerged_fog
from server.mesh import build_chunk_mesh
from server.fire import is_flammable
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


if __name__ == "__main__":
    unittest.main()
