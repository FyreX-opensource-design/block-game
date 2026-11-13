import random
import unittest
from collections import Counter
from unittest.mock import MagicMock

from PIL import Image

from server.biomes import biome_at, BIOMES
from server.blocks import AIR, ASSETS, STONE
from server.cow_model import COW_PARTS, CowMeshes, build_cow_meshes, cow_pose
from server.entity import Cow, Pig, Sheep, entity_from_dict
from server.fluid import get_bucket_fluid, get_filled_bucket, get_fluid_def, is_fluid, is_fluid_source
from server.items import ITEMS
from server.recipes import smelt_match
from server.spawner import MobSpawner
from server.world import World


class CowTests(unittest.TestCase):
    def test_cow_serializes_and_can_be_spawned_by_name(self):
        cow = Cow(pos=[4.0, 70.0, -2.0], yaw=0.4, health=7.0)
        self.assertEqual(cow.mob_type, "cow")
        self.assertEqual(cow.height, 1.4)
        self.assertIsNotNone(cow.goals)

        restored = entity_from_dict(cow.to_dict())
        self.assertIsInstance(restored, Cow)
        self.assertEqual(restored.health, 7.0)
        self.assertEqual(restored.pos, [4.0, 70.0, -2.0])

        world = World(seed=1, caves=[])
        spawned = world.spawn_mob("cow", pos=[1.0, 2.0, 3.0])
        self.assertIsInstance(spawned, Cow)
        self.assertEqual(spawned.pos, [1.0, 2.0, 3.0])

    def test_cow_mesh_has_horns_udder_and_a_walk_pose(self):
        parts = build_cow_meshes()
        for name in ("body", "head", "snout", "udder", "horn_left", "horn_right", "leg_front_left"):
            vertices, indices = parts[name]
            self.assertGreater(len(vertices), 0, name)
            self.assertGreater(len(indices), 0, name)
        poses = cow_pose(1.0, 1.0, 0.0)
        vertices, indices = CowMeshes().posed_vertices([0.0, 0.0, 0.0], 0.2, poses)
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)
        head = COW_PARTS["head"]["box"]
        snout = COW_PARTS["snout"]["box"]
        head_mid = (head[1] + head[4]) / 2.0
        self.assertLess(snout[2], head[2])
        self.assertLessEqual(snout[4], head_mid)
        skin = Image.open(ASSETS / "textures" / "entity" / "cow.png").convert("RGBA")
        # The front face used to carry the eyes at the snout line. They sit
        # above that line now, and on the sides of the head.
        self.assertGreater(skin.getpixel((1, 27))[0], 200)
        self.assertLess(skin.getpixel((1, 25))[0], 80)
        self.assertLess(skin.getpixel((22, 25))[0], 80)
        self.assertLess(skin.getpixel((25, 25))[0], 80)

    def test_cow_drops_beef_and_leather_and_beef_cooks(self):
        world = World(seed=1, caves=[])
        cow = world.spawn_mob(Cow(pos=[2.0, 1.0, 2.0], health=1.0))
        cow.take_damage(5.0, attacker=MagicMock(pos=[0.0, 0.0, 0.0]), world=world)
        self.assertFalse(cow.alive)
        dropped = {getattr(entity, "item", "") for entity in world.entities}
        self.assertIn("raw_beef", dropped)
        self.assertIn("cow_leather", dropped)
        self.assertEqual(smelt_match("raw_beef")[0], "cooked_beef")
        self.assertEqual(ITEMS["raw_beef"].food, 3)
        self.assertEqual(ITEMS["cooked_beef"].food, 8)
        self.assertEqual(ITEMS["raw_beef"].right, "eat")

    def test_pig_and_sheep_drop_hides(self):
        world = World(seed=1, caves=[])
        pig = world.spawn_mob(Pig(pos=[2.0, 1.0, 2.0], health=1.0))
        pig.take_damage(5.0, attacker=MagicMock(pos=[0.0, 0.0, 0.0]), world=world)
        sheep = world.spawn_mob(Sheep(pos=[4.0, 1.0, 2.0], health=1.0))
        sheep.take_damage(5.0, attacker=MagicMock(pos=[0.0, 0.0, 0.0]), world=world)
        dropped = {getattr(entity, "item", "") for entity in world.entities}
        self.assertIn("pig_hide", dropped)
        self.assertIn("raw_porkchop", dropped)
        self.assertIn("sheep_hide", dropped)
        self.assertIn("raw_mutton", dropped)

    def test_empty_buckets_milk_a_cow(self):
        cow = Cow()
        self.assertEqual(cow.milk("iron_bucket"), "milk_bucket")
        self.assertEqual(cow.milk("wood_bucket"), "wood_milk_bucket")
        self.assertEqual(cow.milk("clay_bucket"), "clay_milk_bucket")
        self.assertIsNone(cow.milk("water_bucket"))
        self.assertIsNone(cow.milk("stick"))
        cow.alive = False
        self.assertIsNone(cow.milk("iron_bucket"))
        self.assertEqual(get_bucket_fluid("milk_bucket"), "milk")
        self.assertEqual(get_filled_bucket("iron_bucket", "milk"), "milk_bucket")

    def test_milk_is_a_placeable_finite_fluid(self):
        self.assertTrue(is_fluid("milk"))
        self.assertTrue(is_fluid_source("milk"))
        self.assertFalse(get_fluid_def("milk").infinite_source)

        world = World(seed=1, caves=[])
        world.load_chunk(0, 5, 0)
        world.chunks[(0, 5, 0)].data.fill(0)
        for x in range(8):
            world.set_block(x, 80, 5, STONE)
            world.set_block(x, 81, 4, STONE)
            world.set_block(x, 81, 6, STONE)
        world.set_block(1, 83, 5, "milk")
        for _ in range(8):
            world.update_fluids(0.25)
        self.assertEqual(world.get_block(1, 82, 5), "milk_falling")

        for y in range(81, 86):
            world.set_block(1, y, 5, AIR)
        world.set_block(1, 81, 5, "milk")
        world.set_block(3, 81, 5, "milk")
        for _ in range(12):
            world.update_fluids(0.25)
        middle = world.get_block(2, 81, 5)
        self.assertTrue(str(middle).startswith("milk"))
        self.assertNotEqual(middle, "milk")

    def test_cows_mainly_spawn_in_plains_and_savanna(self):
        spawner = MobSpawner(World(seed=2, caves=[]))
        plains = savanna = other = None
        for z in range(-400, 401, 8):
            for x in range(-400, 401, 8):
                biome_id = biome_at(x, z, 2, BIOMES).id
                if biome_id == "plains" and plains is None:
                    plains = (x, z)
                elif biome_id == "savanna" and savanna is None:
                    savanna = (x, z)
                elif biome_id in ("forest", "desert", "pine_forest") and other is None:
                    other = (x, z)
            if plains and savanna and other:
                break
        self.assertIsNotNone(plains)
        self.assertIsNotNone(savanna)
        self.assertIsNotNone(other)

        pasture, outside = spawner.passive_mob_weights("plains"), spawner.passive_mob_weights("forest")
        self.assertEqual(pasture[0][0], "cow")
        self.assertGreater(pasture[1][0], pasture[1][1])
        self.assertLess(outside[1][0], outside[1][1])
        self.assertEqual(spawner.passive_mob_weights("savanna"), pasture)
        self.assertEqual(spawner.passive_mob_weights("savanna_hills"), pasture)
        self.assertEqual(spawner.passive_mob_weights("desert"), outside)

        def tally(origin):
            random.seed(4)
            counts = Counter(spawner._passive_mob_type(*origin) for _ in range(240))
            return counts

        plains_counts = tally(plains)
        other_counts = tally(other)
        self.assertGreater(plains_counts["cow"], plains_counts["pig"] + plains_counts["sheep"])
        self.assertGreater(other_counts["pig"] + other_counts["sheep"], other_counts["cow"])


if __name__ == "__main__":
    unittest.main()
