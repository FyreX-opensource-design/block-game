import tempfile
import unittest

from server.blocks import AIR, ASSETS, BLOCKS, can_place_block
from server.items import ITEMS
from server.loot import get_block_drops
from server.saplings import (
    DEFAULT_MAX_TIME,
    DEFAULT_MIN_TIME,
    load_saplings,
    sapling_from_mapping,
)
from server.storage import WorldCatalog
from server.trees import DECORATORS
from server.world import World


class _AlwaysDropRng:
    def random(self):
        return 0.0

    def randint(self, low, high):
        return low


def _flat_world():
    world = World(seed=1)
    world.load_chunk(0, 5, 0)
    world.chunks[(0, 5, 0)].data.fill(0)
    return world


class SaplingTests(unittest.TestCase):
    def test_default_sapling_assets_load(self):
        saplings = {sapling.id: sapling for sapling in load_saplings()}
        tree_ids = {tree.id for tree in DECORATORS}

        self.assertEqual(set(saplings), {"oak_sapling", "pine_sapling"})
        self.assertEqual(saplings["oak_sapling"].block, "oak_sapling")
        self.assertEqual(set(saplings["oak_sapling"].tree_ids), {"oak", "fancy_oak", "tall_oak"})
        self.assertEqual(saplings["pine_sapling"].tree_ids, ("pine",))
        for sapling in saplings.values():
            self.assertTrue(set(sapling.tree_ids) <= tree_ids)
            self.assertGreaterEqual(sapling.min_time, 15 * 60)
            self.assertLessEqual(sapling.max_time, 30 * 60)

    def test_sapling_blocks_and_items_are_registered(self):
        for sapling_id in ("oak_sapling", "pine_sapling"):
            block = BLOCKS[sapling_id]
            self.assertFalse(block.solid)
            self.assertTrue(block.cutout)
            self.assertTrue(block.requires_soil)
            self.assertTrue(block.fluid_breakable)

            item = ITEMS[sapling_id]
            self.assertEqual(item.block, sapling_id)
            self.assertEqual(item.right, "place")
            self.assertTrue((ASSETS / "textures" / f"{sapling_id}.png").is_file())

    def test_leaves_drop_their_own_sapling(self):
        oak_drops = dict(get_block_drops("leaves", rng=_AlwaysDropRng()))
        pine_drops = dict(get_block_drops("pine_leaves", rng=_AlwaysDropRng()))

        self.assertEqual(oak_drops.get("oak_sapling"), 1)
        self.assertNotIn("pine_sapling", oak_drops)
        self.assertEqual(pine_drops.get("pine_sapling"), 1)
        self.assertNotIn("oak_sapling", pine_drops)

    def test_saplings_can_only_be_placed_on_soil(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "dirt")
        world.set_block(6, 80, 5, "stone")

        dirt_hit = {"x": 5, "y": 80, "z": 5, "ny": 1}
        stone_hit = {"x": 6, "y": 80, "z": 5, "ny": 1}
        self.assertTrue(can_place_block(world, 5, 81, 5, "oak_sapling", hit=dirt_hit))
        self.assertFalse(can_place_block(world, 6, 81, 5, "oak_sapling", hit=stone_hit))

    def test_placed_sapling_grows_into_a_configured_tree(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "dirt")
        world.set_block(5, 81, 5, "oak_sapling")
        self.assertIn((5, 81, 5), world.saplings.timers)

        world.saplings.timers[(5, 81, 5)][1] = 0.1
        grown = world.update_saplings(1.0)

        self.assertEqual(len(grown), 1)
        self.assertIn(grown[0][3], {"oak", "fancy_oak", "tall_oak"})
        self.assertEqual(world.get_block(5, 81, 5), "log")
        self.assertTrue(
            any(
                world.get_block(x, y, z) == "leaves"
                for x in range(2, 9)
                for y in range(81, 95)
                for z in range(2, 9)
            )
        )

    def test_removing_soil_removes_the_sapling(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "dirt")
        world.set_block(5, 81, 5, "pine_sapling")

        world.set_block(5, 80, 5, AIR)

        self.assertEqual(world.get_block(5, 81, 5), AIR)
        self.assertNotIn((5, 81, 5), world.saplings.timers)
        self.assertTrue(any(entity.item == "pine_sapling" for entity in world.entities))

    def test_replacing_a_sapling_starts_a_new_timer(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "dirt")
        world.set_block(5, 81, 5, "oak_sapling")
        old_timer = world.saplings.timers[(5, 81, 5)]
        old_timer[1] = 1.0

        world.set_block(5, 81, 5, "pine_sapling")

        timer = world.saplings.timers[(5, 81, 5)]
        self.assertIsNot(timer, old_timer)
        self.assertEqual(timer[0].block, "pine_sapling")
        self.assertGreater(timer[1], 60.0)

    def test_growth_timers_are_not_reset_by_rescanning(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "dirt")
        world.set_block(5, 81, 5, "oak_sapling")
        timer = world.saplings.timers[(5, 81, 5)]
        timer[1] = 123.0

        world.saplings.scan_chunk((0, 5, 0))

        self.assertIs(world.saplings.timers[(5, 81, 5)], timer)
        self.assertEqual(timer[1], 123.0)

    def test_growth_timers_survive_world_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            info = catalog.create("Saplings", 1)
            store = catalog.store_for(info.id)
            world = World(seed=1, store=store)
            world.load_chunk(0, 5, 0)
            world.chunks[(0, 5, 0)].data.fill(0)
            world.set_block(5, 80, 5, "dirt")
            world.set_block(5, 81, 5, "oak_sapling")
            world.saplings.timers[(5, 81, 5)][1] = 42.5
            world.flush()

            restored = World(seed=1, store=store)
            self.assertIn((5, 81, 5), restored.saplings.timers)
            sapling, remaining = restored.saplings.timers[(5, 81, 5)]
            self.assertEqual(sapling.id, "oak_sapling")
            self.assertAlmostEqual(remaining, 42.5, places=2)

            restored.load_chunk(0, 5, 0)
            self.assertAlmostEqual(restored.saplings.timers[(5, 81, 5)][1], 42.5, places=2)

    def test_growth_timers_pause_while_chunk_is_unloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            info = catalog.create("Saplings", 1)
            store = catalog.store_for(info.id)
            world = World(seed=1, store=store)
            world.load_chunk(0, 5, 0)
            world.chunks[(0, 5, 0)].data.fill(0)
            world.set_block(5, 80, 5, "dirt")
            world.set_block(5, 81, 5, "oak_sapling")
            world.saplings.timers[(5, 81, 5)][1] = 30.0

            world.evict_chunk((0, 5, 0))
            world.update_saplings(10.0)
            self.assertAlmostEqual(world.saplings.timers[(5, 81, 5)][1], 30.0)

            world.load_chunk(0, 5, 0)
            world.update_saplings(10.0)
            self.assertAlmostEqual(world.saplings.timers[(5, 81, 5)][1], 20.0)

    def test_removed_sapling_does_not_come_back_after_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            info = catalog.create("Saplings", 1)
            store = catalog.store_for(info.id)
            world = World(seed=1, store=store)
            world.load_chunk(0, 5, 0)
            world.chunks[(0, 5, 0)].data.fill(0)
            world.set_block(5, 80, 5, "dirt")
            world.set_block(5, 81, 5, "pine_sapling")
            world.set_block(5, 81, 5, AIR)
            world.flush()

            restored = World(seed=1, store=store)
            self.assertNotIn((5, 81, 5), restored.saplings.timers)

    def test_custom_sapling_definition(self):
        sapling = sapling_from_mapping(
            {
                "id": "test_sapling",
                "block": "oak_sapling",
                "trees": ["oak"],
            },
            "test",
        )
        self.assertEqual(sapling.tree_ids, ("oak",))
        self.assertEqual(sapling.min_time, DEFAULT_MIN_TIME)
        self.assertEqual(sapling.max_time, DEFAULT_MAX_TIME)

    def test_invalid_sapling_definitions_are_rejected(self):
        with self.assertRaises(ValueError):
            sapling_from_mapping({"id": "x", "block": "missing", "trees": ["oak"]}, "t")
        with self.assertRaises(ValueError):
            sapling_from_mapping({"id": "x", "block": "oak_sapling", "trees": ["missing"]}, "t")
        with self.assertRaises(ValueError):
            sapling_from_mapping({"id": "x", "block": "oak_sapling", "trees": []}, "t")
        with self.assertRaises(ValueError):
            sapling_from_mapping(
                {"id": "x", "block": "oak_sapling", "trees": ["oak"], "min_time": 30, "max_time": 15},
                "t",
            )


if __name__ == "__main__":
    unittest.main()
