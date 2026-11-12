import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from server.blocks import (
    AIR,
    BLOCKS,
    COAL_ORE,
    COPPER_ORE,
    GRAVEL,
    IRON_ORE,
    STONE,
    block_index,
)
from server.items import ITEMS
from server.loot import get_block_drops
from server.ores import Ore, load_ores, ore_from_mapping
from server.recipes import (
    craft_match,
    fuel_burn_time,
    smelt_match,
)
from server.world import World


class OreTests(unittest.TestCase):
    def test_ore_assets_loaded(self):
        ores = load_ores()
        ore_ids = [ore.id for ore in ores]
        self.assertIn("coal", ore_ids)
        self.assertIn("copper", ore_ids)
        self.assertIn("iron", ore_ids)
        self.assertIn("gravel", ore_ids)

        coal = next(o for o in ores if o.id == "coal")
        self.assertEqual(coal.block, "coal_ore")
        self.assertEqual(coal.target, "stone")
        self.assertTrue(coal.enabled)

    def test_custom_ore_loading_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "test_ore.json").write_text(
                json.dumps(
                    {
                        "id": "test_ore",
                        "block": "gold_ore",
                        "target": "stone",
                        "min_y": 10,
                        "max_y": 60,
                        "scale": 0.15,
                        "threshold": 0.7,
                        "octaves": 3,
                        "seed_offset": 55,
                        "enabled": True,
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_ores(root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].id, "test_ore")
            self.assertEqual(loaded[0].block, "gold_ore")
            self.assertEqual(loaded[0].octaves, 3)

        with self.assertRaises(ValueError):
            ore_from_mapping({"id": "INVALID_UPPERCASE"}, "test")

        with self.assertRaises(ValueError):
            ore_from_mapping({"id": "bad_range", "min_y": 80, "max_y": 20}, "test")

    def test_blocks_and_items_registered(self):
        for ore_block in ("coal_ore", "copper_ore", "iron_ore"):
            self.assertIn(ore_block, BLOCKS)
            self.assertIn(ore_block, ITEMS)
            self.assertEqual(ITEMS[ore_block].block, ore_block)

        for item_id in (
            "coal",
            "raw_copper",
            "raw_copper_bit",
            "copper_nugget",
            "copper_dust",
            "copper_ingot",
            "raw_iron",
            "raw_iron_bit",
            "iron_nugget",
            "iron_dust",
            "iron_ingot",
        ):
            self.assertIn(item_id, ITEMS)

    def test_stone_ore_variants_registered(self):
        stones = ("granite", "diorite", "andesite", "marble", "limestone")
        ores = ("coal", "copper", "iron", "tin")
        mining_levels = {"coal": 1, "tin": 1, "copper": 2, "iron": 3}

        for s in stones:
            for o in ores:
                ore_block = f"{s}_{o}_ore"
                self.assertIn(ore_block, BLOCKS)
                self.assertIn(ore_block, ITEMS)
                self.assertEqual(ITEMS[ore_block].block, ore_block)
                self.assertEqual(BLOCKS[ore_block].mining_level, mining_levels[o])

    def test_stone_ore_variants_loot_drops(self):
        flint_pick = ITEMS["flint_pickaxe"]  # mining level 1
        stone_pick = ITEMS["stone_pickaxe"]  # mining level 2
        iron_pick = ITEMS["iron_pickaxe"]    # mining level 3
        stones = ("granite", "diorite", "andesite", "marble", "limestone")

        for s in stones:
            # Coal ore drops coal with flint pickaxe
            coal_drops = dict(get_block_drops(f"{s}_coal_ore", held_item=flint_pick))
            self.assertEqual(coal_drops.get("coal"), 1)

            # Tin ore drops raw_tin and bits with flint pickaxe
            tin_drops = dict(get_block_drops(f"{s}_tin_ore", held_item=flint_pick))
            self.assertEqual(tin_drops.get("raw_tin"), 1)
            self.assertIn("raw_tin_bit", tin_drops)

            # Copper ore drops raw_copper with flint pickaxe (min_level 1)
            copper_drops = dict(get_block_drops(f"{s}_copper_ore", held_item=flint_pick))
            self.assertEqual(copper_drops.get("raw_copper"), 1)

            # Iron ore requires min_level 2 (stone pickaxe); flint pick drops nothing
            self.assertEqual(get_block_drops(f"{s}_iron_ore", held_item=flint_pick), [])
            iron_drops = dict(get_block_drops(f"{s}_iron_ore", held_item=stone_pick))
            self.assertEqual(iron_drops.get("raw_iron"), 1)

            # Mining without pickaxe yields nothing
            self.assertEqual(get_block_drops(f"{s}_coal_ore", held_item=None), [])
            self.assertEqual(get_block_drops(f"{s}_iron_ore", held_item=None), [])

    def test_stone_ore_variants_smelting(self):
        stones = ("granite", "diorite", "andesite", "marble", "limestone")
        for s in stones:
            self.assertEqual(smelt_match(f"{s}_copper_ore")[:2], ("copper_ingot", 1))
            self.assertEqual(smelt_match(f"{s}_iron_ore")[:2], ("iron_ingot", 1))
            self.assertEqual(smelt_match(f"{s}_tin_ore")[:2], ("tin_ingot", 1))

    def test_ore_target_blocks_generation(self):
        # Test that an Ore with target_blocks mapping replaces stone variants accordingly
        target_map = {
            "stone": "coal_ore",
            "granite": "granite_coal_ore",
            "diorite": "diorite_coal_ore",
        }
        test_ore = Ore(
            "test_target_map_ore",
            "coal_ore",
            "stone",
            min_y=1,
            max_y=100,
            scale=0.1,
            threshold=-2.0,
            octaves=1,
            seed_offset=1,
            enabled=True,
            target_blocks=target_map,
        )
        world = World(seed=1, ores=[test_ore], caves=[])
        data = np.zeros((16, 16, 16), dtype=np.uint16)
        data[0, 0, 0] = block_index("stone")
        data[0, 0, 1] = block_index("granite")
        data[0, 0, 2] = block_index("diorite")
        data[0, 0, 3] = block_index("dirt")

        world._place_ores(data, 0, 16, 0)
        self.assertEqual(data[0, 0, 0], block_index("coal_ore"))
        self.assertEqual(data[0, 0, 1], block_index("granite_coal_ore"))
        self.assertEqual(data[0, 0, 2], block_index("diorite_coal_ore"))
        self.assertEqual(data[0, 0, 3], block_index("dirt"))

    def test_world_generation_contains_ores_and_gravel_in_stone(self):
        world = World(seed=42)
        found_blocks = set()
        # Generate multiple columns to check ore distribution
        for cx in range(-2, 3):
            for cz in range(-2, 3):
                chunk_col = world.generate_column(cx, cz)
                for chunk in chunk_col:
                    unique = set(chunk.data.flatten())
                    for idx in unique:
                        block = BLOCKS.get(world.chunk_at(chunk.cx * 16, chunk.cy * 16, chunk.cz * 16).data[0, 0, 0])
                    for y in range(16):
                        for z in range(16):
                            for x in range(16):
                                b = world.get_block(chunk.cx * 16 + x, chunk.cy * 16 + y, chunk.cz * 16 + z)
                                if b in (COAL_ORE, COPPER_ORE, IRON_ORE, GRAVEL):
                                    found_blocks.add(b)

        self.assertIn(COAL_ORE, found_blocks)
        self.assertIn(COPPER_ORE, found_blocks)
        self.assertIn(IRON_ORE, found_blocks)
        self.assertIn(GRAVEL, found_blocks)

    def test_mining_drops_coal_ore(self):
        pickaxe = ITEMS["flint_pickaxe"]
        drops = get_block_drops("coal_ore", held_item=pickaxe)
        items = dict(drops)
        self.assertIn("coal", items)
        self.assertEqual(items["coal"], 1)

        # Mining without pickaxe drops nothing
        no_drops = get_block_drops("coal_ore", held_item=None)
        self.assertEqual(no_drops, [])

    def test_mining_drops_copper_and_iron_ore(self):
        flint_pick = ITEMS["flint_pickaxe"]
        stone_pick = ITEMS["stone_pickaxe"]

        # Seed 2 triggers the ~20% dust chance
        rng_with_dust = random.Random(2)
        copper_drops = get_block_drops("copper_ore", held_item=flint_pick, rng=rng_with_dust)
        copper_dict = dict(copper_drops)
        self.assertIn("raw_copper", copper_dict)
        self.assertEqual(copper_dict["raw_copper"], 1)
        self.assertIn("raw_copper_bit", copper_dict)
        self.assertGreaterEqual(copper_dict["raw_copper_bit"], 1)
        self.assertLessEqual(copper_dict["raw_copper_bit"], 3)
        self.assertIn("copper_dust", copper_dict)
        self.assertEqual(copper_dict["copper_dust"], 1)

        # Flint pickaxe cannot harvest iron ore (requires stone pickaxe level 2)
        self.assertEqual(get_block_drops("iron_ore", held_item=flint_pick), [])

        # Stone pickaxe can harvest iron ore
        # Seed 0 does not trigger dust
        rng_no_dust = random.Random(0)
        iron_drops = get_block_drops("iron_ore", held_item=stone_pick, rng=rng_no_dust)
        iron_dict = dict(iron_drops)
        self.assertIn("raw_iron", iron_dict)
        self.assertEqual(iron_dict["raw_iron"], 1)
        self.assertIn("raw_iron_bit", iron_dict)
        self.assertGreaterEqual(iron_dict["raw_iron_bit"], 1)
        self.assertLessEqual(iron_dict["raw_iron_bit"], 3)
        self.assertNotIn("iron_dust", iron_dict)

        # Seed 2 also triggers iron dust
        iron_dust_drops = dict(get_block_drops("iron_ore", held_item=stone_pick, rng=random.Random(2)))
        self.assertIn("iron_dust", iron_dust_drops)
        self.assertEqual(iron_dust_drops["iron_dust"], 1)

        # Without pickaxe: empty drops
        self.assertEqual(get_block_drops("copper_ore", held_item=None), [])
        self.assertEqual(get_block_drops("iron_ore", held_item=None), [])

    def test_smelting_recipes(self):
        # Copper smelting
        self.assertEqual(smelt_match("raw_copper")[:2], ("copper_ingot", 1))
        self.assertEqual(smelt_match("copper_ore")[:2], ("copper_ingot", 1))
        self.assertEqual(smelt_match("copper_dust")[:2], ("copper_ingot", 1))
        self.assertEqual(smelt_match("raw_copper_bit")[:2], ("copper_nugget", 1))

        # Iron smelting
        self.assertEqual(smelt_match("raw_iron")[:2], ("iron_ingot", 1))
        self.assertEqual(smelt_match("iron_ore")[:2], ("iron_ingot", 1))
        self.assertEqual(smelt_match("iron_dust")[:2], ("iron_ingot", 1))
        self.assertEqual(smelt_match("raw_iron_bit")[:2], ("iron_nugget", 1))

    def test_crafting_recipes(self):
        # 9 nuggets -> 1 ingot (copper)
        copper_nuggets_9 = ("copper_nugget",) * 9
        res = craft_match(copper_nuggets_9)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "copper_ingot")
        self.assertEqual(res[1], 1)

        # 1 ingot -> 9 nuggets (copper shapeless)
        res = craft_match(("copper_ingot",) + ("",) * 8)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "copper_nugget")
        self.assertEqual(res[1], 9)

        # 9 raw copper bits -> 1 raw copper
        raw_copper_bits_9 = ("raw_copper_bit",) * 9
        res = craft_match(raw_copper_bits_9)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "raw_copper")
        self.assertEqual(res[1], 1)

        # 1 raw copper -> 9 raw copper bits
        res = craft_match(("raw_copper",) + ("",) * 8)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "raw_copper_bit")
        self.assertEqual(res[1], 9)

        # 9 iron nuggets -> 1 iron ingot
        iron_nuggets_9 = ("iron_nugget",) * 9
        res = craft_match(iron_nuggets_9)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "iron_ingot")
        self.assertEqual(res[1], 1)

        # 1 iron ingot -> 9 iron nuggets
        res = craft_match(("iron_ingot",) + ("",) * 8)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "iron_nugget")
        self.assertEqual(res[1], 9)

        # 9 raw iron bits -> 1 raw iron
        raw_iron_bits_9 = ("raw_iron_bit",) * 9
        res = craft_match(raw_iron_bits_9)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "raw_iron")
        self.assertEqual(res[1], 1)

        # 1 raw iron -> 9 raw iron bits
        res = craft_match(("raw_iron",) + ("",) * 8)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "raw_iron_bit")
        self.assertEqual(res[1], 9)

    def test_coal_as_fuel(self):
        self.assertEqual(fuel_burn_time("coal"), 80.0)

    def test_max_vein_size_enforced(self):
        try:
            from scipy.ndimage import label
        except ImportError:
            self.skipTest("scipy not installed")

        # Ore with max_vein_size of 5 and threshold -1.0 (covers lots of stone)
        capped_ore = Ore(
            "capped",
            "coal_ore",
            "stone",
            min_y=10,
            max_y=60,
            scale=0.1,
            threshold=-1.0,
            octaves=1,
            seed_offset=12,
            enabled=True,
            max_vein_size=5,
        )
        world = World(seed=99, ores=[capped_ore], caves=[])
        chunk = world.generate_chunk(0, 1, 0)
        # Coal ore mask in this chunk
        coal_ore_idx = BLOCKS["coal_ore"].id
        coal_mask = np.zeros((16, 16, 16), dtype=bool)
        for y in range(16):
            for z in range(16):
                for x in range(16):
                    if world.get_block(x, y + 16, z) == "coal_ore":
                        coal_mask[y, z, x] = True

        self.assertTrue(np.any(coal_mask))
        labeled, num_features = label(coal_mask)
        for feat in range(1, num_features + 1):
            cluster_size = np.sum(labeled == feat)
            self.assertLessEqual(cluster_size, 5)

    def test_ores_generate_in_soil_layers_of_stone_surfaced_biomes(self):
        # In stone-surfaced biomes (such as cliff_beach), soil and subsoil are stone.
        # Ores targeting stone generate in these stone soil layers.
        from server.biomes import BIOMES, biome_at
        from server.noise import terrain_height

        everywhere_ore = Ore(
            "test_surface_ore",
            "iron_ore",
            "stone",
            min_y=1,
            max_y=255,
            scale=0.01,
            threshold=-2.0,  # Guarantees ore placement wherever target block is present
            octaves=1,
            seed_offset=0,
            enabled=True,
            max_vein_size=0,
        )
        world = World(seed=1, ores=[everywhere_ore], caves=[])

        cliff_pt = None
        for z in range(-300, 301, 8):
            for x in range(-300, 301, 8):
                b = biome_at(x, z, 1, BIOMES)
                if b.id == "cliff_beach":
                    h = terrain_height(x, z, 1)
                    if h >= 64:
                        cliff_pt = (x, z)
                        break
            if cliff_pt:
                break

        self.assertIsNotNone(cliff_pt, "Should find a cliff_beach coordinate")
        cx, cz = cliff_pt
        h = terrain_height(cx, cz, 1)
        world.generate_column(cx // 16, cz // 16)

        # The soil layer beneath the surface in cliff_beach is stone, so it should have ores placed
        soil_block = world.get_block(cx, h - 1, cz)
        self.assertEqual(soil_block, "iron_ore")


if __name__ == "__main__":
    unittest.main()
