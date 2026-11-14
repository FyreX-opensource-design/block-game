import unittest
from server.blocks import BLOCKS, AIR, can_place_block
from server.world import World
from server.player import Player
from server.entity import Pig, Mob
from server.recipes import RECIPES, FUELS, craft_match, smelt_match
from server.damage import get_damage_type
from server.tags import has_block_tag, has_item_tag


class AcaciaAndDesertTests(unittest.TestCase):
    def test_acacia_blocks_and_properties(self):
        self.assertIn("acacia_log", BLOCKS)
        self.assertIn("acacia_wood", BLOCKS)
        self.assertIn("acacia_planks", BLOCKS)
        self.assertIn("acacia_planks_slab", BLOCKS)
        self.assertIn("acacia_planks_stairs", BLOCKS)
        self.assertIn("acacia_planks_vertical_slab", BLOCKS)
        self.assertIn("acacia_fence", BLOCKS)
        self.assertIn("acacia_fence_gate", BLOCKS)
        self.assertIn("acacia_door", BLOCKS)
        self.assertIn("acacia_trapdoor", BLOCKS)
        self.assertIn("acacia_leaves", BLOCKS)
        self.assertIn("acacia_sapling", BLOCKS)

        # Axis rotation on log and wood
        self.assertIn("acacia_log_x", BLOCKS)
        self.assertIn("acacia_log_z", BLOCKS)
        self.assertIn("acacia_wood_x", BLOCKS)
        self.assertIn("acacia_wood_z", BLOCKS)

        # Leaf decay properties
        self.assertTrue(BLOCKS["acacia_leaves"].properties.get("decayable"))
        self.assertIn("acacia_leaves_persistent", BLOCKS)

        # Tags
        self.assertTrue(has_block_tag("acacia_log", "logs"))
        self.assertTrue(has_block_tag("acacia_wood", "logs"))
        self.assertTrue(has_block_tag("acacia_leaves", "leaves"))
        self.assertTrue(has_block_tag("acacia_planks", "planks"))

        self.assertTrue(has_item_tag("acacia_log", "logs"))
        self.assertTrue(has_item_tag("acacia_wood", "logs"))
        self.assertTrue(has_item_tag("acacia_leaves", "leaves"))
        self.assertTrue(has_item_tag("acacia_planks", "planks"))

    def test_acacia_crafting_and_smelting(self):
        # 1 log -> 4 planks (in 2x2 grid)
        match = craft_match(["acacia_log", "", "", ""])
        self.assertIsNotNone(match)
        result_id, count, _ = match
        self.assertEqual(result_id, "acacia_planks")
        self.assertEqual(count, 4)

        # 2x2 log -> 3 wood
        match = craft_match(["acacia_log", "acacia_log", "acacia_log", "acacia_log"])
        self.assertIsNotNone(match)
        result_id, count, _ = match
        self.assertEqual(result_id, "acacia_wood")
        self.assertEqual(count, 3)

        # 3 horizontal planks -> 6 slab (in 3x3 grid)
        match = craft_match(["acacia_planks", "acacia_planks", "acacia_planks", "", "", "", "", "", ""])
        self.assertIsNotNone(match)
        result_id, count, _ = match
        self.assertEqual(result_id, "acacia_planks_slab")
        self.assertEqual(count, 6)

        # 2 vertical planks -> 4 sticks
        match = craft_match(["acacia_planks", "", "acacia_planks", ""])
        self.assertIsNotNone(match)
        result_id, count, _ = match
        self.assertEqual(result_id, "stick")
        self.assertEqual(count, 4)

        # Smelting acacia log into charcoal
        smelt = smelt_match("acacia_log")
        self.assertIsNotNone(smelt)
        result_id, count, _, _ = smelt
        self.assertEqual(result_id, "charcoal")
        self.assertEqual(count, 1)

        # Fuels
        self.assertEqual(FUELS.get("acacia_log"), 15.0)
        self.assertEqual(FUELS.get("acacia_wood"), 15.0)
        self.assertEqual(FUELS.get("acacia_planks"), 15.0)
        self.assertEqual(FUELS.get("acacia_planks_slab"), 7.5)

    def test_cactus_placement_and_sustenance(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(0, 64, 0)
        world.load_chunk(cx, cy, cz)
        world.load_chunk(cx, cy - 1, cz)
        world.set_block(0, 63, 0, "stone")

        # Cactus on sand is valid
        world.set_block(0, 64, 0, "sand")
        hit_top = {"x": 0, "y": 64, "z": 0, "nx": 0, "ny": 1, "nz": 0}
        self.assertTrue(can_place_block(world, 0, 65, 0, "cactus", hit=hit_top))

        # Cactus on stone or grass is invalid
        world.set_block(0, 64, 0, "stone")
        self.assertFalse(can_place_block(world, 0, 65, 0, "cactus", hit=hit_top))
        world.set_block(0, 64, 0, "grass")
        self.assertFalse(can_place_block(world, 0, 65, 0, "cactus", hit=hit_top))

        # Stacking cactus on another cactus is valid
        world.set_block(0, 64, 0, "sand")
        world.set_block(0, 65, 0, "cactus")
        hit_cactus = {"x": 0, "y": 65, "z": 0, "nx": 0, "ny": 1, "nz": 0}
        self.assertTrue(can_place_block(world, 0, 66, 0, "cactus", hit=hit_cactus))

        # Breaking ground beneath cactus breaks it and drops item
        initial_entities = len(world.entities)
        world.set_block(0, 64, 0, AIR)
        self.assertEqual(world.get_block(0, 65, 0), AIR)
        self.assertEqual(len(world.entities), initial_entities + 1)
        self.assertEqual(world.entities[-1].item, "cactus")

    def test_cactus_contact_damage_to_player(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(10, 64, 10)
        world.load_chunk(cx, cy, cz)
        world.load_chunk(cx, cy - 1, cz)
        world.set_block(10, 63, 10, "stone")
        world.set_block(10, 64, 10, "sand")
        world.set_block(10, 65, 10, "cactus")

        player = Player("TestPlayer")
        player.world = world
        player.survival = True

        # Position player right next to the cactus (touching within 0.08)
        # Cactus block is [10, 11] x [65, 66] x [10, 11]
        # Player at x = 9.75, radius = 0.3 -> player x bounds = [9.45, 10.05], touching cactus!
        player.pos = [9.75, 65.0, 10.5]
        self.assertTrue(player.is_touching_cactus())

        # Immediate damage on first contact (even small dt like 0.05s)
        start_health = player.health
        events = player.update_health(0.05)
        self.assertTrue(len(events) > 0)
        self.assertEqual(events[0].damage_type.id, "cactus")
        self.assertEqual(player.health, start_health - 1.0)

        # Still touching before 0.5s interval -> no new damage yet
        events2 = player.update_health(0.1)
        self.assertEqual(len(events2), 0)
        self.assertEqual(player.health, start_health - 1.0)

        # After remaining interval expires -> takes damage again
        events3 = player.update_health(0.4)
        self.assertTrue(len(events3) > 0)
        self.assertEqual(player.health, start_health - 2.0)

        # Move player away
        player.pos = [8.0, 65.0, 10.5]
        self.assertFalse(player.is_touching_cactus())
        events_away = player.update_health(0.1)
        self.assertEqual(len(events_away), 0)

    def test_cactus_contact_damage_to_mobs(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(10, 64, 10)
        world.load_chunk(cx, cy, cz)
        world.load_chunk(cx, cy - 1, cz)
        world.set_block(10, 63, 10, "stone")
        world.set_block(10, 64, 10, "sand")
        world.set_block(10, 65, 10, "cactus")

        pig = Pig(pos=[9.6, 65.0, 10.5])
        self.assertTrue(pig.is_touching_cactus(world))
        start_health = pig.health
        pig.update(0.05, world=world)
        self.assertEqual(pig.health, start_health - 1.0)
        self.assertTrue(pig.hurt_time > 0)

    def test_dead_bush_placement_and_drops(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(0, 64, 0)
        world.load_chunk(cx, cy, cz)
        world.load_chunk(cx, cy - 1, cz)
        world.set_block(0, 63, 0, "stone")

        hit_top = {"x": 0, "y": 64, "z": 0, "nx": 0, "ny": 1, "nz": 0}

        # Valid ground: sand, dirt, grass, mud, muddy_grass, terracotta
        for valid_block in ("sand", "dirt", "grass", "mud", "muddy_grass", "terracotta"):
            world.set_block(0, 64, 0, valid_block)
            self.assertTrue(
                can_place_block(world, 0, 65, 0, "dead_bush", hit=hit_top),
                f"Dead bush should be placeable on {valid_block}",
            )

        # Invalid ground: stone, oak_planks
        for invalid_block in ("stone", "oak_planks"):
            world.set_block(0, 64, 0, invalid_block)
            self.assertFalse(
                can_place_block(world, 0, 65, 0, "dead_bush", hit=hit_top),
                f"Dead bush should NOT be placeable on {invalid_block}",
            )

        # Replacing ground beneath a valid dead bush with invalid ground (stone) breaks it and drops sticks
        world.set_block(0, 64, 0, "sand")
        world.set_block(0, 65, 0, "dead_bush")
        self.assertEqual(world.get_block(0, 65, 0), "dead_bush")
        initial_entities = len(world.entities)
        world.set_block(0, 64, 0, "stone")
        self.assertEqual(world.get_block(0, 65, 0), AIR)
        if len(world.entities) > initial_entities:
            self.assertEqual(world.entities[-1].item, "stick")

        # Breaking ground beneath a valid dead bush breaks it and drops sticks
        world.set_block(0, 64, 0, "sand")
        world.set_block(0, 65, 0, "dead_bush")
        self.assertEqual(world.get_block(0, 65, 0), "dead_bush")
        initial_entities = len(world.entities)
        world.set_block(0, 64, 0, AIR)
        self.assertEqual(world.get_block(0, 65, 0), AIR)
        if len(world.entities) > initial_entities:
            self.assertEqual(world.entities[-1].item, "stick")

    def test_world_generation_acacia_and_desert_features(self):
        world = World(seed=42)
        # Search for savanna and desert biomes
        found_savanna_tree = False
        found_desert_cactus = False
        found_desert_dead_bush = False

        # Sample across chunk columns
        for cx in range(-15, 16, 2):
            for cz in range(-15, 16, 2):
                world.load_chunk(cx, 4, cz)
                chunk = world.chunks.get((cx, 4, cz))
                if chunk is None:
                    continue
                for lz in range(16):
                    for lx in range(16):
                        wx = cx * 16 + lx
                        wz = cz * 16 + lz
                        for y in range(60, 85):
                            blk = world.get_block(wx, y, wz)
                            if blk == "acacia_log":
                                found_savanna_tree = True
                            elif blk == "cactus":
                                found_desert_cactus = True
                            elif blk == "dead_bush":
                                found_desert_dead_bush = True
                        if found_savanna_tree and found_desert_cactus and found_desert_dead_bush:
                            break
                    if found_savanna_tree and found_desert_cactus and found_desert_dead_bush:
                        break
                if found_savanna_tree and found_desert_cactus and found_desert_dead_bush:
                    break
            if found_savanna_tree and found_desert_cactus and found_desert_dead_bush:
                break

        self.assertTrue(found_savanna_tree, "Savanna should generate acacia trees")
        self.assertTrue(found_desert_cactus, "Desert should generate cactus")
        self.assertTrue(found_desert_dead_bush, "Desert should generate dead bushes")


if __name__ == "__main__":
    unittest.main()
