import unittest
import random
from server.blocks import AIR, CLAY, GRAVEL, GRASS, LOG, STONE, WATER
from server.loot import get_block_drops, load_loot_table, roll_loot_table
from server.recipes import find_block_hit_recipe, find_fluid_transform, RECIPES
from server.features import FEATURES, load_features
from server.trees import DECORATORS
from server.world import World


class InteractionAndFeatureTests(unittest.TestCase):
    def test_block_hit_recipe_flint_on_stone(self):
        # 1. Recipe lookup
        recipe_stone = find_block_hit_recipe("flint", "stone")
        self.assertIsNotNone(recipe_stone)
        self.assertEqual(recipe_stone.chance, 0.60)
        self.assertEqual(recipe_stone.loot_table, "knapped_flint")
        self.assertTrue(recipe_stone.consume_item)

        recipe_cobble = find_block_hit_recipe("flint", "cobble")
        self.assertIsNotNone(recipe_cobble)

        recipe_dirt = find_block_hit_recipe("flint", "dirt")
        self.assertIsNone(recipe_dirt)

        # 2. Loot table for knapped flint
        table = load_loot_table("knapped_flint")
        self.assertIsNotNone(table)
        drops = roll_loot_table(table)
        for drop_id, count in drops:
            self.assertEqual(drop_id, "knapped_flint")
            self.assertIn(count, (0, 1, 2))

        # 3. Striking in World
        world = World()
        world.load_chunk(0, 3, 0)
        world.set_block(4, 50, 4, "stone")

        # Mock deterministic RNG: roll <= 0.60 should trigger
        class MockRNG:
            def __init__(self, val):
                self.val = val

            def random(self):
                return self.val

            def randint(self, a, b):
                return 1

        trig, drops, consume = world.hit_block_with_item(4, 50, 4, "flint", rng=MockRNG(0.5))
        self.assertTrue(trig)
        self.assertTrue(consume)
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0][0], "knapped_flint")
        self.assertEqual(drops[0][1], 1)

        # Failure roll > 0.60
        trig_fail, drops_fail, consume_fail = world.hit_block_with_item(4, 50, 4, "flint", rng=MockRNG(0.8))
        self.assertFalse(trig_fail)
        self.assertEqual(drops_fail, [])

    def test_gravel_drops_flint_sometimes(self):
        # Gravel should drop flint sometimes instead of gravel block
        counts = {"flint": 0, "gravel": 0}
        for _ in range(500):
            drops = get_block_drops("gravel")
            self.assertEqual(len(drops), 1)
            item_id, count = drops[0]
            self.assertEqual(count, 1)
            self.assertIn(item_id, counts)
            counts[item_id] += 1

        self.assertGreater(counts["flint"], 0, "Flint should drop sometimes from gravel")
        self.assertGreater(counts["gravel"], counts["flint"], "Gravel should drop more often than flint")

    def test_fluid_collision_transformation(self):
        # 1. Recipe lookup
        self.assertEqual(find_fluid_transform("water", "dirt"), "mud")
        self.assertEqual(find_fluid_transform("water", "grass"), "muddy_grass")
        self.assertEqual(find_fluid_transform("water_falling", "dirt"), "mud")
        self.assertEqual(find_fluid_transform("water_falling", "grass"), "muddy_grass")

        # 2. Simulation transformation: water touching dirt becomes mud
        world = World()
        world.load_chunk(0, 3, 0)
        world.set_block(4, 50, 4, "dirt")
        world.set_block(4, 50, 5, "water")
        world.fluid.update(0.2)
        self.assertEqual(world.get_block(4, 50, 4), "mud")

        # 3. Simulation transformation: water touching grass becomes muddy_grass
        world.set_block(6, 50, 4, "grass")
        world.set_block(6, 50, 5, "water")
        world.fluid.update(0.2)
        self.assertEqual(world.get_block(6, 50, 4), "muddy_grass")

    def test_tree_definitions_support_mud_and_muddy_grass(self):
        # Tree decorator should have valid_blocks containing mud and muddy_grass
        oak = next(t for t in DECORATORS if t.id == "oak")
        self.assertIn("mud", oak.valid_blocks)
        self.assertIn("muddy_grass", oak.valid_blocks)
        self.assertIn("grass", oak.valid_blocks)
        self.assertIn("dirt", oak.valid_blocks)

        world = World()
        world.load_chunk(0, 3, 0)
        chunk = world.chunks[(0, 3, 0)]

        # Muddy grass supports tree growth
        world.set_block(5, 50, 5, "muddy_grass")
        world.set_block(5, 51, 5, AIR)
        world._place_tree(chunk.data, 5, 5, 50, 5, 5, 48, oak)
        self.assertEqual(world.get_block(5, 51, 5), LOG)

        # Mud supports tree growth
        world.set_block(9, 50, 9, "mud")
        world.set_block(9, 51, 9, AIR)
        world._place_tree(chunk.data, 9, 9, 50, 9, 9, 48, oak)
        self.assertEqual(world.get_block(9, 51, 9), LOG)

        # Stone rejects tree growth
        world.set_block(12, 50, 12, "stone")
        world.set_block(12, 51, 12, AIR)
        world._place_tree(chunk.data, 12, 12, 50, 12, 12, 48, oak)
        self.assertEqual(world.get_block(12, 51, 12), AIR)

    def test_configured_feature_water_lake_lua(self):
        # Feature loaded from lua
        self.assertTrue(len(FEATURES) >= 1)
        lake_feat = next((f for f in FEATURES if f.id == "water_lake"), None)
        self.assertIsNotNone(lake_feat)
        self.assertEqual(lake_feat.type, "lake")
        self.assertEqual(lake_feat.fluid, "water")
        self.assertIn("clay", lake_feat.bed_blocks)
        self.assertIn("gravel", lake_feat.bed_blocks)
        self.assertIn("mud", lake_feat.shore_blocks)
        self.assertIn("muddy_grass", lake_feat.shore_blocks)

        # World generation generates lakes with water, bed, and mud/muddy grass shores
        world = World(seed=42)
        mud_found = False
        water_found = False
        for cz in range(-4, 5):
            for cx in range(-4, 5):
                world.generate_column(cx, cz)
                for z in range(cz * 16, (cz + 1) * 16, 2):
                    for x in range(cx * 16, (cx + 1) * 16, 2):
                        for y in range(65, 110):
                            b = world.get_block(x, y, z)
                            if b in ("mud", "muddy_grass"):
                                mud_found = True
                            if b == WATER:
                                water_found = True
                        if mud_found and water_found:
                            break
                    if mud_found and water_found:
                        break
            if mud_found and water_found:
                break

        self.assertTrue(mud_found, "Lakes should generate mud and muddy grass shores")
        self.assertTrue(water_found, "Lakes should contain water")

        # Verify lakes only generate exposed to the surface and not underground
        for cz in range(-4, 5):
            for cx in range(-4, 5):
                for z in range(cz * 16, (cz + 1) * 16, 4):
                    for x in range(cx * 16, (cx + 1) * 16, 4):
                        for y in range(0, 65):
                            b = world.get_block(x, y, z)
                            self.assertNotIn(b, ("mud", "muddy_grass"), f"Lake shores should not generate underground at y={y}")

    def test_block_hit_recipe_in_creative_does_not_break_block(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from server.stacks import make_stack

        mock_game = MagicMock()
        mock_game.chat.open = False
        mock_game.mode = "play"
        mock_game.gamemode = "creative"
        mock_game.selected = 0
        mock_game.hotbar = [make_stack("flint", 1)]
        mock_game.world = World(seed=1)
        mock_game.world.load_chunk(0, 0, 0)
        mock_game.world.set_block(0, 5, 0, "stone")
        mock_game.network_client = None
        mock_game.player = MagicMock()
        mock_game.player.pos = [0.0, 5.0, -2.0]
        mock_game.player.eye = 1.6
        mock_game.player.yaw = 0.0
        mock_game.player.pitch = 0.0

        # Aim at stone block
        mock_game.target = {"x": 0, "y": 5, "z": 0, "nx": 0, "ny": 0, "nz": -1}
        mock_game._aim = MagicMock()
        mock_game._break_block = MagicMock()
        mock_game._handle_block_hit_interaction = lambda hit: Game._handle_block_hit_interaction(mock_game, hit)

        # Left click (button 1) on stone in creative with flint
        Game.on_mouse_press(mock_game, 0, 0, 1, 0)

        # Stone block should NOT have been broken
        mock_game._break_block.assert_not_called()
        self.assertEqual(mock_game.world.get_block(0, 5, 0), "stone")

        # Now left click with shift (sneaking to intentionally break, key.MOD_SHIFT = 1)
        Game.on_mouse_press(mock_game, 0, 0, 1, 1)
        mock_game._break_block.assert_called_once()


if __name__ == "__main__":
    unittest.main()

