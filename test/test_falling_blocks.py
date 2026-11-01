import unittest
from server.blocks import AIR, ASSETS, BLOCKS, block_index
from server.recipes import craft_match, smelt_match
from server.entity import FallingBlockEntity, ItemEntity
from server.images import read_png
from server.loot import get_block_drops
from server.player import Player
from server.world import Chunk, World


class TestFallingBlocksAndClay(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        # Create chunk (0, 0, 0) and clear it
        self.chunk = Chunk(0, 0, 0)
        self.chunk.data.fill(0)
        self.world.chunks[(0, 0, 0)] = self.chunk

    def test_block_gravity_trait(self):
        self.assertTrue(BLOCKS["sand"].gravity)
        self.assertTrue(BLOCKS["gravel"].gravity)
        self.assertFalse(BLOCKS["stone"].gravity)
        self.assertFalse(BLOCKS["dirt"].gravity)

    def test_unsupported_placed_block_falls(self):
        # Place stone at (5, 5, 5) -> stays because stone has no gravity
        self.world.set_block(5, 5, 5, "stone")
        self.assertEqual(self.world.get_block(5, 5, 5), "stone")
        self.assertEqual(len(self.world.entities), 0)

        # Place sand at (5, 10, 5) where (5, 9, 5) is air -> immediately falls!
        self.world.set_block(5, 10, 5, "sand")
        self.assertEqual(self.world.get_block(5, 10, 5), "air")
        self.assertEqual(len(self.world.entities), 1)
        entity = self.world.entities[0]
        self.assertIsInstance(entity, FallingBlockEntity)
        self.assertEqual(entity.block_id, "sand")
        self.assertEqual(entity.pos, [5.5, 10.0, 5.5])

    def test_supported_placed_block_stays(self):
        self.world.set_block(5, 5, 5, "stone")
        # Place sand on top of stone at (5, 6, 5) -> stays!
        self.world.set_block(5, 6, 5, "sand")
        self.assertEqual(self.world.get_block(5, 6, 5), "sand")
        self.assertEqual(len(self.world.entities), 0)

    def test_block_update_triggers_fall(self):
        # Build tower: stone at y=5, sand at y=6
        self.world.set_block(5, 5, 5, "stone")
        self.world.set_block(5, 6, 5, "sand")
        self.assertEqual(self.world.get_block(5, 6, 5), "sand")
        self.assertEqual(len(self.world.entities), 0)

        # Remove stone beneath sand -> sand falls!
        self.world.set_block(5, 5, 5, "air")
        self.assertEqual(self.world.get_block(5, 6, 5), "air")
        self.assertEqual(len(self.world.entities), 1)
        self.assertEqual(self.world.entities[0].block_id, "sand")

    def test_column_cascade(self):
        # Set up a tower of gravel: stone at y=5, gravel at y=6, 7, 8
        self.world.set_block(3, 5, 3, "stone")
        self.world.set_block(3, 6, 3, "gravel")
        self.world.set_block(3, 7, 3, "gravel")
        self.world.set_block(3, 8, 3, "gravel")
        self.assertEqual(len(self.world.entities), 0)

        # Break the stone at the bottom
        self.world.set_block(3, 5, 3, "air")

        # All 3 gravel blocks should have converted to falling blocks!
        falling = [e for e in self.world.entities if isinstance(e, FallingBlockEntity)]
        self.assertEqual(len(falling), 3)
        self.assertEqual(self.world.get_block(3, 6, 3), "air")
        self.assertEqual(self.world.get_block(3, 7, 3), "air")
        self.assertEqual(self.world.get_block(3, 8, 3), "air")

    def test_falling_and_landing_on_ground(self):
        # Stone floor at y=2
        self.world.set_block(4, 2, 4, "stone")

        # Spawn falling sand at y=5
        falling = self.world.spawn_falling_block("sand", pos=[4.5, 5.0, 4.5])
        self.assertIn(falling, self.world.entities)

        # Simulate physics steps until landed
        for _ in range(50):
            self.world.update_entities(0.05)
            if not falling.alive:
                break

        self.assertFalse(falling.alive)
        # Should have landed on top of stone at y=3
        self.assertEqual(self.world.get_block(4, 3, 4), "sand")

    def test_stepped_on_triggers_fall(self):
        # Create floating sand directly in chunk data (simulating initial worldgen)
        # Sand at (8, 6, 8) with air underneath at (8, 5, 8)
        self.chunk.data[6, 8, 8] = block_index("sand")
        self.assertEqual(self.world.get_block(8, 6, 8), "sand")
        self.assertEqual(len(self.world.entities), 0)

        # Player steps onto the floating sand (pos y = 7.0, standing on y = 6.0)
        player = Player(self.world)
        player.pos = [8.5, 7.0, 8.5]
        player.vel = [0.0, 0.0, 0.0]
        player.on_ground = True

        # Update player
        player.update(0.05)

        # Stepping on the floating sand should have triggered it to fall!
        self.assertEqual(self.world.get_block(8, 6, 8), "air")
        self.assertEqual(len(self.world.entities), 1)
        self.assertIsInstance(self.world.entities[0], FallingBlockEntity)

    def test_clay_drops_clay_balls(self):
        drops = get_block_drops("clay")
        self.assertEqual(drops, [("clay_ball", 4)])

    def test_clay_ball_texture_and_item(self):
        tex_path = ASSETS / "textures" / "clay_ball.png"
        self.assertTrue(tex_path.is_file())
        img = read_png(tex_path)
        self.assertEqual(img.shape, (16, 16, 4))

        from server.items import ITEMS
        self.assertIn("clay_ball", ITEMS)
        clay_ball = ITEMS["clay_ball"]
        self.assertEqual(clay_ball.max_stack, 64)

    def test_clay_recipes(self):
        # 1. Crafting clay block from 4 clay balls (2x2)
        grid_2x2 = (
            "clay_ball", "clay_ball",
            "clay_ball", "clay_ball",
        )
        res = craft_match(grid_2x2)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "clay")
        self.assertEqual(res[1], 1)

        # 2. Crafting unfired clay bucket from 3 clay balls (V shape in 3x3)
        grid_3x3 = (
            "clay_ball", "", "clay_ball",
            "", "clay_ball", "",
            "", "", "",
        )
        res_bucket = craft_match(grid_3x3)
        self.assertIsNotNone(res_bucket)
        self.assertEqual(res_bucket[0], "unfired_clay_bucket")
        self.assertEqual(res_bucket[1], 1)

        # 3. Smelting unfired clay bucket yields clay bucket
        smelt_res = smelt_match("unfired_clay_bucket")
        self.assertIsNotNone(smelt_res)
        self.assertEqual(smelt_res[0], "clay_bucket")
        self.assertEqual(smelt_res[1], 1)

    def test_falling_block_near_player_no_crash(self):
        player = Player(self.world)
        player.pos = [5.0, 5.0, 5.0]
        # Spawn falling block right next to player
        falling = self.world.spawn_falling_block("sand", pos=[5.2, 5.0, 5.2])
        self.assertIn(falling, self.world.entities)

        # Emulate player entity pickup loop from app.py
        px = player.pos[0]
        py = player.pos[1] + 0.8
        pz = player.pos[2]
        for entity in list(self.world.entities):
            if not isinstance(entity, ItemEntity):
                continue
            if entity.pickup_delay <= 0 and entity.count > 0:
                pass
        # Ensure falling block was not picked up and has fallback attributes
        self.assertIn(falling, self.world.entities)
        self.assertEqual(falling.block_id, "sand")
        self.assertGreater(falling.pickup_delay, 1000)

    def _game_with_player(self, player):
        from client.app import Game

        game = Game.__new__(Game)
        game.world = self.world
        game.player = player
        return game

    def test_fallen_block_on_player_returns_block_id(self):
        player = Player(self.world)
        player.pos = [5.0, 5.0, 5.0]
        # Block at the player's head height occludes the view.
        falling = self.world.spawn_falling_block("sand", pos=[5.2, player.pos[1] + player.height - 0.8, 5.2])
        game = self._game_with_player(player)
        try:
            self.assertEqual(game._fallen_block_on_player(), "sand")
        finally:
            self.world.remove_entity(falling.id)

    def test_fallen_block_at_feet_does_not_block_view(self):
        player = Player(self.world)
        player.pos = [5.0, 5.0, 5.0]
        # A block resting on the player's feet does not occlude the view.
        falling = self.world.spawn_falling_block("sand", pos=[5.2, player.pos[1] + 0.2, 5.2])
        game = self._game_with_player(player)
        try:
            self.assertIsNone(game._fallen_block_on_player())
        finally:
            self.world.remove_entity(falling.id)

    def test_no_fallen_block_on_player_returns_none(self):
        player = Player(self.world)
        player.pos = [5.0, 5.0, 5.0]
        game = self._game_with_player(player)
        self.assertIsNone(game._fallen_block_on_player())

    def test_fallen_block_above_player_does_not_block_view(self):
        player = Player(self.world)
        player.pos = [5.0, 5.0, 5.0]
        # Block resting on top of the player (just above the head) is not "inside".
        falling = self.world.spawn_falling_block("sand", pos=[5.2, player.pos[1] + player.height, 5.2])
        game = self._game_with_player(player)
        try:
            self.assertIsNone(game._fallen_block_on_player())
        finally:
            self.world.remove_entity(falling.id)

    def test_falling_block_origin_removal_is_broadcast(self):
        changes = []
        self.world.on_block_set = lambda x, y, z, b: changes.append((x, y, z, b))

        self.world.set_block(2, 5, 2, "sand")

        # The server clears the origin when the sand starts falling...
        self.assertEqual(self.world.get_block(2, 5, 2), AIR)
        # ...and clients must be told, otherwise they keep a phantom block.
        self.assertIn((2, 5, 2, AIR), changes)

    def test_non_simulating_world_does_not_spawn_falling_blocks(self):
        # Multiplayer clients apply server block changes without running the
        # world simulation locally.
        world = World(seed=1)
        world.simulate_blocks = False
        chunk = Chunk(0, 0, 0)
        chunk.data.fill(0)
        world.chunks[(0, 0, 0)] = chunk

        self.assertTrue(world.set_block(2, 5, 2, "sand"))

        self.assertEqual(world.get_block(2, 5, 2), "sand")
        self.assertEqual(world.entities, [])

    def test_non_simulating_falling_block_does_not_place_block(self):
        world = World(seed=1)
        world.simulate_blocks = False
        chunk = Chunk(0, 0, 0)
        chunk.data.fill(0)
        world.chunks[(0, 0, 0)] = chunk
        world.set_block(2, 1, 2, "stone")

        world.spawn_falling_block("sand", pos=[2.5, 5.0, 2.5])
        for _ in range(200):
            world.update_entities(0.02)

        # The server broadcasts the landing; the client must not invent it.
        self.assertEqual(world.get_block(2, 2, 2), AIR)
        self.assertEqual(world.entities, [])

    def test_falling_block_stays_in_sync_between_server_and_clients(self):
        def make_world():
            world = World(seed=1)
            chunk = Chunk(0, 0, 0)
            chunk.data.fill(0)
            chunk.data[1, 2, 2] = block_index("stone")  # floor at y=1
            world.chunks[(0, 0, 0)] = chunk
            return world

        server = make_world()
        changes = []
        server.on_block_set = lambda x, y, z, b: changes.append((x, y, z, b))
        clients = [make_world(), make_world()]
        for world in clients:
            world.simulate_blocks = False

        def apply_changes():
            for x, y, z, block in changes:
                for world in clients:
                    world.set_block(x, y, z, block)
            changes.clear()

        server.set_block(2, 5, 2, "sand")
        apply_changes()

        known = set()
        for _ in range(300):
            server.update_entities(0.02)
            current = {e.id: e for e in server.entities}
            for eid, entity in current.items():
                if eid not in known and getattr(entity, "is_falling_block", False):
                    known.add(eid)
                    data = {
                        "id": eid,
                        "block": entity.block_id,
                        "pos": entity.pos,
                        "vel": entity.vel,
                        "time": entity.time_falling,
                    }
                    for world in clients:
                        world.entities.append(FallingBlockEntity.from_dict(data))
            for eid in list(known):
                if eid not in current:
                    known.discard(eid)
                    for world in clients:
                        world.remove_entity(eid)
            apply_changes()
            for world in clients:
                world.update_entities(0.02)

        self.assertEqual(server.get_block(2, 5, 2), AIR)
        self.assertEqual(server.get_block(2, 2, 2), "sand")
        for world in clients:
            self.assertEqual(world.get_block(2, 5, 2), AIR)
            self.assertEqual(world.get_block(2, 2, 2), "sand")

