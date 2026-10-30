import math
import unittest

import numpy as np

from server.blocks import (
    AIR,
    CHUNK_SIZE,
    CLAY,
    GRAVEL,
    GRASS,
    LEAVES,
    LOG,
    SAND,
    SECTION_COUNT,
    STONE,
    STONE_BASE,
    WORLD_HEIGHT,
    block_index,
)
from server.mesh import build_chunk_mesh
from server.noise import terrain_height
from server.player import Player, look_direction
from server.world import Chunk, World, tree_at


def fill_area(world, cx, cz, radius=1):
    for z in range(cz - radius, cz + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            world.generate_column(x, z)


class WorldTests(unittest.TestCase):
    def test_columns_are_filled_up_to_the_surface(self):
        world = World()
        world.generate_column(0, 0)
        for z in range(0, CHUNK_SIZE, 3):
            for x in range(0, CHUNK_SIZE, 3):
                height = terrain_height(x, z)
                self.assertEqual(world.get_block(x, 0, z), STONE)
                self.assertTrue(world.is_solid(x, height, z) or world.get_block(x, height, z) == AIR)
                self.assertFalse(world.is_solid(x, height + 12, z))

    def test_border_trees_include_a_trunk(self):
        found = None
        for z in range(64):
            for x in range(12, 20):
                if not tree_at(x, z):
                    continue
                world = World()
                cx = x // CHUNK_SIZE
                cz = z // CHUNK_SIZE
                world.generate_column(cx, cz)
                world.generate_column(cx + 1, cz)
                ground = terrain_height(x, z)
                if world.get_block(x, ground + 1, z) == LOG:
                    found = (x, z, ground)
                    break
            if found:
                break
        self.assertIsNotNone(found)

    def test_isolated_block_has_six_outward_faces(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        chunk.data[3, 2, 4] = block_index(STONE)
        world.chunks[(0, 0, 0)] = chunk
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh.indices), 36)
        triangles = mesh.vertices[mesh.indices].reshape(-1, 3, 9)
        geometric = np.cross(triangles[:, 1, :3] - triangles[:, 0, :3], triangles[:, 2, :3] - triangles[:, 0, :3])
        dots = np.sum(geometric * triangles[:, 0, 3:6], axis=1)
        self.assertTrue(np.all(dots > 0))

    def test_shared_faces_are_hidden(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        chunk.data[3, 2, 4] = block_index(STONE)
        chunk.data[3, 2, 5] = block_index(STONE)
        world.chunks[(0, 0, 0)] = chunk
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertEqual(len(mesh.indices), 60)
        self.assertIsNone(mesh.cutout)

    def test_cutout_leaves_keep_a_face_across_a_gap_and_against_stone(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        leaves = block_index(LEAVES)
        chunk.data[3, 2, 4] = leaves
        chunk.data[3, 2, 6] = leaves
        world.chunks[(0, 0, 0)] = chunk
        separated = build_chunk_mesh(world, 0, 0, 0)
        self.assertEqual(len(separated.indices), 0)
        self.assertEqual(len(separated.cutout[1]), 72)
        chunk.data[3, 2, 5] = leaves
        touching = build_chunk_mesh(world, 0, 0, 0)
        self.assertEqual(len(touching.indices), 0)
        self.assertEqual(len(touching.cutout[1]), 96)
        chunk.data[3, 2, 5] = block_index(STONE)
        blocked = build_chunk_mesh(world, 0, 0, 0)
        self.assertEqual(len(blocked.indices), 36)
        self.assertEqual(len(blocked.cutout[1]), 60)

    def test_downward_ray_hits_the_top_face(self):
        world = World()
        world.generate_column(0, 0)
        top = world.column_top(2, 2)
        hit = world.raycast(2.5, top + 2.2, 2.5, 0, -1, 0, 8)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["x"], 2)
        self.assertEqual(hit["z"], 2)
        self.assertEqual(hit["ny"], 1)
        self.assertEqual(hit["y"], top)

    def test_valleys_dip_below_the_stone_base(self):
        heights = [terrain_height(x, z, 1) for x in range(-120, 121, 3) for z in range(-120, 121, 3)]
        below = [height for height in heights if height < STONE_BASE]
        above = [height for height in heights if height > STONE_BASE + 4]
        self.assertTrue(below)
        self.assertTrue(above)
        self.assertGreaterEqual(min(below), 50)
        self.assertLessEqual(min(below), 60)
        self.assertLessEqual(max(below), STONE_BASE - 1)
        low = next((x, z) for x in range(-120, 121, 3) for z in range(-120, 121, 3) if terrain_height(x, z, 1) < STONE_BASE)
        high = next((x, z) for x in range(-120, 121, 3) for z in range(-120, 121, 3) if terrain_height(x, z, 1) > STONE_BASE + 4)
        world = World(seed=1, caves=[])
        world.generate_column(low[0] // CHUNK_SIZE, low[1] // CHUNK_SIZE)
        world.generate_column(high[0] // CHUNK_SIZE, high[1] // CHUNK_SIZE)
        low_height = terrain_height(*low, 1)
        high_height = terrain_height(*high, 1)
        self.assertEqual(world.get_block(low[0], 0, low[1]), STONE)
        self.assertNotEqual(world.get_block(low[0], STONE_BASE, low[1]), STONE)
        self.assertIn(world.get_block(low[0], low_height, low[1]), (GRASS, SAND, GRAVEL))
        self.assertEqual(world.get_block(high[0], STONE_BASE, high[1]), STONE)
        self.assertIn(world.get_block(high[0], high_height, high[1]), (GRASS, SAND, GRAVEL))
        self.assertFalse(world.set_block(low[0], 0, low[1], AIR))
        self.assertTrue(world.blocks_movement(low[0], -1, low[1]))
        self.assertFalse(world.blocks_movement(low[0], WORLD_HEIGHT, low[1]))
        self.assertEqual(SECTION_COUNT, WORLD_HEIGHT // CHUNK_SIZE)


class PlayerTests(unittest.TestCase):
    def test_player_stands_on_the_ground(self):
        world = World()
        fill_area(world, 0, 0, 1)
        player = Player(world)
        player.spawn(3, 3)
        ground = world.column_top(3, 3)
        for _ in range(180):
            player.update(1 / 60)
        self.assertTrue(player.on_ground)
        self.assertLess(abs(player.pos[1] - (ground + 1)), 0.05)

    def test_forward_moves_along_the_look_direction(self):
        self.assertAlmostEqual(look_direction(0, 0)[2], -1)
        world = World()
        fill_area(world, 0, 0, 1)
        player = Player(world)
        player.spawn(8, 8)
        start = player.pos[2]
        player.input.forward = True
        for _ in range(40):
            player.update(1 / 60)
        self.assertLess(player.pos[2], start - 1.0)
        self.assertFalse(player.collides_at(*player.pos))

    def test_player_steps_up_one_block(self):
        world = World()
        fill_area(world, 0, 0, 1)
        player = Player(world)
        player.spawn(8, 8)
        feet = player.pos[1]
        x = math.floor(player.pos[0])
        ahead = math.floor(player.pos[2]) - 2
        for z in range(ahead, ahead - 3, -1):
            for ox in (-1, 0, 1):
                world.set_block(x + ox, math.floor(feet), z, STONE)
        player.input.forward = True
        peak = feet
        for _ in range(45):
            player.update(1 / 60)
            peak = max(peak, player.pos[1])
        self.assertGreater(peak, feet + 0.8)
        self.assertGreater(player.pos[1], feet + 0.8)
        self.assertFalse(player.collides_at(*player.pos))


if __name__ == "__main__":
    unittest.main()
