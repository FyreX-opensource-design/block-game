import unittest

import numpy as np

from server.blocks import BLOCKS, CREATIVE, can_place_block, combine_vine, placement_block
from server.items import ITEMS
from server.lighting import fluorescence_response, parse_fluorescence
from server.mesh import build_chunk_mesh
from server.world import World


class FluorescenceTests(unittest.TestCase):
    def test_definition_sets_the_color_and_the_uv_window(self):
        lichen = BLOCKS["fluor_lichen"].fluorescence
        amber = BLOCKS["fluor_amber"].fluorescence
        sand = BLOCKS["lichen_sand"].fluorescence
        self.assertIsNone(BLOCKS["stone"].fluorescence)
        self.assertNotIn("fluor_dust", BLOCKS)
        self.assertNotIn("uv_lamp", BLOCKS)
        self.assertNotIn("dim_uv_lamp", BLOCKS)
        self.assertEqual((lichen.start, lichen.full, lichen.strength), (1.0, 5.0, 1.0))
        self.assertEqual(lichen.texture, "fluor_lichen_glow")
        self.assertEqual(amber.texture, "fluor_amber_glow")
        self.assertEqual((amber.start, amber.full, amber.strength), (4.0, 10.0, 1.0))
        self.assertEqual(sand.strength, 0.5)
        self.assertIsNotNone(BLOCKS["fluor_amber"].fluor_face_uvs)
        self.assertEqual(BLOCKS["lichen_lamp"].emission[3], 15)
        self.assertEqual(BLOCKS["lichen_torch"].emission[3], 8)
        lamp = BLOCKS["lichen_lamp"].emission
        self.assertGreater(lamp[2], lamp[1])
        self.assertGreater(lamp[0], lamp[1])
        self.assertGreater(lamp[2], 8)
        for block_id in ("lichen_lamp", "lichen_torch", "fluor_lichen", "fluor_amber"):
            self.assertIn(block_id, CREATIVE)
            self.assertEqual(ITEMS[block_id].block, block_id)

        # A lichen torch (UV 8) saturates the plant. Sap needs a stronger lamp.
        self.assertAlmostEqual(fluorescence_response(8, lichen), 1.0)
        self.assertGreater(fluorescence_response(8, amber), 0.0)
        self.assertAlmostEqual(fluorescence_response(15, amber), 1.0)
        self.assertAlmostEqual(fluorescence_response(15, sand), 0.5)
        self.assertEqual(fluorescence_response(5, sand), 0.0)

    def test_color_and_window_can_be_written_loosely(self):
        spec = parse_fluorescence({"color": "#39ff6a", "start": 12, "full": 2, "strength": 0.45})
        self.assertEqual((spec.start, spec.full), (2.0, 12.0))
        self.assertAlmostEqual(spec.strength, 7.0 / 16.0)
        self.assertAlmostEqual(spec.color[0], 0x39 / 255.0)
        self.assertAlmostEqual(fluorescence_response(12, spec), spec.strength)
        self.assertEqual(fluorescence_response(1, spec), 0.0)
        with self.assertRaises(ValueError):
            parse_fluorescence({"start": 1, "full": 4})
        self.assertIsNone(parse_fluorescence({"color": [1, 1, 1], "strength": 0}))

    def test_fluorescent_faces_carry_the_response_and_stone_does_not(self):
        world = World(seed=11, caves=[])
        world.load_chunk(0, 0, 0)
        x = z = 3
        y = 10
        for yy in range(y - 1, y + 2):
            for zz in range(z - 1, z + 2):
                for xx in range(x - 1, x + 2):
                    world.set_block(xx, yy, zz, "air")
        world.set_block(x, y, z, "fluor_amber")
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertEqual(mesh.fluorescence.shape, (len(mesh.vertices), 4))
        self.assertEqual(mesh.fluor_uvs.shape, (len(mesh.vertices), 2))
        pos = mesh.vertices[:, 0:3]
        mask = (
            (pos[:, 0] >= x) & (pos[:, 0] <= x + 1)
            & (pos[:, 1] >= y) & (pos[:, 1] <= y + 1)
            & (pos[:, 2] >= z) & (pos[:, 2] <= z + 1)
        )
        self.assertGreaterEqual(int(mask.sum()), 24)
        packed = BLOCKS["fluor_amber"].fluorescence.packed
        self.assertTrue(np.allclose(mesh.fluorescence[mask], packed))
        self.assertTrue(np.all(mesh.fluorescence[~mask] == 0))
        self.assertTrue(np.all(mesh.fluor_uvs[mask, 0] >= 0))
        self.assertTrue(np.all(mesh.fluor_uvs[~mask, 0] < 0))

    def test_uv_lamps_light_with_uv_rather_than_visible_color(self):
        world = World(seed=12, caves=[])
        world.load_chunk(0, 0, 0)
        for yy in range(9, 12):
            for zz in range(3, 8):
                for xx in range(3, 6):
                    world.set_block(xx, yy, zz, "air")
        world.set_block(4, 10, 4, "lichen_lamp")
        world.set_block(4, 10, 6, "lichen_torch")
        bright = world.get_blocklight(4, 10, 5)
        dim = world.get_blocklight(4, 10, 7)
        self.assertGreater(bright[3], 10)
        self.assertGreater(bright[3], bright[0])
        self.assertGreater(bright[3], bright[2])
        self.assertGreater(dim[3], 0)
        self.assertLess(dim[3], bright[3])
        self.assertGreater(bright[2], bright[1])

    def test_lichen_hangs_on_a_wall_and_falls_off_without_it(self):
        self.assertEqual(placement_block("fluor_lichen", (1, 0, 0)), "fluor_lichen_v1")
        self.assertFalse(BLOCKS["fluor_lichen_v1"].solid)
        world = World(seed=13, caves=[])
        world.load_chunk(0, 0, 0)
        for pos in ((4, 10, 4), (5, 10, 4), (5, 10, 3), (6, 10, 4)):
            world.set_block(*pos, "air")
        world.set_block(4, 10, 4, "stone")
        world.set_block(5, 10, 3, "stone")
        self.assertTrue(can_place_block(world, 5, 10, 4, "fluor_lichen_v1"))
        self.assertFalse(can_place_block(world, 6, 10, 4, "fluor_lichen_v1"))
        world.set_block(5, 10, 4, "fluor_lichen_v1")
        merged = combine_vine(world.get_block(5, 10, 4), placement_block("fluor_lichen", (0, 0, 1)))
        self.assertEqual(merged, "fluor_lichen_v5")
        world.set_block(5, 10, 4, merged)
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh.cutout_fluor_uvs)
        pos = mesh.cutout[0][:, 0:3]
        on_vine = (pos[:, 0] >= 5) & (pos[:, 0] <= 6) & (pos[:, 1] >= 10) & (pos[:, 1] <= 11) & (pos[:, 2] >= 4) & (pos[:, 2] <= 5)
        self.assertGreater(int(on_vine.sum()), 0)
        self.assertTrue(np.any(mesh.cutout_fluor_uvs[on_vine, 0] >= 0))
        self.assertTrue(np.any(mesh.cutout_fluorescence[on_vine, 3] > 0))
        world.set_block(4, 10, 4, "air")
        self.assertEqual(world.get_block(5, 10, 4), "fluor_lichen_v4")
        world.set_block(5, 10, 3, "air")
        self.assertEqual(world.get_block(5, 10, 4), "air")
