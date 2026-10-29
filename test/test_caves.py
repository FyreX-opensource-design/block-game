import json
import tempfile
import unittest
from pathlib import Path

from server.blocks import AIR, STONE
from server.caves import cave_from_mapping, load_caves
from server.noise import terrain_height
from server.world import World


class CaveTests(unittest.TestCase):
    def test_default_caves_open_stone_and_leave_the_floor(self):
        carved = World(seed=3)
        solid = World(seed=3, caves=[], ores=[])
        carved.generate_column(0, 0)
        solid.generate_column(0, 0)
        opened = 0
        for z in range(0, 16, 4):
            for x in range(0, 16, 4):
                height = terrain_height(x, z, 3)
                self.assertEqual(carved.get_block(x, 0, z), STONE)
                for y in range(8, max(9, height - 8)):
                    self.assertEqual(solid.get_block(x, y, z), STONE)
                    if carved.get_block(x, y, z) == AIR:
                        opened += 1
        self.assertGreater(opened, 0)
        mouths = 0
        for cz in range(-1, 2):
            for cx in range(-1, 2):
                carved.generate_column(cx, cz)
                for z in range(cz * 16, cz * 16 + 16):
                    for x in range(cx * 16, cx * 16 + 16):
                        height = terrain_height(x, z, 3)
                        if carved.get_block(x, height, z) == AIR:
                            mouths += 1
        self.assertGreater(mouths, 0)

    def test_json_and_lua_files_configure_caves(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "always.json").write_text(
                json.dumps(
                    {
                        "id": "always",
                        "style": "cavern",
                        "min_y": 4,
                        "max_y": 30,
                        "surface_margin": 0,
                        "threshold": -1,
                    }
                ),
                encoding="utf-8",
            )
            (root / "quiet.lua").write_text(
                "return { id = 'quiet', style = 'tunnel', enabled = false, thickness = 1 }\n",
                encoding="utf-8",
            )
            caves = load_caves(root)
            self.assertEqual([cave.id for cave in caves], ["always", "quiet"])
            self.assertEqual(caves[0].source.endswith("always.json"), True)
            self.assertFalse(caves[1].enabled)
            world = World(seed=1, caves=caves)
            world.generate_chunk(0, 1, 0)
            world.generate_chunk(0, 0, 0)
            self.assertEqual(world.get_block(2, 0, 2), STONE)
            self.assertEqual(world.get_block(2, 20, 2), AIR)

    def test_bad_cave_file_is_rejected(self):
        with self.assertRaises(ValueError):
            cave_from_mapping({"id": "Bad", "style": "worm"}, "cave")
