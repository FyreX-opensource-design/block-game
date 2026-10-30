import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from server.biomes import BIOMES, biome_at, load_biomes
from server.blocks import GRASS, LOG, STONE, STONE_BASE, WORLD_HEIGHT, WATER_LEVEL
from server.noise import fbm, terrain_height, terrain_grid
from server.world import World


class MountainBiomeTests(unittest.TestCase):
    def test_mountain_biomes_registered(self):
        biome_map = {b.id: b for b in BIOMES}
        self.assertIn("mountains", biome_map)
        self.assertIn("wooded_mountains", biome_map)
        self.assertIn("mountain_meadow", biome_map)
        self.assertIn("mountain_peaks", biome_map)
        self.assertIn("mountain_cliffs", biome_map)

        mtn = biome_map["mountains"]
        wooded = biome_map["wooded_mountains"]
        meadow = biome_map["mountain_meadow"]
        peaks = biome_map["mountain_peaks"]
        cliffs = biome_map["mountain_cliffs"]

        # Check height offsets and scales are tiered
        self.assertGreater(mtn.height_offset, 0)
        self.assertGreater(wooded.height_offset, mtn.height_offset)
        self.assertGreater(meadow.height_offset, wooded.height_offset)
        self.assertGreater(peaks.height_offset, meadow.height_offset)
        self.assertGreater(cliffs.height_offset, peaks.height_offset)

        self.assertGreater(wooded.height_scale, mtn.height_scale)
        self.assertGreater(meadow.height_scale, wooded.height_scale)
        self.assertGreater(peaks.height_scale, meadow.height_scale)

        # Check smooth vs cliff flags
        self.assertFalse(mtn.cliff, "Mountains should be smooth in generation")
        self.assertFalse(wooded.cliff, "Wooded mountains should be smooth in generation")
        self.assertFalse(meadow.cliff, "Mountain meadow should be smooth in generation")
        self.assertFalse(peaks.cliff, "Mountain peaks should be smooth in generation")
        self.assertTrue(cliffs.cliff, "Mountain cliffs should make sheer cliffs")
        self.assertGreater(cliffs.cliff_step, 0.0)

        # Check surfaces
        self.assertEqual(mtn.surface, GRASS)
        self.assertEqual(wooded.surface, GRASS)
        self.assertEqual(meadow.surface, GRASS)
        self.assertEqual(peaks.surface, STONE)
        self.assertEqual(cliffs.surface, STONE)

        # Check tree cover
        self.assertTrue(mtn.trees)
        self.assertTrue(wooded.trees)
        self.assertGreater(wooded.tree_chance, mtn.tree_chance)
        self.assertTrue(meadow.trees)
        self.assertEqual(peaks.trees, ())
        self.assertEqual(peaks.tree_chance, 0.0)
        self.assertEqual(cliffs.trees, ())
        self.assertEqual(cliffs.tree_chance, 0.0)

    def test_expanded_biome_asset_definition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            biomes_dir = root / "biomes"
            biomes_dir.mkdir()

            (biomes_dir / "custom_cliff.json").write_text(
                json.dumps(
                    {
                        "id": "custom_cliff",
                        "surface": "stone",
                        "soil": "stone",
                        "soil_depth": 1,
                        "scale": 0.006,
                        "octaves": 2,
                        "seed_offset": 90,
                        "ceiling": 1.0,
                        "tree_chance": 0.0,
                        "height_offset": 40.0,
                        "height_scale": 1.8,
                        "cliff": True,
                        "cliff_step": 12.0,
                        "blend": 0.01,
                        "trees": [],
                        "decorations": [],
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_biomes(biomes_dir, trees=(), decorations=())
            self.assertEqual(len(loaded), 1)
            b = loaded[0]
            self.assertEqual(b.id, "custom_cliff")
            self.assertEqual(b.height_offset, 40.0)
            self.assertEqual(b.height_scale, 1.8)
            self.assertTrue(b.cliff)
            self.assertEqual(b.cliff_step, 12.0)
            self.assertEqual(b.blend, 0.01)

    def test_mountain_heights_elevated_and_within_world_height(self):
        # Sample coordinates across the world
        mtn_heights = []
        cliff_heights = []
        plains_heights = []

        for z in range(-300, 301, 10):
            for x in range(-300, 301, 10):
                b = biome_at(x, z, 1, BIOMES)
                h = terrain_height(x, z, 1)
                # All terrain must remain safely below WORLD_HEIGHT
                self.assertLess(h, WORLD_HEIGHT - 3)
                self.assertGreaterEqual(h, 50)

                if b.id == "plains":
                    plains_heights.append(h)
                elif b.id in ("mountains", "wooded_mountains", "mountain_meadow", "mountain_peaks"):
                    mtn_heights.append(h)
                elif b.id == "mountain_cliffs":
                    cliff_heights.append(h)

        self.assertTrue(mtn_heights)
        self.assertTrue(cliff_heights)
        self.assertTrue(plains_heights)

        # Mountains and cliffs must be significantly higher than plains
        avg_plains = np.mean(plains_heights)
        avg_mtn = np.mean(mtn_heights)
        avg_cliff = np.mean(cliff_heights)

        self.assertGreater(avg_mtn, avg_plains + 10.0)
        self.assertGreater(avg_cliff, avg_mtn + 5.0)

    def test_smooth_mountain_transitions(self):
        # Sampling across boundaries into smooth mountains should have low slope steps
        biome_map = {b.id: b for b in BIOMES}
        mtn_ceil = biome_map["mountains"].ceiling
        forest_ceil = biome_map["forest"].ceiling

        # Find transitions from forest/pine_forest to mountains
        found_transition = False
        for z in range(-200, 201, 15):
            for x in range(-200, 200):
                b1 = biome_at(x, z, 1, BIOMES)
                b2 = biome_at(x + 1, z, 1, BIOMES)
                if b1.id in ("forest", "pine_forest") and b2.id == "mountains":
                    found_transition = True
                    # Check local step size along 10 blocks across the transition
                    row = [terrain_height(px, z, 1) for px in range(x - 5, x + 6)]
                    steps = [abs(row[i + 1] - row[i]) for i in range(len(row) - 1)]
                    # Smooth transition: each block step should not exceed 3 blocks per meter
                    self.assertLessEqual(max(steps), 3, f"Smooth mountain transition too steep: {steps}")
                    break
            if found_transition:
                break
        self.assertTrue(found_transition, "Should find a forest-to-mountains transition")

    def test_sheer_cliffs_transition(self):
        # Mountain cliffs should create dramatic sheer vertical steps
        found_cliff = False
        max_cliff_step = 0

        for z in range(-300, 301, 8):
            for x in range(-300, 300):
                b1 = biome_at(x, z, 1, BIOMES)
                b2 = biome_at(x + 1, z, 1, BIOMES)
                if (b1.id != "mountain_cliffs" and b2.id == "mountain_cliffs") or (
                    b1.id == "mountain_cliffs" and b2.id != "mountain_cliffs"
                ):
                    # Measure elevation change over a short 2-block window
                    h_before = terrain_height(x - 1, z, 1)
                    h_after = terrain_height(x + 2, z, 1)
                    step = abs(h_after - h_before)
                    if step > max_cliff_step:
                        max_cliff_step = step
                    if step >= 6:
                        found_cliff = True
                        break
            if found_cliff:
                break

        self.assertTrue(found_cliff, f"Should produce sheer cliffs (max step found: {max_cliff_step})")

    def test_world_generation_in_cliffs_and_peaks(self):
        world = World(seed=1, caves=[])
        # Find coordinates in mountain_cliffs and generate column
        found = False
        for z in range(-250, 251, 16):
            for x in range(-250, 251, 16):
                b = biome_at(x, z, 1, BIOMES)
                if b.id == "mountain_cliffs":
                    cx = x // 16
                    cz = z // 16
                    world.generate_column(cx, cz)
                    h = terrain_height(x, z, 1)
                    # Surface block on mountain cliff should be stone
                    surf = world.get_block(x, h, z)
                    self.assertEqual(surf, STONE)
                    # Block beneath surface should be stone
                    below = world.get_block(x, h - 1, z)
                    self.assertEqual(below, STONE)
                    found = True
                    break
            if found:
                break
        self.assertTrue(found)

    def test_trees_never_generate_on_stone(self):
        # Scan several columns across mountainous terrain and verify no tree sits on stone
        world = World(seed=2, caves=[])
        for cz in range(-2, 3):
            for cx in range(-2, 3):
                world.generate_column(cx, cz)
                for z in range(cz * 16, (cz + 1) * 16):
                    for x in range(cx * 16, (cx + 1) * 16):
                        for y in range(50, 120):
                            if world.get_block(x, y, z) in (LOG, "pine_log"):
                                under = world.get_block(x, y - 1, z)
                                self.assertNotEqual(
                                    under,
                                    STONE,
                                    f"Tree at ({x}, {y}, {z}) was generated directly on stone!",
                                )
                                self.assertIn(
                                    under,
                                    (GRASS, "dirt", LOG, "pine_log"),
                                    f"Tree at ({x}, {y}, {z}) must sit on grass, dirt, or trunk log, but found {under}",
                                )


if __name__ == "__main__":
    unittest.main()

