import unittest
import numpy as np

from server.biomes import (
    BIOMES,
    all_biomes,
    biome_at,
    biome_index_grid,
    can_biomes_touch,
    climate_at,
    temperature_at,
    temperature_grid,
)
from server.blocks import SAND, WATER_LEVEL
from server.rivers import (
    DESERT_RIVER_BIOME,
    RIVER_BIOME,
    SAVANNA_RIVER_BIOME,
    river_biome_at,
)
from server.world import World


class TemperatureSystemTests(unittest.TestCase):
    def test_temperature_field_bounds_and_consistency(self):
        seed = 12345
        xx, zz = np.meshgrid(np.arange(-100, 101, 10), np.arange(-100, 101, 10))
        grid = temperature_grid(xx, zz, seed)
        self.assertTrue(np.all(grid >= 0.0), "Temperature must be >= 0.0")
        self.assertTrue(np.all(grid <= 1.0), "Temperature must be <= 1.0")

        # Grid and scalar results must match exactly
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                scalar = temperature_at(xx[i, j], zz[i, j], seed)
                self.assertAlmostEqual(float(grid[i, j]), scalar, places=5)

    def test_temperature_field_is_smooth_and_continuous(self):
        # Step delta between adjacent blocks must be small (< 0.01 per block)
        seed = 9999
        max_delta = 0.0
        for z in range(-80, 81, 8):
            for x in range(-80, 81, 8):
                t0 = temperature_at(x, z, seed)
                t_x = temperature_at(x + 1, z, seed)
                t_z = temperature_at(x, z + 1, seed)
                max_delta = max(max_delta, abs(t_x - t0), abs(t_z - t0))
        self.assertLess(max_delta, 0.01, f"Max single-block delta {max_delta} indicates a discontinuity")

    def test_biome_temperature_ranges(self):
        catalog = {b.id: b for b in all_biomes(BIOMES)}
        self.assertIn("savanna", catalog)
        self.assertIn("desert", catalog)
        self.assertIn("savanna_hills", catalog)
        self.assertIn("desert_hills", catalog)

        plains = catalog["plains"]
        savanna = catalog["savanna"]
        desert = catalog["desert"]

        self.assertEqual(plains.temperature, (0.0, 0.50))
        self.assertEqual(savanna.temperature, (0.50, 0.75))
        self.assertEqual(desert.temperature, (0.75, 1.0))

        highland = catalog["highland_plains"]
        savanna_hills = catalog["savanna_hills"]
        desert_hills = catalog["desert_hills"]

        self.assertEqual(highland.temperature, (0.0, 0.50))
        self.assertEqual(savanna_hills.temperature, (0.50, 0.75))
        self.assertEqual(desert_hills.temperature, (0.75, 1.0))

        # River biomes
        self.assertEqual(RIVER_BIOME.temperature, (0.0, 0.50))
        self.assertEqual(SAVANNA_RIVER_BIOME.temperature, (0.50, 0.75))
        self.assertEqual(DESERT_RIVER_BIOME.temperature, (0.75, 1.0))

    def test_can_biomes_touch_compatibility(self):
        catalog = {b.id: b for b in all_biomes(BIOMES)}
        plains = catalog["plains"]
        savanna = catalog["savanna"]
        desert = catalog["desert"]

        # Desert cannot directly touch Plains
        self.assertFalse(can_biomes_touch(plains, desert))
        self.assertFalse(can_biomes_touch(desert, plains))

        # Savanna can touch both Plains and Desert
        self.assertTrue(can_biomes_touch(plains, savanna))
        self.assertTrue(can_biomes_touch(savanna, desert))

        # Hill counterparts
        highland = catalog["highland_plains"]
        savanna_hills = catalog["savanna_hills"]
        desert_hills = catalog["desert_hills"]
        self.assertFalse(can_biomes_touch(highland, desert_hills))
        self.assertTrue(can_biomes_touch(highland, savanna_hills))
        self.assertTrue(can_biomes_touch(savanna_hills, desert_hills))

    def test_world_adjacency_plains_to_desert_always_passes_through_savanna(self):
        # Walk along lines across multiple seeds; any transition from plains to desert must touch savanna
        found_any_desert = False
        for seed in (42, 100, 777, 1234):
            for z in range(-800, 801, 50):
                line_biomes = []
                for x in range(-800, 801, 4):
                    c = climate_at(x, z, seed, BIOMES)
                    if 0.43 < c <= 0.49: # plains climate band
                        b = biome_at(x, z, seed, BIOMES)
                        line_biomes.append(b.id)
                        if b.id == "desert":
                            found_any_desert = True
                    elif 0.49 < c <= 0.54: # hills climate band
                        b = biome_at(x, z, seed, BIOMES)
                        line_biomes.append(b.id)
                        if b.id == "desert_hills":
                            found_any_desert = True
                    else:
                        line_biomes.append(None)

                # Verify no adjacent step directly transitions from plains to desert or vice versa
                for i in range(len(line_biomes) - 1):
                    b1 = line_biomes[i]
                    b2 = line_biomes[i + 1]
                    if b1 is None or b2 is None:
                        continue
                    if b1 in ("plains", "rose_thicket", "wildflower_patch") and b2 == "desert":
                        self.fail(f"Plains ({b1}) directly touches Desert at seed {seed}, z {z}, index {i}")
                    if b1 == "desert" and b2 in ("plains", "rose_thicket", "wildflower_patch"):
                        self.fail(f"Desert directly touches Plains ({b2}) at seed {seed}, z {z}, index {i}")

                    # Same for hills
                    if b1 in ("highland_plains", "lavender_hollow", "orchid_clearing") and b2 == "desert_hills":
                        self.fail(f"Highland Plains ({b1}) directly touches Desert Hills at seed {seed}, z {z}, index {i}")
                    if b1 == "desert_hills" and b2 in ("highland_plains", "lavender_hollow", "orchid_clearing"):
                        self.fail(f"Desert Hills directly touches Highland Plains ({b2}) at seed {seed}, z {z}, index {i}")

        self.assertTrue(found_any_desert, "Should generate desert somewhere in test area across seeds")

    def test_matching_river_biomes_by_temperature(self):
        seed = 42
        found_temperate_river = False
        found_savanna_river = False
        found_desert_river = False

        for z in range(-600, 601, 25):
            for x in range(-600, 601, 25):
                rb = river_biome_at(x, WATER_LEVEL, z, seed, BIOMES)
                if rb is not None and rb.id != "estuary":
                    t = temperature_at(x, z, seed)
                    if t >= 0.75:
                        self.assertEqual(rb.id, "desert_river")
                        found_desert_river = True
                    elif t >= 0.50:
                        self.assertEqual(rb.id, "savanna_river")
                        found_savanna_river = True
                    else:
                        self.assertEqual(rb.id, "river")
                        found_temperate_river = True

        self.assertTrue(found_temperate_river, "Should find temperate river")
        self.assertTrue(found_savanna_river, "Should find savanna river")
        self.assertTrue(found_desert_river, "Should find desert river")

    def test_desert_chunk_generates_sand_surface_and_sandstone_soil(self):
        seed = 42
        # Find a dry, terrestrial desert coordinate (not submerged in a river)
        desert_pt = None
        from server.rivers import is_river
        for z in range(-800, 801, 20):
            for x in range(-800, 801, 20):
                c = climate_at(x, z, seed, BIOMES)
                if 0.43 < c <= 0.49 and temperature_at(x, z, seed) >= 0.75:
                    if not is_river(x, z, seed, climate=c):
                        b = biome_at(x, z, seed, BIOMES)
                        if b.id == "desert":
                            desert_pt = (x, z)
                            break
            if desert_pt:
                break

        self.assertIsNotNone(desert_pt, "Must find a desert location")
        dx, dz = desert_pt
        world = World(seed=seed, caves=[], biomes=BIOMES)
        cx, cz = dx // 16, dz // 16
        world.load_chunk(cx, 0, cz)
        world.generate_column(cx, cz)

        # Inspect surface and soil
        surface_y = None
        for y in range(120, 50, -1):
            if world.get_block(dx, y, dz) != "air":
                surface_y = y
                break

        self.assertIsNotNone(surface_y)
        self.assertEqual(world.get_block(dx, surface_y, dz), SAND)
        self.assertEqual(world.get_block(dx, surface_y - 1, dz), "sand")
        self.assertEqual(world.get_block(dx, surface_y - 4, dz), "sandstone")
