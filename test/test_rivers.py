import unittest
import numpy as np

from server.biomes import BIOMES, biome_at, climate_at, climate_grid
from server.blocks import (
    BRACKISH_WATER,
    CLAY,
    GRASS,
    GRAVEL,
    MUD,
    MUDDY_GRASS,
    SALT_WATER,
    SAND,
    WATER,
    WATER_LEVEL,
    block_index,
)
from server.noise import terrain_grid, terrain_height
from server.rivers import (
    BASE_CHANNEL_W,
    BASE_VALLEY_W,
    carve_river_grid,
    carve_river_scalar,
    is_river,
    is_river_mouth,
    river_biome_at,
    river_water_type,
    river_water_type_grid,
)
from server.world import World


class RiverTests(unittest.TestCase):
    def test_rivers_wind_from_high_elevation_to_ocean(self):
        # Scan a region to find river points across all elevation zones
        seed = 42
        climates_in_river = []
        biomes_in_river = set()

        for z in range(-300, 301, 8):
            for x in range(-300, 301, 8):
                c = climate_at(x, z, seed, BIOMES)
                if is_river(x, z, seed, climate=c):
                    climates_in_river.append(c)
                    b = biome_at(x, z, seed, BIOMES)
                    biomes_in_river.add(b.id)

        self.assertTrue(climates_in_river)
        # Rivers should span high elevation (climate > 0.65) to low/ocean (climate <= 0.36)
        min_c = min(climates_in_river)
        max_c = max(climates_in_river)
        self.assertLessEqual(min_c, 0.36, "River must reach ocean/coastal elevations")
        self.assertGreaterEqual(max_c, 0.65, "River must originate in high elevation biomes")

        # Must pass through mountainous, inland, and coastal biomes
        mountain_biomes = {"mountains", "wooded_mountains", "mountain_meadow", "mountain_peaks"}
        inland_biomes = {"plains", "lowland_plains", "forest", "highland_plains", "pine_forest"}
        coastal_biomes = {"beach", "ocean"}

        self.assertTrue(biomes_in_river & mountain_biomes, f"Missing mountain biomes in river: {biomes_in_river}")
        self.assertTrue(biomes_in_river & inland_biomes, f"Missing inland biomes in river: {biomes_in_river}")
        self.assertTrue(biomes_in_river & coastal_biomes, f"Missing coastal biomes in river: {biomes_in_river}")

    def test_brackish_water_at_ocean_meeting_points(self):
        seed = 42
        found_inland_freshwater = False
        found_brackish_meeting = False
        found_ocean_saltwater = False

        for z in range(-300, 301, 6):
            for x in range(-300, 301, 6):
                c = climate_at(x, z, seed, BIOMES)
                w_type = river_water_type(x, z, seed, c)
                if is_river(x, z, seed, climate=c):
                    if c > 0.36 and w_type == WATER:
                        found_inland_freshwater = True
                    elif 0.30 < c <= 0.36 and w_type == BRACKISH_WATER:
                        found_brackish_meeting = True
                elif c <= 0.30 and w_type == SALT_WATER:
                    found_ocean_saltwater = True

                if found_inland_freshwater and found_brackish_meeting and found_ocean_saltwater:
                    break
            if found_inland_freshwater and found_brackish_meeting and found_ocean_saltwater:
                break

        self.assertTrue(found_inland_freshwater, "Should have freshwater inland")
        self.assertTrue(found_brackish_meeting, "Should have brackish water at river-ocean meeting points")
        self.assertTrue(found_ocean_saltwater, "Should have salt water in open ocean")

    def test_river_carves_smooth_valley(self):
        # Step across adjacent blocks along river valley should not exceed 3 blocks per meter
        seed = 42
        max_step = 0
        for z in range(-100, 101, 10):
            row = [terrain_height(x, z, seed) for x in range(-100, 100)]
            for i in range(len(row) - 1):
                step = abs(row[i + 1] - row[i])
                if step > max_step:
                    max_step = step

        self.assertLessEqual(max_step, 3, f"River bank slope too steep: max step was {max_step}")

    def test_grid_and_scalar_carving_are_identical(self):
        seed = 42
        xs = np.arange(40, 56)
        zs = np.arange(80, 96)
        xx, zz = np.meshgrid(xs, zs)
        clim = climate_grid(xx, zz, seed, BIOMES)
        raw_h = np.full(xx.shape, 76.0)

        grid_h = carve_river_grid(xx, zz, raw_h, seed, clim)

        for i in range(16):
            for j in range(16):
                x = int(xx[i, j])
                z = int(zz[i, j])
                c = climate_at(x, z, seed, BIOMES)
                scalar_h = carve_river_scalar(x, z, 76.0, seed, c)
                self.assertEqual(
                    scalar_h,
                    grid_h[i, j],
                    f"Mismatch at ({x}, {z}): scalar={scalar_h}, grid={grid_h[i, j]}",
                )

    def test_chunk_generation_places_correct_water_types(self):
        # Generate chunks where river meets ocean and verify blocks in world
        seed = 42
        world = World(seed=seed, caves=[], ores=[])

        # Find a chunk with river-ocean meeting point (brackish water)
        found_brackish = False
        found_fresh = False

        for cz in range(-10, 11):
            for cx in range(-10, 11):
                chunk = world.load_chunk(cx, 4, cz)  # cy=4 is y=64
                if chunk is None:
                    continue
                block_ids = {chunk.data[y, z, x] for y in range(16) for z in range(16) for x in range(16)}
                if block_index(BRACKISH_WATER) in block_ids:
                    found_brackish = True
                if block_index(WATER) in block_ids:
                    found_fresh = True
                if found_brackish and found_fresh:
                    break
            if found_brackish and found_fresh:
                break

        self.assertTrue(found_fresh, "World should generate freshwater river blocks")
        self.assertTrue(found_brackish, "World should generate brackish water blocks at meeting points")

    def test_river_biome_detection(self):
        seed = 42
        # Find a river point and test river_biome_at
        found_river = False
        found_estuary = False
        for z in range(-200, 201, 10):
            for x in range(-200, 201, 10):
                c = climate_at(x, z, seed, BIOMES)
                if is_river(x, z, seed, climate=c):
                    rb = river_biome_at(x, WATER_LEVEL, z, seed, BIOMES)
                    if rb is not None:
                        if c > 0.36:
                            self.assertIn(rb.id, ("river", "savanna_river", "desert_river"))
                            found_river = True
                        elif 0.30 < c <= 0.36:
                            self.assertEqual(rb.id, "estuary")
                            found_estuary = True
                if found_river and found_estuary:
                    break
            if found_river and found_estuary:
                break

        self.assertTrue(found_river, "Should detect river biome in inland channel")
        self.assertTrue(found_estuary, "Should detect estuary biome at river mouth")

    def test_river_extends_inland_with_water(self):
        # Verify river channels maintain water (terrain_height <= 61, water up to 64) deep inland
        seed = 42
        world = World(seed=seed, caves=[], ores=[])
        inland_river_points = []

        for z in range(-300, 301, 8):
            for x in range(-300, 301, 8):
                c = climate_at(x, z, seed, BIOMES)
                # Inland in plains, forests, and mountains
                if c >= 0.45 and is_river(x, z, seed, climate=c):
                    h = terrain_height(x, z, seed)
                    inland_river_points.append((x, z, c, h))

        self.assertTrue(len(inland_river_points) >= 10, "Should find multiple river points extending deep inland")
        # Every inland river point must have its bed carved down to water level so water flows
        has_submerged = 0
        for x, z, c, h in inland_river_points:
            self.assertLessEqual(h, WATER_LEVEL, f"Inland river at ({x}, {z}) should have water (height was {h})")
            if h < WATER_LEVEL:
                has_submerged += 1
                world.load_chunk(x // 16, WATER_LEVEL // 16, z // 16)
                self.assertEqual(world.get_block(x, WATER_LEVEL, z), WATER)
        self.assertGreater(has_submerged, 0, "River must have submerged water blocks inland")

    def test_river_cuts_through_coastal_beach_cliffs(self):
        # Verify river channels carve below WATER_LEVEL through coastal beach zones
        seed = 42
        mouth_points = []
        for z in range(-300, 301, 6):
            for x in range(-300, 301, 6):
                c = climate_at(x, z, seed, BIOMES)
                if 0.30 < c <= 0.36 and is_river(x, z, seed, climate=c):
                    h = terrain_height(x, z, seed)
                    mouth_points.append((x, z, h))

        self.assertTrue(mouth_points, "Should find river mouth points in coastal beach zone")
        for x, z, h in mouth_points:
            self.assertLessEqual(h, WATER_LEVEL, f"River mouth at ({x}, {z}) should cut through beach down to water level")

    def test_riverbed_blocks_sand_and_mud_instead_of_grass(self):
        # Verify riverbed surface under water uses sand, mud, gravel, clay and never grass
        seed = 42
        world = World(seed=seed, caves=[], ores=[])
        bed_blocks = set()
        has_sand = False
        has_mud = False

        for z in range(-200, 201, 8):
            for x in range(-200, 201, 8):
                c = climate_at(x, z, seed, BIOMES)
                if c >= 0.45 and is_river(x, z, seed, climate=c):
                    h = terrain_height(x, z, seed)
                    if h < WATER_LEVEL:
                        world.load_chunk(x // 16, h // 16, z // 16)
                        b = world.get_block(x, h, z)
                        bed_blocks.add(b)
                        if b == SAND:
                            has_sand = True
                        elif b == MUD:
                            has_mud = True

        self.assertNotIn(GRASS, bed_blocks, "Submerged riverbed should never have grass")
        self.assertTrue(has_sand, "Riverbed should contain sand patches")
        self.assertTrue(has_mud, "Riverbed should contain mud patches")
        for b in bed_blocks:
            self.assertIn(b, (SAND, MUD, CLAY, GRAVEL), f"Unexpected riverbed block: {b}")

    def test_river_edge_has_muddy_grass(self):
        # Verify river edge at waterline (y = 64..65) has muddy_grass, while higher terrain has grass
        seed = 42
        world = World(seed=seed, caves=[], ores=[])
        found_muddy_grass = False
        found_upland_grass = False

        for z in range(-200, 201, 8):
            for x in range(-200, 201, 8):
                c = climate_at(x, z, seed, BIOMES)
                if c >= 0.45:
                    h = terrain_height(x, z, seed)
                    if h in (WATER_LEVEL, WATER_LEVEL + 1):
                        world.load_chunk(x // 16, h // 16, z // 16)
                        b = world.get_block(x, h, z)
                        if b == MUDDY_GRASS:
                            found_muddy_grass = True
                    elif h >= WATER_LEVEL + 2:
                        world.load_chunk(x // 16, h // 16, z // 16)
                        b = world.get_block(x, h, z)
                        if b == GRASS:
                            found_upland_grass = True
                if found_muddy_grass and found_upland_grass:
                    break
            if found_muddy_grass and found_upland_grass:
                break

        self.assertTrue(found_muddy_grass, "River edge at waterline should have muddy_grass")
        self.assertTrue(found_upland_grass, "Upland terrain should retain standard grass")


if __name__ == "__main__":
    unittest.main()

