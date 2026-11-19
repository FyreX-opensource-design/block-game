import unittest

import numpy as np

from server.biomes import BIOMES, biome_at
from server.blocks import AIR, BLOCKS, CHUNK_SIZE, WATER_LEVEL
from server.items import ITEMS
from server.noise import terrain_height
from server.underground_biomes import load_underground_biomes, underground_biome_at
from server.weather import pack_grid
from server.world import World

SNOW_BIOMES = ("cold_plains", "arctic_forest", "snowy_peaks")


def _find(biome_id, seed=1, limit=12000, above_water=False):
    for index in range(limit):
        x = (index * 37) % 2400 - 1200
        z = (index * 91) % 2400 - 1200
        if biome_at(x, z, seed, BIOMES).id != biome_id:
            continue
        if above_water and terrain_height(x, z, seed) <= WATER_LEVEL:
            continue
        return x, z
    raise AssertionError(f"no {biome_id} column in {limit} samples")


def _rain(world, x, z, wet=1.0):
    world.weather = {
        "cell": 64.0,
        "origin": [float(x) - 0.5, float(z) - 0.5],
        "size": 1,
        "rain": pack_grid(np.array([[wet]], dtype=np.float64)),
    }


def _clear_column(world, x, z, y):
    cx, cz = x // CHUNK_SIZE, z // CHUNK_SIZE
    cy = y // CHUNK_SIZE
    world.load_chunk(cx, cy, cz)
    world.chunks[(cx, cy, cz)].data.fill(0)
    return cy


class SnowTests(unittest.TestCase):
    def test_snow_biomes_are_patches_of_their_parents(self):
        plains = next(biome for biome in BIOMES if biome.id == "plains")
        forest = next(biome for biome in BIOMES if biome.id == "forest")
        peaks = next(biome for biome in BIOMES if biome.id == "mountain_peaks")
        self.assertEqual(plains.precipitation, "rain")
        self.assertEqual(forest.precipitation, "rain")
        cold = next(biome for biome in plains.variants if biome.id == "cold_plains")
        arctic = next(biome for biome in forest.variants if biome.id == "arctic_forest")
        snowy = next(biome for biome in peaks.variants if biome.id == "snowy_peaks")
        for biome in (cold, arctic, snowy):
            self.assertEqual(biome.precipitation, "snow")
        forest_ids = [biome.id for biome in forest.variants]
        self.assertLess(forest_ids.index("arctic_forest"), forest_ids.index("birch_forest"))
        self.assertEqual([tree.id for tree in arctic.trees], ["spruce", "tall_spruce"])
        self.assertEqual(snowy.height_offset, peaks.height_offset)
        for biome_id in SNOW_BIOMES:
            self.assertIsNotNone(_find(biome_id, above_water=True))

    def test_snow_layers_blocks_and_collision(self):
        self.assertEqual(BLOCKS["snow_layer"].properties["snow_layers"], 1)
        self.assertFalse(BLOCKS["snow_layer"].full_cube)
        self.assertEqual(BLOCKS["snow_layer_8"].properties["snow_layers"], 8)
        self.assertTrue(BLOCKS["ice"].translucent)
        self.assertTrue(BLOCKS["salt_ice"].properties["ice"])
        self.assertTrue(BLOCKS["brackish_ice"].properties["ice"])
        self.assertTrue(BLOCKS["packed_ice"].properties["ice"])
        self.assertIn("snow_layer", ITEMS)
        self.assertEqual(ITEMS["snow_layer"].block, "snow_layer")

        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, 1, 1, 80)
        world.set_block(1, 80, 1, "snow_layer")
        height = world.get_collision_boxes(1, 80, 1)[0]
        self.assertAlmostEqual(height[4] - height[1], 0.125)
        world.set_block(1, 80, 1, "snow_layer_4")
        height = world.get_collision_boxes(1, 80, 1)[0]
        self.assertAlmostEqual(height[4] - height[1], 0.5)

    def test_snowfall_lays_and_thickens_layers_only_in_snow_biomes(self):
        snow_x, snow_z = _find("cold_plains", above_water=True)
        rain_x, rain_z = _find("plains", above_water=True)
        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, snow_x, snow_z, 80)
        _clear_column(world, rain_x, rain_z, 80)
        world.set_block(snow_x, 79, snow_z, "stone")
        world.set_block(snow_x, 80, snow_z, "grass")
        world.set_block(rain_x, 79, rain_z, "stone")
        world.set_block(rain_x, 80, rain_z, "grass")

        self.assertIsNone(world.snow.accumulate(snow_x, snow_z))
        _rain(world, snow_x, snow_z, 0.0)
        self.assertIsNone(world.snow.accumulate(snow_x, snow_z))
        _rain(world, snow_x, snow_z, 1.0)
        self.assertEqual(world.snow.accumulate(snow_x, snow_z), (snow_x, 81, snow_z, "snow_layer"))
        self.assertEqual(world.snow.accumulate(snow_x, snow_z), (snow_x, 81, snow_z, "snow_layer_2"))

        _rain(world, rain_x, rain_z, 1.0)
        self.assertIsNone(world.snow.accumulate(rain_x, rain_z))
        self.assertEqual(world.get_block(rain_x, 81, rain_z), AIR)

        for y in range(81, 85):
            world.set_block(snow_x, y, snow_z, AIR)
        world.set_block(snow_x, 80, snow_z, "grass")
        world.set_block(snow_x, 81, snow_z, "leaves")
        world.set_block(snow_x, 82, snow_z, "leaves")
        _rain(world, snow_x, snow_z, 1.0)
        self.assertIsNone(world.snow.accumulate(snow_x, snow_z))
        self.assertEqual(world.get_block(snow_x, 81, snow_z), "leaves")

        world.set_block(snow_x, 81, snow_z, "snow_layer_8")
        world.set_block(snow_x, 82, snow_z, AIR)
        self.assertIsNone(world.snow.accumulate(snow_x, snow_z))

        world.set_block(snow_x, 80, snow_z, "water")
        world.set_block(snow_x, 81, snow_z, AIR)
        self.assertIsNone(world.snow.accumulate(snow_x, snow_z))
        self.assertEqual(world.get_block(snow_x, 80, snow_z), "water")

        world.snow._sample_columns = lambda: [(snow_x, snow_z)]
        world.snow.tick_timer = world.snow.tick_interval
        world.set_block(snow_x, 79, snow_z, "stone")
        world.set_block(snow_x, 80, snow_z, "grass")
        world.set_block(snow_x, 81, snow_z, AIR)
        changed = world.update_snow(0.0)
        self.assertIn((snow_x, 81, snow_z, "snow_layer"), changed)

    def test_generated_snow_biome_starts_with_a_layer(self):
        x, z = _find("cold_plains", above_water=True)
        world = World(seed=1, caves=[], ores=[], features=[])
        world.generate_column(x // CHUNK_SIZE, z // CHUNK_SIZE)
        height = terrain_height(x, z, 1)
        self.assertEqual(world.get_block(x, height, z), "grass")
        self.assertEqual(world.get_block(x, height + 1, z), "snow_layer")

    def test_exposed_water_freezes_into_its_own_ice(self):
        x, z = _find("arctic_forest", above_water=True)
        plains_x, plains_z = _find("plains", above_water=True)
        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, x, z, 180)
        _clear_column(world, plains_x, plains_z, 180)
        column = [
            (176, "stone"),
            (177, "water"),
            (178, "water"),
            (179, AIR),
            (180, "stone"),
            (181, "salt_water"),
            (182, AIR),
            (183, "stone"),
            (184, "brackish_water"),
            (185, AIR),
            (186, "stone"),
            (187, "water_5"),
            (188, AIR),
            (189, "stone"),
            (190, "lava"),
            (191, AIR),
        ]
        for y, block in column:
            world.set_block(x, y, z, block)
        world.set_block(plains_x, 180, plains_z, "stone")
        world.set_block(plains_x, 181, plains_z, "water")
        world.set_block(plains_x, 182, plains_z, "stone")
        world.set_block(plains_x, 183, plains_z, "water")

        rules = {rule.id: rule for rule in world.transformations.rules}
        for rule_id, y in (
            ("water_freezes", 177),
            ("water_freezes", 178),
            ("salt_water_freezes", 181),
            ("brackish_water_freezes", 184),
            ("water_freezes", 187),
        ):
            world.transformations.timers[(x, y, z)] = [rules[rule_id], 0.0]
        world.transformations.timers[(plains_x, 181, plains_z)] = [rules["water_freezes"], 0.0]
        world.transformations.timers[(plains_x, 183, plains_z)] = [rules["water_freezes"], 0.0]
        world.transformations.update(1.0)

        self.assertEqual(world.get_block(x, 177, z), "water")
        self.assertEqual(world.get_block(x, 178, z), "ice")
        self.assertEqual(world.get_block(x, 181, z), "salt_ice")
        self.assertEqual(world.get_block(x, 184, z), "brackish_ice")
        self.assertEqual(world.get_block(x, 187, z), "ice")
        self.assertEqual(world.get_block(x, 190, z), "lava")
        self.assertEqual(world.get_block(plains_x, 181, plains_z), "water")
        self.assertEqual(world.get_block(plains_x, 183, plains_z), "water")

        world.set_block(x, 178, z, "water")
        world.set_block(x, 179, z, "ice")
        world.transformations.timers[(x, 178, z)] = [rules["water_freezes"], 0.0]
        world.transformations.update(1.0)
        self.assertEqual(world.get_block(x, 178, z), "water")

    def test_ice_caves_only_open_under_snow_biomes(self):
        biomes = load_underground_biomes()
        ice = next(biome for biome in biomes if biome.id == "ice_caves")
        self.assertFalse(ice.always)
        self.assertEqual(set(ice.surface_biomes), set(SNOW_BIOMES))
        self.assertLess(ice.threshold, 1.0)
        self.assertEqual(ice.default_stone, "packed_ice")

        found = False
        for index in range(2500):
            x = (index * 19) % 1800 - 900
            z = (index * 23) % 1800 - 900
            surface = biome_at(x, z, 1, BIOMES).id
            for y in (12, 24, 36):
                underground = underground_biome_at(x, y, z, 1, biomes)
                if surface not in SNOW_BIOMES:
                    self.assertNotEqual(getattr(underground, "id", None), "ice_caves")
                elif underground is not None and underground.id == "ice_caves":
                    found = True
            if found and index > 400:
                break
        self.assertTrue(found, "ice caves should appear under some snowy columns")

    def test_snow_particles_replace_rain_in_a_snow_biome(self):
        from client.render import _rain_vertices

        snow_x, snow_z = _find("snowy_peaks")
        rain_x, rain_z = _find("plains")

        class Sky:
            biomes = BIOMES
            seed = 1

            def column_top(self, x, z):
                return 64

        def streaks(x, z):
            weather = {
                "cloud_y": 200.0,
                "_cloud_ready": True,
                "_puffs": np.array([[float(x) + 0.4, float(z) + 0.4, 4.0, 0.9]], dtype=np.float32),
            }
            verts = _rain_vertices(Sky(), weather, (float(x) + 0.4, 90.0, float(z) + 0.4), 1.0)
            self.assertIsNotNone(verts)
            shaped = verts.reshape(-1, 6, 9)
            return shaped[:, :, 1].max(axis=1) - shaped[:, :, 1].min(axis=1), shaped[:, 0, 3]

        snow_span, snow_red = streaks(snow_x, snow_z)
        rain_span, rain_red = streaks(rain_x, rain_z)
        self.assertLess(float(snow_span.mean()), 1.0)
        self.assertGreater(float(snow_red.mean()), 0.9)
        self.assertGreater(float(rain_span.mean()), 2.0)
        self.assertLess(float(rain_red.mean()), 0.85)

    def test_snow_regions_stay_together(self):
        flowers = {
            "wildflower_patch",
            "rose_thicket",
            "alpine_flower_patch",
            "mint_glade",
            "orchid_clearing",
            "lavender_hollow",
            "sunburst_glade",
            "autumn_blossom_patch",
        }
        for biome_id in SNOW_BIOMES:
            found = False
            checked = 0
            for index in range(9000):
                x = (index * 37) % 2400 - 1200
                z = (index * 91) % 2400 - 1200
                if biome_at(x, z, 1, BIOMES).id != biome_id:
                    continue
                checked += 1
                neighborhood = [
                    biome_at(x + dx, z + dz, 1, BIOMES).id
                    for dx in (-8, 0, 8)
                    for dz in (-8, 0, 8)
                ]
                if all(name == biome_id for name in neighborhood):
                    found = True
                    self.assertFalse(flowers.intersection(neighborhood))
                    break
                if checked >= 80:
                    break
            self.assertTrue(found, f"{biome_id} should form a solid patch, not a lace of other biomes")

    def test_spruce_trunk_replaces_the_snow_layer(self):
        from server.trees import DECORATORS

        spruce = next(tree for tree in DECORATORS if tree.id == "spruce")
        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, 8, 8, 80)
        world.set_block(8, 80, 8, "grass")
        world.set_block(8, 81, 8, "snow_layer")
        chunk = world.chunks[(0, 5, 0)]
        world._place_tree(chunk.data, 8, 8, 80, 8, 8, 80, spruce)
        self.assertEqual(world.get_block(8, 81, 8), "spruce_log")
        self.assertEqual(world.get_block(8, 82, 8), "spruce_log")

    def test_player_steps_from_bare_ground_onto_snow(self):
        from server.player import Player

        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, 0, 0, 5)
        for x in range(4):
            for z in range(3):
                world.set_block(x, 5, z, "stone")
        world.set_block(2, 6, 1, "stone")
        world.set_block(2, 7, 1, "snow_layer")
        player = Player(world)
        player.pos[:] = [1.0, 6.0, 1.5]
        player.move_horizontal(1.6, 0.0)
        self.assertGreater(player.pos[0], 1.6)
        self.assertAlmostEqual(player.pos[1], 7.125, delta=0.08)

        world.set_block(2, 7, 1, "stone")
        player.pos[:] = [1.0, 6.0, 1.5]
        player.move_horizontal(1.6, 0.0)
        self.assertLess(player.pos[1], 6.5)
        self.assertLess(player.pos[0], 2.0)

    def test_player_walks_onto_snow_at_the_same_height(self):
        from server.player import Player

        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, 0, 0, 5)
        for x in range(4):
            for z in range(3):
                world.set_block(x, 5, z, "stone")
        world.set_block(2, 6, 1, "snow_layer")
        player = Player(world)
        player.pos[:] = [1.0, 6.0, 1.5]
        player.move_horizontal(1.6, 0.0)
        self.assertGreater(player.pos[0], 1.6)
        self.assertAlmostEqual(player.pos[1], 6.125, delta=0.08)

        world.set_block(2, 6, 1, "snow_layer_3")
        player.pos[:] = [1.0, 6.0, 1.5]
        player.move_horizontal(1.6, 0.0)
        self.assertGreater(player.pos[0], 1.6)
        self.assertAlmostEqual(player.pos[1], 6.375, delta=0.08)

    def test_ice_hides_the_faces_inside_a_sheet(self):
        from server.mesh import build_chunk_mesh

        world = World(seed=1, caves=[], ores=[], features=[])
        cy = _clear_column(world, 0, 0, 4)
        world.set_block(2, 4, 2, "ice")
        world.set_block(3, 4, 2, "ice")
        mesh = build_chunk_mesh(world, 0, cy, 0)
        self.assertIsNotNone(mesh.translucent)
        # Two cubes share one face, so the pair keeps 10 quads instead of 12.
        self.assertEqual(len(mesh.translucent[0]), 10 * 4)

        world.set_block(3, 4, 2, "salt_ice")
        seam = build_chunk_mesh(world, 0, cy, 0)
        self.assertEqual(len(seam.translucent[0]), 12 * 4)

    def test_snow_layer_mesh_uses_its_height(self):
        from server.mesh import build_chunk_mesh

        world = World(seed=1, caves=[], ores=[], features=[])
        _clear_column(world, 0, 0, 2)
        world.set_block(1, 1, 1, "stone")
        world.set_block(2, 2, 1, "stone")
        world.set_block(1, 2, 1, "snow_layer")
        mesh = build_chunk_mesh(world, 0, 0, 0)
        verts = mesh.vertices
        tops = verts[(np.abs(verts[:, 4] - 1.0) < 1e-5) & (np.abs(verts[:, 1] - 2.125) < 1e-4)]
        self.assertGreater(len(tops), 0)
        # The stone beside the snow hides the snow's east face. The floor's own
        # east face still ends at y=2, so only vertices above that belong to the snow.
        snow_east = verts[
            (np.abs(verts[:, 3] - 1.0) < 1e-5)
            & (np.abs(verts[:, 0] - 2.0) < 1e-4)
            & (verts[:, 1] > 2.05)
        ]
        self.assertEqual(len(snow_east), 0)

    def test_cold_plains_meet_arctic_forest(self):
        from server.biomes import climate_at

        cold = next(biome for biome in next(item for item in BIOMES if item.id == "plains").variants if biome.id == "cold_plains")
        arctic = next(biome for biome in next(item for item in BIOMES if item.id == "forest").variants if biome.id == "arctic_forest")
        self.assertEqual(cold.variant_noise, arctic.variant_noise)
        directions = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
        gap = {"highland_plains", "lavender_hollow", "orchid_clearing"}
        found = False
        for index in range(2500):
            x = (index * 37) % 2400 - 1200
            z = (index * 91) % 2400 - 1200
            if biome_at(x, z, 1, BIOMES).id != "cold_plains":
                continue
            _rise, dx, dz = max(
                (climate_at(x + step_x * 48, z + step_z * 48, 1, BIOMES), step_x, step_z)
                for step_x, step_z in directions
            )
            path = []
            for step in range(1, 160):
                path.append(biome_at(x + dx * step * 3, z + dz * step * 3, 1, BIOMES).id)
                if path[-1] == "arctic_forest":
                    break
            if "arctic_forest" not in path or gap.intersection(path[: path.index("arctic_forest")]):
                continue
            found = True
            break
        self.assertTrue(found, "cold plains should meet arctic forest without a highland plains gap")

    def test_snow_and_redwood_regions_are_broad(self):
        def longest(biome_id):
            best = 0
            for z in range(-800, 801, 80):
                run = 0
                for x in range(-800, 801, 4):
                    if biome_at(x, z, 1, BIOMES).id == biome_id:
                        run += 4
                        best = max(best, run)
                    else:
                        run = 0
            for x in range(-800, 801, 80):
                run = 0
                for z in range(-800, 801, 4):
                    if biome_at(x, z, 1, BIOMES).id == biome_id:
                        run += 4
                        best = max(best, run)
                    else:
                        run = 0
            return best

        for biome_id in ("cold_plains", "arctic_forest", "redwood_forest"):
            self.assertGreaterEqual(longest(biome_id), 64, biome_id)


if __name__ == "__main__":
    unittest.main()
