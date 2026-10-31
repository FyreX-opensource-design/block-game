import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from server.biomes import BIOMES, all_biomes, biome_at, biome_index_grid, climate_at, climate_grid, load_biomes
from server.blocks import AIR, CLAY, GRAVEL, GRASS, LOG, SALT_WATER, SAND, STONE, WATER, WATER_LEVEL
from server.noise import cliff_factor, coast_noise, fbm, terrain_height
from server.trees import DECORATORS, load_trees
from server.world import World, tree_at


class BiomeTests(unittest.TestCase):
    def test_plains_and_forests_share_the_oak_decorator(self):
        self.assertEqual(
            [biome.id for biome in BIOMES],
            [
                "ocean",
                "beach",
                "lowland_plains",
                "plains",
                "highland_plains",
                "forest",
                "pine_forest",
                "mountains",
                "wooded_mountains",
                "mountain_meadow",
                "mountain_peaks",
                "mountain_cliffs",
            ],
        )
        self.assertEqual(BIOMES[0].surface, SAND)
        self.assertEqual(BIOMES[1].surface, SAND)
        self.assertEqual(BIOMES[2].surface, GRASS)
        forest_b = next(b for b in BIOMES if b.id == "forest")
        plains_b = next(b for b in BIOMES if b.id == "plains")
        self.assertGreater(forest_b.tree_chance, plains_b.tree_chance * 3)
        self.assertIn("oak", [tree.id for tree in DECORATORS])
        self.assertEqual(forest_b.trees[0].trunk, LOG)
        counts = {b.id: 0 for b in all_biomes(BIOMES)}
        for z in range(-240, 241, 12):
            for x in range(-240, 241, 12):
                counts[biome_at(x, z, 1, BIOMES).id] += 1
        for name in (
            "ocean",
            "beach",
            "lowland_plains",
            "plains",
            "highland_plains",
            "forest",
            "pine_forest",
            "mountains",
            "wooded_mountains",
            "mountain_meadow",
            "mountain_peaks",
            "mountain_cliffs",
        ):
            self.assertGreater(counts[name], 0)
        biome = BIOMES[0]
        grid = climate_grid(np.array([[4.0]]), np.array([[-6.0]]), 1, BIOMES)
        expected = climate_at(4, -6, 1, BIOMES)
        self.assertAlmostEqual(float(grid[0, 0]), expected)

    def test_forests_grow_trees_more_often_than_plains(self):
        counts = {"plains": [0, 0], "forest": [0, 0]}
        for z in range(-160, 161, 4):
            for x in range(-160, 161, 4):
                biome = biome_at(x, z, 2, BIOMES)
                if biome.id in counts:
                    counts[biome.id][0] += 1
                    if tree_at(x, z, 2):
                        counts[biome.id][1] += 1
        plains_rate = counts["plains"][1] / counts["plains"][0]
        forest_rate = counts["forest"][1] / counts["forest"][0]
        self.assertGreater(forest_rate, plains_rate)
        plains_rate = counts["plains"][1] / counts["plains"][0]
        forest_rate = counts["forest"][1] / counts["forest"][0]
        self.assertGreater(forest_rate, plains_rate)

    def test_json_and_lua_define_biomes_and_trees(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trees = root / "trees"
            biomes = root / "biomes"
            trees.mkdir()
            biomes.mkdir()
            (trees / "pine.lua").write_text(
                "return { id = 'pine', trunk = 'log', leaves = 'leaves', trunk_min = 6, radius = 1 }\n",
                encoding="utf-8",
            )
            (biomes / "woods.json").write_text(
                json.dumps(
                    {
                        "id": "woods",
                        "ceiling": 1,
                        "scale": 0.01,
                        "octaves": 1,
                        "seed_offset": 3,
                        "tree_chance": 1,
                        "trees": ["pine"],
                    }
                ),
                encoding="utf-8",
            )
            loaded_trees = load_trees(trees)
            loaded = load_biomes(biomes, loaded_trees)
            self.assertEqual(loaded_trees[0].trunk_min, 6)
            self.assertEqual(loaded[0].trees[0].id, "pine")
            world = World(seed=1, caves=[], biomes=loaded)
            world.generate_column(0, 0)
            height = terrain_height(8, 8, 1, world.biomes)
            self.assertIn(world.get_block(8, height, 8), (GRASS, SAND))

    def test_bad_biome_file_is_rejected(self):
        with self.assertRaises(ValueError):
            load_biomes_mapping = {"id": "Hills", "ceiling": 1, "trees": ["missing"]}
            from server.biomes import biome_from_mapping

            biome_from_mapping(load_biomes_mapping, "biome", {})

    def test_ocean_and_beach_generation(self):
        world = World(seed=1, caves=[])
        # Find an ocean coordinate and beach coordinate
        ocean_pt = None
        beach_pt = None
        for z in range(-150, 150, 8):
            for x in range(-150, 150, 8):
                b = biome_at(x, z, 1, BIOMES)
                if b.id == "ocean" and ocean_pt is None:
                    ocean_pt = (x, z)
                elif b.id == "beach" and beach_pt is None:
                    beach = next(biome for biome in BIOMES if biome.id == "beach")
                    cliff = cliff_factor(coast_noise(x, z, 1, beach.coast), beach.coast["cliff"])
                    if cliff <= 0.5:
                        beach_pt = (x, z)
                if ocean_pt and beach_pt:
                    break
            if ocean_pt and beach_pt:
                break

        self.assertIsNotNone(ocean_pt)
        self.assertIsNotNone(beach_pt)

        ox, oz = ocean_pt
        world.generate_column(ox // 16, oz // 16)
        h_ocean = terrain_height(ox, oz, 1)
        self.assertGreaterEqual(h_ocean, 50)
        self.assertLessEqual(h_ocean, 60)
        self.assertIn(world.get_block(ox, h_ocean, oz), (SAND, GRAVEL, CLAY))
        for y in range(h_ocean + 1, WATER_LEVEL + 1):
            self.assertEqual(world.get_block(ox, y, oz), SALT_WATER)

        bx, bz = beach_pt
        world.generate_column(bx // 16, bz // 16)
        h_beach = terrain_height(bx, bz, 1)
        self.assertEqual(world.get_block(bx, h_beach, bz), SAND)

    def test_beach_generates_sandstone_under_the_sand(self):
        world = World(seed=1, caves=[])
        beach = next(biome for biome in BIOMES if biome.id == "beach")
        self.assertEqual(beach.surface, SAND)
        self.assertEqual(beach.soil, "sand")
        self.assertEqual(beach.subsoil, "sandstone")
        self.assertEqual(beach.soil_depth, 3)
        self.assertEqual(beach.subsoil_depth, 3)

        beach_pt = None
        for z in range(-150, 150, 8):
            for x in range(-150, 150, 8):
                if biome_at(x, z, 1, BIOMES).id == "beach":
                    beach_pt = (x, z)
                    break
            if beach_pt:
                break
        self.assertIsNotNone(beach_pt)

        bx, bz = beach_pt
        world.generate_column(bx // 16, bz // 16)
        h_beach = terrain_height(bx, bz, 1)
        self.assertEqual(world.get_block(bx, h_beach, bz), SAND)
        self.assertEqual(world.get_block(bx, h_beach - 1, bz), SAND)
        for y in range(int(h_beach) - int(beach.soil_depth) - int(beach.subsoil_depth), int(h_beach) - int(beach.soil_depth)):
            self.assertEqual(world.get_block(bx, y, bz), "sandstone")
        for y in range(int(h_beach) - int(beach.soil_depth), int(h_beach)):
            self.assertEqual(world.get_block(bx, y, bz), "sand")
        self.assertNotEqual(world.get_block(bx, int(h_beach) - int(beach.soil_depth) - int(beach.subsoil_depth) - 1, bz), "sandstone")

    def test_beach_boundary_stays_smooth_away_from_cliffs(self):
        # Sample across the beach boundary in steps of 1 block. Steep steps are
        # allowed only on stretches that the coastal noise marks as cliffs.
        beach = next(biome for biome in BIOMES if biome.id == "beach")
        cliff_cfg = beach.coast["cliff"]
        for z in range(-60, 60, 15):
            row_heights = [terrain_height(x, z, 1) for x in range(-100, 100)]
            for i in range(len(row_heights) - 1):
                step = abs(row_heights[i + 1] - row_heights[i])
                if step <= 3:
                    continue
                x = -100 + i
                cliff = max(
                    cliff_factor(coast_noise(x, z, 1, beach.coast), cliff_cfg),
                    cliff_factor(coast_noise(x + 1, z, 1, beach.coast), cliff_cfg),
                )
                self.assertGreater(cliff, 0.5, f"Unexpected {step}-block step at {x},{z}")

    def test_coastal_gravel_beaches_and_cliffs(self):
        beach = next(biome for biome in BIOMES if biome.id == "beach")
        gravel_beach = next((variant for variant in beach.variants if variant.id == "gravel_beach"), None)
        cliff_beach = next((variant for variant in beach.variants if variant.id == "cliff_beach"), None)
        self.assertIsNotNone(gravel_beach, "gravel_beach should be a beach variant asset")
        self.assertIsNotNone(cliff_beach, "cliff_beach should be a beach variant asset")
        self.assertEqual(gravel_beach.surface, GRAVEL)
        self.assertEqual(gravel_beach.soil, GRAVEL)
        self.assertEqual(cliff_beach.surface, STONE)
        self.assertEqual(cliff_beach.soil, STONE)
        self.assertIn("cliff", beach.coast)

        world = World(seed=1, caves=[])
        sand_pt = None
        gravel_pt = None
        cliff_pt = None
        for z in range(-300, 301, 8):
            for x in range(-300, 301, 8):
                biome = biome_at(x, z, 1, BIOMES)
                if biome.id not in ("beach", "gravel_beach", "cliff_beach"):
                    continue
                height = terrain_height(x, z, 1)
                if height < WATER_LEVEL:
                    continue
                if biome.id == "cliff_beach" and height >= WATER_LEVEL + 2 and cliff_pt is None:
                    for dz in range(-8, 9):
                        for dx in range(-8, 9):
                            if biome_at(x + dx, z + dz, 1, BIOMES).id != "ocean":
                                continue
                            if terrain_height(x + dx, z + dz, 1) < WATER_LEVEL:
                                cliff_pt = (x, z)
                                break
                        if cliff_pt:
                            break
                elif biome.id == "gravel_beach" and gravel_pt is None:
                    gravel_pt = (x, z)
                elif biome.id == "beach" and sand_pt is None:
                    sand_pt = (x, z)
                if sand_pt and gravel_pt and cliff_pt:
                    break
            if sand_pt and gravel_pt and cliff_pt:
                break

        self.assertIsNotNone(sand_pt, "Expected some sandy beaches to remain")
        self.assertIsNotNone(gravel_pt, "Expected some gravel beaches")
        self.assertIsNotNone(cliff_pt, "Expected some coastal cliffs")

        for (x, z), expected in ((sand_pt, SAND), (gravel_pt, GRAVEL), (cliff_pt, STONE)):
            world.generate_column(x // 16, z // 16)
            height = terrain_height(x, z, 1)
            self.assertEqual(world.get_block(x, height, z), expected)

        # A cliff should rise above the water and overlook nearby ocean water.
        cx, cz = cliff_pt
        cliff_height = terrain_height(cx, cz, 1)
        self.assertGreaterEqual(cliff_height, WATER_LEVEL + 2)
        self.assertEqual(world.get_block(cx, cliff_height - 1, cz), cliff_beach.soil)
        nearby_ocean = False
        for dz in range(-12, 13):
            for dx in range(-12, 13):
                x, z = cx + dx, cz + dz
                if biome_at(x, z, 1, BIOMES).id != "ocean":
                    continue
                ocean_height = terrain_height(x, z, 1)
                if ocean_height < WATER_LEVEL:
                    world.generate_column(x // 16, z // 16)
                    self.assertEqual(world.get_block(x, WATER_LEVEL, z), SALT_WATER)
                    nearby_ocean = True
                    break
            if nearby_ocean:
                break
        self.assertTrue(nearby_ocean, "Coastal cliffs should overlook ocean water")

    def test_variants_do_not_leak_into_other_biomes(self):
        # Variants sit between base biomes in the generated catalog. Their
        # indices must not shift other biomes onto the wrong surface/soil.
        xs = np.arange(-150, 151, 10)
        zs = np.arange(-150, 151, 10)
        xx, zz = np.meshgrid(xs, zs)
        grid, catalog = biome_index_grid(xx, zz, 1, BIOMES)
        world = World(seed=1, caves=[])
        checked = 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x, z = int(xx[i, j]), int(zz[i, j])
                expected = biome_at(x, z, 1, BIOMES)
                self.assertEqual(catalog[grid[i, j]].id, expected.id)
                if expected.surface != GRASS:
                    continue
                height = terrain_height(x, z, 1)
                if height < WATER_LEVEL:
                    continue
                world.generate_column(x // 16, z // 16)
                top = world.get_block(x, height, z)
                if top not in (GRASS, GRAVEL, STONE):
                    continue  # tree or decoration replaced the surface
                self.assertEqual(top, GRASS, f"{expected.id} at {x},{z} should keep its grass surface")
                checked += 1
        self.assertGreater(checked, 0)

    def test_seabed_mixes_sand_gravel_clay(self):
        world = World(seed=1, caves=[])
        blocks_found = set()
        for cz in range(-4, 4):
            for cx in range(-4, 4):
                world.generate_column(cx, cz)
                for lz in range(0, 16, 4):
                    for lx in range(0, 16, 4):
                        wx = cx * 16 + lx
                        wz = cz * 16 + lz
                        h = terrain_height(wx, wz, 1)
                        if h < WATER_LEVEL:
                            blocks_found.add(world.get_block(wx, h, wz))
        self.assertIn(SAND, blocks_found)
        self.assertIn(GRAVEL, blocks_found)
        self.assertIn(CLAY, blocks_found)

    def test_caves_do_not_breach_ocean_water(self):
        world = World(seed=1)
        for z in range(-120, 120, 16):
            for x in range(-120, 120, 16):
                b = biome_at(x, z, 1, BIOMES)
                if b.id == "ocean":
                    world.generate_column(x // 16, z // 16)
                    h = terrain_height(x, z, 1)
                    # Ocean water blocks must remain salt_water
                    for y in range(h + 1, WATER_LEVEL + 1):
                        self.assertEqual(world.get_block(x, y, z), SALT_WATER)
                    # Seabed and top 3 blocks beneath must remain solid, not cave air
                    for y in range(max(1, h - 3), h + 1):
                        self.assertNotEqual(world.get_block(x, y, z), AIR)

    def test_spawn_not_in_ocean_or_beach(self):
        world = World(seed=1)
        sx, sz, sy = world.find_spawn()
        biome = biome_at(sx, sz, 1, BIOMES)
        self.assertNotIn(biome.id, ("ocean", "beach", "gravel_beach", "cliff_beach"))
        self.assertGreater(sy, WATER_LEVEL)

    def test_biome_grass_colors_and_block_tinting(self):
        from server.blocks import BLOCKS
        from server.biomes import parse_color

        # Biomes have distinct grass colors defined
        plains = next(b for b in BIOMES if b.id == "plains")
        forest = next(b for b in BIOMES if b.id == "forest")
        meadow = next(b for b in BIOMES if b.id == "mountain_meadow")

        self.assertIsNotNone(plains.grass_color)
        self.assertIsNotNone(forest.grass_color)
        self.assertIsNotNone(meadow.grass_color)
        self.assertNotEqual(plains.grass_color, forest.grass_color)
        self.assertNotEqual(forest.grass_color, meadow.grass_color)

        # parse_color utility checks
        self.assertEqual(parse_color("#83b749"), (131 / 255.0, 183 / 255.0, 73 / 255.0))
        self.assertEqual(parse_color([100, 200, 50]), (100 / 255.0, 200 / 255.0, 50 / 255.0))

        # Grass and muddy grass blocks have tint="grass"
        self.assertEqual(BLOCKS["grass"].tint, "grass")
        self.assertEqual(BLOCKS["muddy_grass"].tint, "grass")
        self.assertEqual(BLOCKS["tall_grass"].tint, "grass")
        self.assertEqual(BLOCKS["fern"].tint, "grass")
        # Stone and dirt are not grass tinted
        self.assertIsNone(BLOCKS["stone"].tint)
        self.assertIsNone(BLOCKS["dirt"].tint)

    def test_chunk_mesh_biome_grass_coloring(self):
        from server.mesh import build_chunk_mesh

        world = World(seed=42)
        cx, cy, cz = 0, 4, 0
        world.load_chunk(cx, cy, cz)
        chunk = world.chunks[(cx, cy, cz)]
        chunk.data.fill(0)

        # Place a grass block at (5, 65, 5) and a stone block at (10, 65, 10)
        world.set_block(5, 65, 5, "grass")
        world.set_block(10, 65, 10, "stone")

        mesh = build_chunk_mesh(world, cx, cy, cz)
        self.assertIsNotNone(mesh)
        self.assertIsNotNone(mesh.colors)
        self.assertEqual(len(mesh.colors), len(mesh.vertices))
        self.assertEqual(mesh.colors.shape[1], 3)

        # Verify that mesh has some non-white colors for grass, and white colors for stone
        has_tinted_vertex = False
        has_white_vertex = False
        for c in mesh.colors:
            if not np.allclose(c, [1.0, 1.0, 1.0], atol=1e-3):
                has_tinted_vertex = True
            else:
                has_white_vertex = True
        self.assertTrue(has_tinted_vertex, "Grass faces must receive biome tint colors")
        self.assertTrue(has_white_vertex, "Stone faces must remain pure untinted (1, 1, 1)")

    def test_biome_command(self):
        from server.commands import CommandHandler, CommandContext
        from server.player import Player

        world = World(seed=123)
        player = Player(world)
        player.pos = [0.0, 70.0, 0.0]

        handler = CommandHandler()
        ctx = CommandContext(player=player, world=world)
        handled = handler.execute(ctx, "/biome")
        self.assertTrue(handled)
        self.assertTrue(len(ctx.responses) > 0)
        self.assertIn("Biome:", ctx.responses[0][0])

    def test_plains_tiering_and_smooth_coast(self):
        biome_map = {b.id: b for b in BIOMES}
        self.assertIn("lowland_plains", biome_map)
        self.assertIn("plains", biome_map)
        self.assertIn("highland_plains", biome_map)

        beach = biome_map["beach"]
        lowland = biome_map["lowland_plains"]
        midland = biome_map["plains"]
        highland = biome_map["highland_plains"]
        forest = biome_map["forest"]

        # Ceilings climb in sequence from coast to highlands
        self.assertLess(beach.ceiling, lowland.ceiling)
        self.assertLess(lowland.ceiling, midland.ceiling)
        self.assertLess(midland.ceiling, highland.ceiling)
        self.assertLess(highland.ceiling, forest.ceiling)

        # Height offsets are tiered
        self.assertLess(lowland.height_offset, midland.height_offset)
        self.assertLess(midland.height_offset, highland.height_offset)
        self.assertLess(highland.height_offset, forest.height_offset)

        # Height scales are tiered
        self.assertLess(lowland.height_scale, midland.height_scale)
        self.assertLess(midland.height_scale, highland.height_scale)

        # Sample world coordinates to verify average elevation tiering
        low_heights = []
        mid_heights = []
        high_heights = []
        for z in range(-200, 201, 10):
            for x in range(-200, 201, 10):
                b = biome_at(x, z, 1, BIOMES)
                h = terrain_height(x, z, 1)
                if b.id == "lowland_plains":
                    low_heights.append(h)
                elif b.id == "plains":
                    mid_heights.append(h)
                elif b.id == "highland_plains":
                    high_heights.append(h)

        self.assertTrue(low_heights)
        self.assertTrue(mid_heights)
        self.assertTrue(high_heights)

        avg_low = np.mean(low_heights)
        avg_mid = np.mean(mid_heights)
        avg_high = np.mean(high_heights)

        self.assertLess(avg_low, avg_mid)
        self.assertLess(avg_mid, avg_high)
        # Lowland plains should stay close to sea level (WATER_LEVEL=64)
        self.assertLess(avg_low, 74.0)
        self.assertGreaterEqual(avg_low, 65.0)

    def test_biome_transition_assets_and_customization(self):
        from server.noise import _get_transition_params, apply_curve_scalar, apply_curve_grid

        biome_map = {b.id: b for b in BIOMES}
        ocean = biome_map["ocean"]
        beach = biome_map["beach"]
        lowland = biome_map["lowland_plains"]

        # Marine / coastal biomes have inland=False defined from assets
        self.assertFalse(ocean.inland)
        self.assertFalse(beach.inland)
        self.assertTrue(lowland.inland)

        # Asset-defined transition properties
        self.assertEqual(beach.transition_ceiling, 0.42)
        self.assertEqual(beach.transition_width, 0.16)
        self.assertEqual(ocean.floor, 50.0)
        self.assertEqual(ocean.floor_detail, 4.0)
        self.assertEqual(ocean.curve, "smoothstep")

        # Extraction helper
        b_c, o_w, fl_b, fl_d, o_curv, inlands, coast_cfg = _get_transition_params(BIOMES)
        self.assertEqual(b_c, 0.42)
        self.assertEqual(o_w, 0.16)
        self.assertEqual(fl_b, 50.0)
        self.assertEqual(fl_d, 4.0)
        self.assertEqual(o_curv, "smoothstep")
        self.assertNotIn(ocean, inlands)
        self.assertNotIn(beach, inlands)
        self.assertEqual(inlands[0].id, "lowland_plains")
        self.assertIn("cliff", coast_cfg)

        # Test curve functions (linear, smoothstep, smootherstep, cosine)
        for curve in ("linear", "smoothstep", "smootherstep", "cosine"):
            val_0 = apply_curve_scalar(0.0, curve)
            val_1 = apply_curve_scalar(1.0, curve)
            val_half = apply_curve_scalar(0.5, curve)
            self.assertAlmostEqual(val_0, 0.0)
            self.assertAlmostEqual(val_1, 1.0)
            self.assertAlmostEqual(val_half, 0.5)

            grid_vals = apply_curve_grid(np.array([0.0, 0.5, 1.0]), curve)
            self.assertTrue(np.allclose(grid_vals, [0.0, 0.5, 1.0]))

        # Test custom biome loading with custom transition JSON
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            biomes_dir = root / "biomes"
            biomes_dir.mkdir()
            (biomes_dir / "custom_beach.json").write_text(
                json.dumps({
                    "id": "custom_beach",
                    "surface": "sand",
                    "soil": "sand",
                    "ceiling": 0.35,
                    "inland": False,
                    "transition": {
                        "ceiling": 0.45,
                        "width": 0.20,
                        "curve": "cosine"
                    }
                }),
                encoding="utf-8",
            )
            (biomes_dir / "custom_land.json").write_text(
                json.dumps({
                    "id": "custom_land",
                    "surface": "grass",
                    "soil": "dirt",
                    "ceiling": 1.0,
                    "height_offset": 5.0,
                    "transition": {
                        "blend": 0.05,
                        "curve": "smootherstep"
                    }
                }),
                encoding="utf-8",
            )
            loaded = load_biomes(biomes_dir, {})
            self.assertEqual(len(loaded), 2)
            cb = loaded[0]
            cl = loaded[1]
            self.assertFalse(cb.inland)
            self.assertEqual(cb.transition_ceiling, 0.45)
            self.assertEqual(cb.transition_width, 0.20)
            self.assertEqual(cb.curve, "cosine")
            self.assertTrue(cl.inland)
            self.assertEqual(cl.blend, 0.05)
            self.assertEqual(cl.curve, "smootherstep")

    def test_biome_min_size_setting_and_wide_dry_beaches(self):
        # Every built-in biome has a positive min_size setting
        for b in BIOMES:
            self.assertGreater(b.min_size, 0.0, f"{b.id} should define a positive min_size")

        # Verify dry beaches (terrain_height >= WATER_LEVEL) are wide and not thin 1-3 block strips
        dry_beach_runs = []
        for z in range(-300, 301, 25):
            dry_len = 0
            for x in range(-300, 301):
                b = biome_at(x, z, 1, BIOMES).id
                h = terrain_height(x, z, 1, BIOMES)
                if b in ("beach", "gravel_beach", "cliff_beach") and h >= WATER_LEVEL:
                    dry_len += 1
                else:
                    if dry_len > 0:
                        dry_beach_runs.append(dry_len)
                        dry_len = 0
            if dry_len > 0:
                dry_beach_runs.append(dry_len)

        self.assertTrue(dry_beach_runs)
        self.assertGreaterEqual(float(np.median(dry_beach_runs)), 12.0)

    def test_biome_valid_spawn_and_world_spawn_relocation(self):
        from server.commands import CommandContext, CommandHandler
        from server.player import Player

        by_id = {b.id: b for b in BIOMES}
        self.assertFalse(by_id["ocean"].valid_spawn)
        self.assertFalse(by_id["beach"].valid_spawn)
        self.assertTrue(by_id["plains"].valid_spawn)
        self.assertTrue(by_id["forest"].valid_spawn)

        # Find a seed where (0, 0) is in an ocean biome
        ocean_seed = None
        for s in range(200):
            if biome_at(0, 0, s, BIOMES).id == "ocean":
                ocean_seed = s
                break
        self.assertIsNotNone(ocean_seed)

        world = World(seed=ocean_seed)
        sp_x, sp_y, sp_z = world.get_spawn_point()
        sx, sz = int(np.floor(sp_x)), int(np.floor(sp_z))
        spawn_biome = biome_at(sx, sz, ocean_seed, BIOMES)
        self.assertTrue(spawn_biome.valid_spawn, f"Spawn biome {spawn_biome.id} must be a valid spawn biome")
        self.assertGreater(terrain_height(sx, sz, ocean_seed, BIOMES), WATER_LEVEL)

        # Move world spawn via command /setworldspawn
        player = Player(world)
        player.pos = [120.5, 75.0, -45.5]
        handler = CommandHandler()
        ctx = CommandContext(sender="Player", player=player, world=world)
        handler.execute(ctx, "/setworldspawn")
        self.assertEqual(world.get_spawn_point(), (120.5, 75.0, -45.5))

        handler.execute(ctx, "/setworldspawn 200 80 300")
        self.assertEqual(world.get_spawn_point(), (200.0, 80.0, 300.0))
