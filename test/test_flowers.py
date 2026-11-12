import unittest

from server.biomes import BIOMES, all_biomes, biome_at
from server.blocks import (
    AIR,
    BLOCKS,
    CREATIVE,
    DIRT,
    GRASS,
    STONE,
    WORLD_HEIGHT,
    can_place_block,
    get_tall_plant_info,
    is_tall_plant,
)
from server.items import ITEMS
from server.recipes import craft_match
from server.tags import has_block_tag, has_item_tag
from server.world import Chunk, World, terrain_height

GREYSCALE_DYES = {"white_dye", "light_gray_dye", "gray_dye", "black_dye"}

ONE_TALL_FLOWERS = {
    "rose": "red_dye",
    "dandelion": "yellow_dye",
    "orange_tulip": "orange_dye",
    "cornflower": "blue_dye",
    "blue_orchid": "light_blue_dye",
    "sea_holly": "cyan_dye",
    "lime_blossom": "lime_dye",
    "green_carnation": "green_dye",
    "chocolate_cosmos": "brown_dye",
    "marigold": "amber_dye",
    "aqua_iris": "aquamarine_dye",
    "pine_lily": "conifer_dye",
    "lavender": "lavender_dye",
    "mint_blossom": "mint_dye",
    "peach_poppy": "peach_dye",
    "vernal_tulip": "spring_green_dye",
    "cobalt_bellflower": "ultramarine_dye",
    "burgundy_dahlia": "wine_dye",
}

TWO_TALL_FLOWERS = {
    "peony": "pink_dye",
    "lilac": "magenta_dye",
    "delphinium": "purple_dye",
    "sweet_pea": "bubblegum_dye",
    "sunburst_lupine": "fluorescent_dye",
    "sunflower": "honey_dye",
    "crimson_hollyhock": "maroon_dye",
    "indigo_spire": "navy_dye",
    "tiger_lily": "persimmon_dye",
    "rose_bush": "rose_dye",
}


def _flat_world(seed=101, ground=GRASS):
    world = World(seed=seed)
    chunk = Chunk(0, 0, 0)
    top_chunk = Chunk(0, 1, 0)
    chunk.data[:, :, :] = 0
    top_chunk.data[:, :, :] = 0
    world.chunks[(0, 0, 0)] = chunk
    world.chunks[(0, 1, 0)] = top_chunk
    for z in range(16):
        for x in range(16):
            world.set_block(x, 0, z, STONE, simulate=False)
            world.set_block(x, 4, z, ground, simulate=False)
    return world


class FlowerTests(unittest.TestCase):
    def test_all_non_greyscale_dyes_have_flowers_and_recipes(self):
        dyes = [item_id for item_id in ITEMS if item_id.endswith("_dye")]
        self.assertEqual(len(dyes), 32)
        covered_dyes = set()

        for flower_id, dye_id in ONE_TALL_FLOWERS.items():
            self.assertIn(flower_id, BLOCKS)
            self.assertIn(flower_id, ITEMS)
            self.assertIn(flower_id, CREATIVE)
            self.assertFalse(is_tall_plant(flower_id))
            self.assertTrue(has_block_tag(flower_id, "flowers"))
            self.assertTrue(has_item_tag(flower_id, "flowers"))
            res = craft_match((flower_id, "", "", ""))
            self.assertIsNotNone(res, f"Expected crafting recipe for {flower_id}")
            self.assertEqual(res[0], dye_id)
            self.assertEqual(res[1], 1)
            covered_dyes.add(dye_id)

        for flower_id, dye_id in TWO_TALL_FLOWERS.items():
            self.assertIn(flower_id, BLOCKS)
            self.assertIn(f"{flower_id}_top", BLOCKS)
            self.assertIn(flower_id, ITEMS)
            self.assertIn(flower_id, CREATIVE)
            self.assertNotIn(f"{flower_id}_top", CREATIVE)
            self.assertTrue(is_tall_plant(flower_id))
            self.assertTrue(is_tall_plant(f"{flower_id}_top"))
            self.assertEqual(get_tall_plant_info(flower_id)["half"], "lower")
            self.assertEqual(get_tall_plant_info(f"{flower_id}_top")["half"], "upper")
            self.assertEqual(get_tall_plant_info(f"{flower_id}_top")["base_id"], flower_id)
            self.assertTrue(has_block_tag(flower_id, "flowers"))
            self.assertTrue(has_block_tag(f"{flower_id}_top", "flowers"))
            self.assertTrue(has_item_tag(flower_id, "flowers"))
            res = craft_match((flower_id, "", "", ""))
            self.assertIsNotNone(res, f"Expected crafting recipe for {flower_id}")
            self.assertEqual(res[0], dye_id)
            self.assertEqual(res[1], 2)
            covered_dyes.add(dye_id)

        self.assertEqual(covered_dyes, set(dyes) - GREYSCALE_DYES)

    def test_tall_plant_placement_support_and_breaking(self):
        world = _flat_world()
        hit_top = {"x": 5, "y": 4, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        self.assertTrue(can_place_block(world, 5, 5, 5, "peony", hit=hit_top))

        # Cannot place if headroom at y+1 is occupied
        world.set_block(6, 6, 6, STONE, simulate=False)
        self.assertFalse(
            can_place_block(
                world,
                6,
                5,
                6,
                "peony",
                hit={"x": 6, "y": 4, "z": 6, "nx": 0, "ny": 1, "nz": 0},
            )
        )
        # Cannot place at top world limit
        self.assertFalse(can_place_block(world, 7, WORLD_HEIGHT - 1, 7, "peony"))

        # Placing lower half places both halves
        self.assertTrue(world.set_block(5, 5, 5, "peony"))
        self.assertEqual(world.get_block(5, 5, 5), "peony")
        self.assertEqual(world.get_block(5, 6, 5), "peony_top")

        # Breaking upper half clears lower half cleanly
        world.set_block(5, 6, 5, AIR)
        self.assertEqual(world.get_block(5, 5, 5), AIR)
        self.assertEqual(world.get_block(5, 6, 5), AIR)

        # Breaking soil underneath clears both halves and spawns 1 drop
        world.entities.clear()
        world.set_block(4, 5, 4, "sunflower")
        self.assertEqual(world.get_block(4, 6, 4), "sunflower_top")
        world.set_block(4, 4, 4, AIR)
        self.assertEqual(world.get_block(4, 5, 4), AIR)
        self.assertEqual(world.get_block(4, 6, 4), AIR)
        self.assertEqual(sum(it.count for it in world.entities if getattr(it, "item", None) == "sunflower"), 1)

        # Fluid replacing either half cleans partner and spawns 1 drop
        world.set_block(3, 4, 3, DIRT, simulate=False)
        world.entities.clear()
        world.set_block(3, 5, 3, "lilac")
        self.assertEqual(world.get_block(3, 6, 3), "lilac_top")
        world.set_block(3, 6, 3, "water")
        self.assertEqual(world.get_block(3, 5, 3), AIR)
        self.assertEqual(sum(it.count for it in world.entities if getattr(it, "item", None) == "lilac"), 1)

    def test_flower_patch_biomes_exist_and_generate_complete_flowers(self):
        expected_patches = {
            "wildflower_patch": "plains",
            "sunburst_glade": "lowland_plains",
            "rose_thicket": "plains",
            "lavender_hollow": "highland_plains",
            "orchid_clearing": "highland_plains",
            "mint_glade": "forest",
            "autumn_blossom_patch": "pine_forest",
            "alpine_flower_patch": "mountain_meadow",
        }
        by_id = {b.id: b for b in all_biomes(BIOMES)}
        for patch_id, parent_id in expected_patches.items():
            self.assertIn(patch_id, by_id)
            self.assertEqual(by_id[patch_id].variant_of, parent_id)

        # Verify patch biomes appear across deterministic world coordinates and generate flowers
        seen_biomes = set()
        patch_sample_coords = {}
        for z in range(-480, 481, 8):
            for x in range(-480, 481, 8):
                b = biome_at(x, z, 7, BIOMES)
                seen_biomes.add(b.id)
                if b.id in expected_patches and b.id not in patch_sample_coords:
                    patch_sample_coords[b.id] = (x, z)
            if len(patch_sample_coords) == len(expected_patches):
                break

        self.assertEqual(set(patch_sample_coords.keys()), set(expected_patches.keys()))

        world = World(seed=7)
        tall_lowers_seen = 0
        for patch_id, (px, pz) in patch_sample_coords.items():
            cx, cz = px // 16, pz // 16
            top_y = terrain_height(px, pz, world.seed)
            cy = (top_y + 1) // 16
            for dcy in range(max(0, cy - 1), cy + 2):
                world.generate_chunk(cx, dcy, cz)
            # Check columns inside chunk (cx, cz) belonging to this patch biome have valid tall plants
            for lz in range(16):
                for lx in range(16):
                    wx = cx * 16 + lx
                    wz = cz * 16 + lz
                    gy = terrain_height(wx, wz, world.seed)
                    blk = world.get_block(wx, gy + 1, wz)
                    above = world.get_block(wx, gy + 2, wz)
                    tp = get_tall_plant_info(blk)
                    if tp is not None:
                        self.assertEqual(tp["half"], "lower")
                        self.assertEqual(above, tp["upper"], f"Missing upper half at ({wx}, {gy+2}, {wz}) for {blk}")
                        tall_lowers_seen += 1
                    above_tp = get_tall_plant_info(above)
                    if above_tp is not None:
                        self.assertEqual(above_tp["half"], "upper")
                        self.assertEqual(blk, above_tp["lower"], f"Orphaned upper half {above} above {blk}")
        self.assertGreater(tall_lowers_seen, 0)

    def test_tall_plant_chunk_seam_is_consistent_in_any_load_order(self):
        # Seed 1 chunk (-4, 0) has multiple two-tall flowers crossing y=79/80 seam,
        # and chunk (4, 3) has a tree canopy above a tall-flower spot on y=79/80.
        test_coords = [(-4, 0), (4, 3)]
        for mode in ("top_down", "bottom_up", "worker_reverse_commit"):
            world = World(seed=1)
            for cx, cz in test_coords:
                if mode == "top_down":
                    world.generate_chunk(cx, 5, cz)
                    world.generate_chunk(cx, 4, cz)
                elif mode == "bottom_up":
                    world.generate_chunk(cx, 4, cz)
                    world.generate_chunk(cx, 5, cz)
                else:
                    c5 = world.generate_chunk(cx, 5, cz, remember=False)
                    c4 = world.generate_chunk(cx, 4, cz, remember=False)
                    world.commit_chunk(c5)
                    world.commit_chunk(c4)

            tall_on_seam = 0
            for cx, cz in test_coords:
                x0, z0 = cx * 16, cz * 16
                for ly in range(64, 96):
                    for lz in range(16):
                        for lx in range(16):
                            wx, wz = x0 + lx, z0 + lz
                            blk = world.get_block(wx, ly, wz)
                            tp = get_tall_plant_info(blk)
                            if tp is None:
                                continue
                            if tp["half"] == "lower":
                                self.assertEqual(
                                    world.get_block(wx, ly + 1, wz),
                                    tp["upper"],
                                    f"Orphaned lower {blk} in mode={mode} at ({wx}, {ly}, {wz})",
                                )
                                if ly == 79:
                                    tall_on_seam += 1
                            else:
                                self.assertEqual(
                                    world.get_block(wx, ly - 1, wz),
                                    tp["lower"],
                                    f"Orphaned upper {blk} in mode={mode} at ({wx}, {ly}, {wz})",
                                )
            self.assertGreater(tall_on_seam, 0, f"Expected cross-seam tall plants in mode={mode}")
