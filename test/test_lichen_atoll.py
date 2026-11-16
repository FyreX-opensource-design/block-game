import unittest

import numpy as np

from server.biomes import BIOMES, all_biomes, biome_at
from server.blocks import BLOCKS, CHUNK_SIZE, WATER_LEVEL, block_index, can_place_block
from server.interactions import INTERACTIONS, tool_interaction_at
from server.items import ITEMS
from server.noise import terrain_height
from server.recipes import craft_match
from server.trees import DECORATORS
from server.world import World, _column_surface, _supports_tree, _tree_site, tree_blocks


def _find_atoll(seed=1, limit=480, step=8):
    grass = None
    beaches = []
    for z in range(-limit, limit, step):
        for x in range(-limit, limit, step):
            if biome_at(x, z, seed, BIOMES).id != "lichen_atoll":
                continue
            height = terrain_height(x, z, seed, BIOMES)
            if height >= WATER_LEVEL + 3 and grass is None:
                grass = (x, z, height)
            if height <= WATER_LEVEL + 2:
                beaches.append((x, z, height))
            if grass and len(beaches) >= 12:
                return grass, beaches
    return grass, beaches


class LichenAtollTests(unittest.TestCase):
    def test_atoll_is_an_ocean_island_of_fluorescent_lichen(self):
        by_id = {biome.id: biome for biome in all_biomes(BIOMES)}
        atoll = by_id["lichen_atoll"]
        self.assertEqual(atoll.variant_of, "ocean")
        self.assertFalse(atoll.inland)
        self.assertFalse(atoll.valid_spawn)
        self.assertEqual(atoll.surface, "lichen_grass")
        self.assertEqual(atoll.soil, "lichen_soil")
        self.assertIsNotNone(atoll.island)
        self.assertEqual(atoll.trees[0].id, "lichen_mushroom")
        self.assertEqual(atoll.trees[0].shape, "mushroom")

        for block_id, texture in (
            ("lichen_grass", "lichen_grass_glow"),
            ("lichen_stem", "lichen_stem_glow"),
            ("lichen_cap", "lichen_cap_glow"),
            ("lichen_planks", "lichen_planks_glow"),
        ):
            fluor = BLOCKS[block_id].fluorescence
            self.assertEqual(fluor.texture, texture)
            self.assertGreater(fluor.full, fluor.start)

        grass, beaches = _find_atoll()
        self.assertIsNotNone(grass, "the atoll should keep a grassy interior")
        self.assertTrue(beaches, "the atoll should slope down to a beach")
        x, z, height = grass
        world = World(seed=1, caves=[])
        world.load_chunk(x // 16, height // 16, z // 16)
        self.assertEqual(world.get_block(x, height, z), "lichen_grass")
        self.assertEqual(world.get_block(x, height - 1, z), "lichen_soil")

        found_sand = False
        for bx, bz, beach_h in beaches:
            world.load_chunk(bx // 16, beach_h // 16, bz // 16)
            world.load_chunk(bx // 16, (beach_h - 4) // 16, bz // 16)
            if world.get_block(bx, beach_h - 4, bz) != "lichen_sandstone":
                continue
            self.assertEqual(world.get_block(bx, beach_h, bz), "lichen_sand")
            self.assertEqual(world.get_block(bx, beach_h - 1, bz), "lichen_sand")
            found_sand = True
            break
        self.assertTrue(found_sand, "the beach should be lichen sand over lichen sandstone")
        sand = BLOCKS["lichen_sand"].fluorescence
        stone = BLOCKS["lichen_sandstone"].fluorescence
        self.assertEqual(sand.texture, "lichen_sand_glow")
        self.assertGreater(sand.start, 8)
        self.assertLess(stone.strength, sand.strength)

        steep = 0
        low = high = None
        for dz in range(-24, 25):
            for dx in range(-24, 25):
                here_id = biome_at(x + dx, z + dz, 1, BIOMES).id
                east_id = biome_at(x + dx + 1, z + dz, 1, BIOMES).id
                if here_id != "lichen_atoll" and east_id != "lichen_atoll":
                    continue
                here = terrain_height(x + dx, z + dz, 1, BIOMES)
                east = terrain_height(x + dx + 1, z + dz, 1, BIOMES)
                steep = max(steep, abs(here - east))
                if here_id == "lichen_atoll":
                    low = here if low is None else min(low, here)
                    high = here if high is None else max(high, here)
        self.assertLessEqual(steep, 3)
        self.assertGreaterEqual(high - low, 3)

        ocean = None
        for dz in range(-80, 81, 8):
            for dx in range(-80, 81, 8):
                if biome_at(x + dx, z + dz, 1, BIOMES).id == "ocean":
                    ocean = (x + dx, z + dz)
                    break
            if ocean:
                break
        self.assertIsNotNone(ocean)
        self.assertLessEqual(terrain_height(ocean[0], ocean[1], 1, BIOMES), 60)

    def test_mushrooms_do_not_leave_a_cap_on_the_beach(self):
        grass, beaches = _find_atoll()
        self.assertIsNotNone(grass)
        self.assertTrue(beaches)
        tree = next(item for item in DECORATORS if item.id == "lichen_mushroom")
        bx, bz, beach_h = beaches[0]
        self.assertEqual(_column_surface(biome_at(bx, bz, 1, BIOMES), beach_h), "lichen_sand")
        self.assertFalse(_supports_tree(tree, "lichen_sand"))
        self.assertIsNone(
            _tree_site(bx, bz, beach_h, 1, BIOMES, lambda px, pz: terrain_height(px, pz, 1))
        )

        world = World(seed=1, caves=[])
        y0 = (beach_h // CHUNK_SIZE) * CHUNK_SIZE
        overhang = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
        world._place_tree(overhang, -1, 4, beach_h, bx, bz, y0, tree)
        self.assertEqual(int(overhang.sum()), 0)

        grown = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
        grown[2, 4, 4] = block_index("lichen_grass")
        world._place_tree(grown, 4, 4, 66, grass[0], grass[1], 64, tree)
        self.assertIn(block_index("lichen_stem"), grown)
        self.assertIn(block_index("lichen_cap"), grown)

    def test_lichen_spreads_in_uv_and_the_depths_hold_sap(self):
        from server.lichen import spread_targets
        from server.mobs import MOBS, spawn_allowed
        from server.recipes import craft_match
        from server.underground_biomes import load_underground_biomes

        world = World(seed=1, caves=[])
        world.load_chunk(0, 0, 0)
        for yy in range(9, 14):
            for zz in range(3, 8):
                for xx in range(3, 8):
                    world.set_block(xx, yy, zz, "air")
        world.set_block(4, 10, 4, "lichen_grass")
        world.set_block(5, 10, 4, "lichen_grass")
        world.set_block(4, 11, 4, "fluor_lichen_v32")
        self.assertEqual(spread_targets(world, 4, 11, 4), [])
        world.set_block(6, 11, 4, "lichen_lamp")
        self.assertTrue(spread_targets(world, 4, 11, 4))

        grass, _beaches = _find_atoll()
        gx, gz, gh = grass
        world.load_chunk(gx // 16, 1, gz // 16)
        self.assertEqual(world.get_block(gx, 20, gz), "lichen_stone")
        found_sap = False
        for lz in range(16):
            for lx in range(16):
                for ly in range(16):
                    if world.get_block((gx // 16) * 16 + lx, 16 + ly, (gz // 16) * 16 + lz) == "fluor_amber":
                        found_sap = True
                        break
        self.assertTrue(found_sap, "lichen sap should vein through the stone under the atoll")
        depths = next(biome for biome in load_underground_biomes() if biome.id == "lichen_depths")
        self.assertTrue(depths.always)
        self.assertIn("lichen_atoll", depths.surface_biomes)

        fly = MOBS["lichenete"]
        self.assertTrue(fly.can_fly)
        self.assertEqual(fly.radius, 0.45)
        self.assertLessEqual(fly.height, 1.0)
        self.assertEqual(fly.spawn.weight_for("lichen_atoll"), 8)
        self.assertEqual(fly.spawn.weight_for("plains"), 0)
        self.assertIn("retaliate", [goal["type"] for goal in fly.goals])
        world.set_block(6, 11, 4, "air")
        world.set_block(4, 10, 4, "lichen_grass")
        self.assertFalse(spawn_allowed(fly, world, 4, 10, 4))
        world.set_block(5, 11, 4, "lichen_lamp")
        self.assertTrue(spawn_allowed(fly, world, 4, 10, 4))
        world.set_block(4, 10, 4, "grass")
        self.assertFalse(spawn_allowed(fly, world, 4, 10, 4))

        mob = world.spawn_mob("lichenete", pos=[4.5, 14.0, 4.5])
        retaliate = next(goal for _priority, goal in mob.goals.goals if type(goal).__name__ == "RetaliateGoal")
        self.assertFalse(retaliate.can_start())

        class Attacker:
            alive = True
            dead = False
            survival = True
            pos = [6.0, 14.0, 4.5]

        creative = Attacker()
        creative.survival = False
        mob.take_damage(1, attacker=creative)
        self.assertFalse(retaliate.can_start())
        mob.hurt_time = 0.0

        mob.take_damage(1, attacker=Attacker())
        self.assertTrue(retaliate.can_start())
        mob.update(0.2, world)
        self.assertGreater(mob.pos[1], 12.0)

        torch = craft_match(("lichenete_wing", "", "", "stick", "", "", "", "", ""))
        self.assertEqual(torch[0], "lichen_torch")
        self.assertEqual(torch[1], 4)
        lamp = craft_match(("fluor_amber", "", "", "lichenete_eye", "", "", "fluor_amber", "", ""))
        self.assertEqual(lamp[0], "lichen_lamp")
        lantern = craft_match(("lichen_planks", "", "", "lichenete_eye", "", "", "lichen_planks", "", ""))
        self.assertEqual(lantern[0], "lichen_lantern")

    def test_mushrooms_are_the_wood_and_the_crop_needs_tilled_lichen_soil(self):
        tree = next(item for item in DECORATORS if item.id == "lichen_mushroom")
        placed = tree_blocks(tree, 4, 70, 4, 3)
        kinds = {block_id for _x, _y, _z, block_id, _replace in placed}
        self.assertIn("lichen_stem", kinds)
        self.assertIn("lichen_cap", kinds)
        self.assertGreater(sum(1 for _x, _y, _z, block_id, _r in placed if block_id == "lichen_cap"), 8)

        made = craft_match(("lichen_stem", "", "", "", "", "", "", "", ""))
        self.assertIsNotNone(made)
        self.assertEqual(made[0], "lichen_planks")
        self.assertEqual(made[1], 4)

        cup = BLOCKS["glow_cup"].emission
        frond = BLOCKS["spore_frond"].emission
        self.assertGreater(cup[3], cup[2])
        self.assertGreater(cup[2], cup[1])
        self.assertGreater(frond[3], 0)

        ripe = BLOCKS["glow_bulb_3"]
        young = BLOCKS["glow_bulb_0"]
        self.assertEqual(ripe.valid_ground, ("lichen_farmland",))
        self.assertEqual(ripe.fluorescence.texture, "glow_bulb_glow")
        self.assertEqual(ripe.emission[3], 12)
        self.assertGreater(young.emission[3], 0)
        self.assertLess(young.emission[3], ripe.emission[3])
        self.assertGreater(BLOCKS["wild_glow_bulb"].emission[3], 0)

        world = World(seed=3, caves=[])
        world.load_chunk(0, 5, 0)
        world.chunks[(0, 5, 0)].data.fill(0)
        world.set_block(2, 80, 4, "lichen_grass")
        world.set_block(3, 80, 4, "farmland")
        world.set_block(4, 80, 4, "lichen_farmland")
        world.set_block(5, 80, 4, "dirt")
        hoe = ITEMS["wood_hoe"]
        tilled = tool_interaction_at(world, hoe, 2, 80, 4)
        self.assertEqual(tilled[3].id, "till_lichen_soil")
        self.assertEqual(tilled[3].result, "lichen_farmland")
        dirt = tool_interaction_at(world, hoe, 5, 80, 4)
        self.assertEqual(dirt[3].result, "farmland")

        self.assertTrue(can_place_block(world, 4, 81, 4, "glow_bulb_0"))
        self.assertFalse(can_place_block(world, 3, 81, 4, "glow_bulb_0"))
        self.assertTrue(can_place_block(world, 3, 81, 4, "wheat_0"))
        self.assertFalse(can_place_block(world, 4, 81, 4, "wheat_0"))
        self.assertIn("till_lichen_soil", {rule.id for rule in INTERACTIONS})

    def test_lichenete_wanders_up_and_down(self):
        from server.ai import WanderGoal, pick_fly_altitude
        from server.entity import AssetMob

        class Flat:
            def __init__(self):
                self.entities = []
                self.ceiling = None

            def get_block(self, x, y, z):
                return "stone" if self.is_solid(x, y, z) else "air"

            def is_solid(self, x, y, z):
                return y == 4 or (self.ceiling is not None and y == self.ceiling)

        flat = Flat()
        mob = AssetMob("lichenete", pos=[0.5, 9.0, 0.5])
        mob.world = flat
        surface = 5.0

        highs = []
        for _ in range(24):
            highs.append(pick_fly_altitude(mob, flat) - surface)
        self.assertLess(min(highs), 2.0)
        self.assertGreater(max(highs), 5.0)

        flat.ceiling = 9
        mob.pos[1] = 7.0
        for _ in range(16):
            chosen = pick_fly_altitude(mob, flat)
            self.assertLess(chosen + mob.height, flat.ceiling + 0.05)
        flat.ceiling = None
        mob.pos[1] = 9.0

        mob.goals = None
        mob.pos[1] = 7.0
        mob.fly_target_y = 12.0
        for _ in range(16):
            mob.update(0.05, flat)
        self.assertGreater(mob.pos[1], 9.0)
        self.assertGreater(mob.pitch, 0.0)

        mob.fly_target_y = 6.0
        for _ in range(16):
            mob.update(0.05, flat)
        self.assertLess(mob.pos[1], 9.0)
        self.assertLess(mob.pitch, 0.0)

        wander_mob = AssetMob("lichenete", pos=[0.5, 9.0, 0.5])
        wander_mob.world = flat
        wander = next(goal for _priority, goal in wander_mob.goals.goals if isinstance(goal, WanderGoal))
        cruises = []
        for _ in range(8):
            wander._new_cruise()
            cruises.append(wander._cruise_y)
        self.assertGreater(max(cruises) - min(cruises), 2.5)
        wander_mob.update(0.05, flat)
        above = wander_mob.fly_target_y - surface
        self.assertGreaterEqual(above, 0.9)
        self.assertLessEqual(above, 8.0)

        class Attacker:
            alive = True
            survival = True
            pos = [0.5, 7.0, 4.0]

        wander_mob.aggro = Attacker()
        wander_mob.update(0.05, flat)
        self.assertAlmostEqual(wander_mob.fly_target_y, 7.3, places=2)
