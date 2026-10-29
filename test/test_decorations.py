import json
import tempfile
import unittest
from pathlib import Path

from server.biomes import BIOMES, biome_at, load_biomes
from server.blocks import AIR, BLOCKS, GRASS, block_from_index, block_index
from server.decorations import DECORATIONS, DecorationFeature, load_decorations
from server.items import ITEMS
from server.loot import get_block_drops
from server.world import World, terrain_height


class DecorationFeatureTests(unittest.TestCase):
    def test_default_decorations_loaded(self):
        dec_map = {dec.id: dec for dec in DECORATIONS}
        self.assertIn("tall_grass", dec_map)
        self.assertIn("fern", dec_map)
        self.assertIn("dandelion", dec_map)
        self.assertIn("rose", dec_map)

        grass_dec = dec_map["tall_grass"]
        self.assertEqual(grass_dec.block, "tall_grass")
        self.assertGreater(grass_dec.chance, 0.0)
        self.assertIn("grass", grass_dec.surfaces)

        fern_dec = dec_map["fern"]
        self.assertEqual(fern_dec.block, "fern")
        self.assertGreater(fern_dec.chance, 0.0)
        self.assertIn("grass", fern_dec.surfaces)

    def test_plant_blocks_registered_correctly(self):
        for plant_id in ("tall_grass", "fern", "dandelion", "rose"):
            self.assertIn(plant_id, BLOCKS)
            block = BLOCKS[plant_id]
            self.assertFalse(block.solid, f"{plant_id} should be non-solid")
            self.assertFalse(block.full_cube, f"{plant_id} should not be a full cube")
            self.assertTrue(block.cutout, f"{plant_id} should use cutout rendering")
            self.assertEqual(block.hardness, 0.0)

            # Check corresponding item
            self.assertIn(plant_id, ITEMS)
            item = ITEMS[plant_id]
            self.assertEqual(item.block, plant_id)
            self.assertEqual(item.right, "place")
            self.assertEqual(item.left, "break")

    def test_biomes_have_expected_decorations(self):
        biome_map = {b.id: b for b in BIOMES}
        self.assertIn("plains", biome_map)
        self.assertIn("forest", biome_map)

        plains_decs = [d.id for d in biome_map["plains"].decorations]
        self.assertIn("tall_grass", plains_decs)
        self.assertIn("dandelion", plains_decs)
        self.assertIn("rose", plains_decs)

        forest_decs = [d.id for d in biome_map["forest"].decorations]
        self.assertIn("tall_grass", forest_decs)
        self.assertIn("fern", forest_decs)
        self.assertIn("dandelion", forest_decs)
        self.assertIn("rose", forest_decs)

        # Oceans and beaches should have no decorations
        self.assertEqual(len(biome_map["ocean"].decorations), 0)
        self.assertEqual(len(biome_map["beach"].decorations), 0)

    def test_custom_decorations_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            decs_dir = root / "decorations"
            biomes_dir = root / "biomes"
            decs_dir.mkdir()
            biomes_dir.mkdir()

            (decs_dir / "shrub.json").write_text(
                json.dumps(
                    {
                        "id": "shrub",
                        "block": "leaves",
                        "chance": 0.5,
                        "surfaces": ["grass", "dirt"],
                    }
                ),
                encoding="utf-8",
            )
            (biomes_dir / "meadow.json").write_text(
                json.dumps(
                    {
                        "id": "meadow",
                        "ceiling": 1.0,
                        "scale": 0.006,
                        "octaves": 2,
                        "seed_offset": 90,
                        "tree_chance": 0.0,
                        "trees": [],
                        "decorations": ["shrub"],
                    }
                ),
                encoding="utf-8",
            )

            loaded_decs = load_decorations(decs_dir)
            self.assertEqual(len(loaded_decs), 1)
            self.assertEqual(loaded_decs[0].id, "shrub")

            loaded_biomes = load_biomes(biomes_dir, trees=(), decorations=loaded_decs)
            self.assertEqual(len(loaded_biomes), 1)
            self.assertEqual(len(loaded_biomes[0].decorations), 1)
            self.assertEqual(loaded_biomes[0].decorations[0].id, "shrub")

    def test_world_generation_places_decorations_on_grass(self):
        world = World(seed=42, caves=[])
        # Find a plains chunk and a forest chunk
        found_plains = False
        found_forest = False
        decorations_found = set()

        for cz in range(-5, 6):
            for cx in range(-5, 6):
                wx = cx * 16 + 8
                wz = cz * 16 + 8
                b = biome_at(wx, wz, world.seed, world.biomes)
                if b.id not in ("plains", "forest"):
                    continue
                if b.id == "plains":
                    found_plains = True
                if b.id == "forest":
                    found_forest = True

                world.generate_column(cx, cz)
                # Inspect blocks on top of grass in loaded chunks of this column
                for lx in range(16):
                    for lz in range(16):
                        col_x = cx * 16 + lx
                        col_z = cz * 16 + lz
                        h = terrain_height(col_x, col_z, world.seed)
                        surf_block = world.get_block(col_x, h, col_z)
                        above_block = world.get_block(col_x, h + 1, col_z)
                        if above_block in ("tall_grass", "fern", "dandelion", "rose"):
                            decorations_found.add(above_block)
                            # Plant must be sitting on grass
                            self.assertEqual(surf_block, GRASS)

        self.assertTrue(found_plains)
        self.assertTrue(found_forest)
        self.assertIn("tall_grass", decorations_found)
        self.assertTrue(
            any(f in decorations_found for f in ("dandelion", "rose", "fern")),
            f"Expected flowers or ferns, got: {decorations_found}",
        )

    def test_raycast_targeting_and_solid_only(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(10, 64, 10)
        world.load_chunk(cx, cy, cz)
        world.set_block(10, 64, 10, GRASS)
        world.set_block(10, 65, 10, "rose")

        # Raycast targeting block: with solid_only=False, should hit the rose
        hit = world.raycast(10.5, 65.5, 5.0, 0, 0, 1, 10, solid_only=False)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["x"], 10)
        self.assertEqual(hit["y"], 65)
        self.assertEqual(hit["z"], 10)

        # Raycast for camera collision: with solid_only=True, should pass through rose and hit grass
        # Ray from above:
        hit_solid = world.raycast(10.5, 68.0, 10.5, 0, -1, 0, 10, solid_only=True)
        self.assertIsNotNone(hit_solid)
        self.assertEqual(hit_solid["y"], 64)  # Hits the solid grass, ignores the rose!

    def test_breaking_decorations_drops_item(self):
        for plant_id in ("tall_grass", "fern", "dandelion", "rose"):
            drops = get_block_drops(plant_id, None)
            self.assertEqual(drops, [(plant_id, 1)])

    def test_decorations_do_not_block_movement(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(0, 64, 0)
        world.load_chunk(cx, cy, cz)
        world.set_block(0, 64, 0, GRASS)
        world.set_block(0, 65, 0, "tall_grass")
        self.assertTrue(world.is_solid(0, 64, 0))
        self.assertFalse(world.is_solid(0, 65, 0))
        self.assertFalse(world.blocks_movement(0, 65, 0))

    def test_cannot_place_plants_on_non_soil_or_on_each_other(self):
        from server.blocks import can_place_block, is_soil

        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(5, 64, 5)
        world.load_chunk(cx, cy, cz)

        # Soil check
        self.assertTrue(is_soil("dirt"))
        self.assertTrue(is_soil("grass"))
        self.assertTrue(is_soil("mud"))
        self.assertTrue(is_soil("muddy_grass"))
        self.assertFalse(is_soil("stone"))
        self.assertFalse(is_soil("sand"))
        self.assertFalse(is_soil("tall_grass"))

        plants = ("tall_grass", "fern", "dandelion", "rose")
        non_soils = ("stone", "cobble", "sand", "gravel", "planks", "leaves")

        for plant in plants:
            for non_soil in non_soils:
                world.set_block(5, 64, 5, non_soil)
                hit = {"x": 5, "y": 64, "z": 5, "nx": 0, "ny": 1, "nz": 0}
                # Attempting to place on non-soil block must be rejected
                self.assertFalse(
                    can_place_block(world, 5, 65, 5, plant, hit=hit),
                    f"Should not allow placing {plant} on {non_soil}",
                )

            # Test placing plant on top of another plant (stacking)
            world.set_block(5, 64, 5, "grass")
            world.set_block(5, 65, 5, "tall_grass")
            hit_on_plant = {"x": 5, "y": 65, "z": 5, "nx": 0, "ny": 1, "nz": 0}
            self.assertFalse(
                can_place_block(world, 5, 66, 5, plant, hit=hit_on_plant),
                f"Should not allow placing {plant} on top of tall_grass",
            )

    def test_can_place_plants_on_all_soil_types(self):
        from server.blocks import can_place_block

        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(5, 64, 5)
        world.load_chunk(cx, cy, cz)

        soils = ("dirt", "grass", "mud", "muddy_grass")
        plants = ("tall_grass", "fern", "dandelion", "rose")

        for soil in soils:
            for plant in plants:
                world.set_block(5, 64, 5, soil)
                world.set_block(5, 65, 5, AIR)
                hit = {"x": 5, "y": 64, "z": 5, "nx": 0, "ny": 1, "nz": 0}
                self.assertTrue(
                    can_place_block(world, 5, 65, 5, plant, hit=hit),
                    f"Should allow placing {plant} on {soil}",
                )

    def test_cannot_place_plants_on_side_or_in_water(self):
        from server.blocks import can_place_block

        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(5, 64, 5)
        world.load_chunk(cx, cy, cz)
        world.set_block(5, 64, 5, "grass")

        # Side face click: ny == 0
        hit_side = {"x": 5, "y": 64, "z": 5, "nx": 1, "ny": 0, "nz": 0}
        self.assertFalse(can_place_block(world, 6, 64, 5, "rose", hit=hit_side))

        # Trying to place into water
        world.set_block(5, 65, 5, "water")
        hit_top = {"x": 5, "y": 64, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        self.assertFalse(can_place_block(world, 5, 65, 5, "rose", hit=hit_top))

    def test_breaking_ground_under_plant_breaks_plant_and_drops_loot(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(5, 64, 5)
        world.load_chunk(cx, cy, cz)

        # Place grass floor and a rose above it
        world.set_block(5, 64, 5, "grass")
        world.set_block(5, 65, 5, "rose")
        self.assertEqual(world.get_block(5, 65, 5), "rose")

        initial_entities = len(world.entities)
        # Break the grass block underneath (set to air)
        world.set_block(5, 64, 5, AIR)

        # The rose at y=65 must be broken automatically to AIR and drop the item
        self.assertEqual(world.get_block(5, 65, 5), AIR)
        self.assertEqual(len(world.entities), initial_entities + 1)
        dropped = world.entities[-1]
        self.assertEqual(dropped.item, "rose")

    def test_server_rejects_invalid_plant_set_block_packet(self):
        from server.network import MultiplayerServer, RemoteClient
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(saves_dir=tmp, world_name="test_plant_net")
            world = World(seed=1)
            server.world = world
            cx, cy, cz = world.chunk_key(0, 64, 0)
            world.load_chunk(cx, cy, cz)
            world.set_block(0, 64, 0, "stone")

            # Client tries to send set_block for tall_grass on stone
            client = RemoteClient(None, ("127.0.0.1", 12345))
            server._handle_packet(client, {"type": "set_block", "x": 0, "y": 65, "z": 0, "block": "tall_grass"})

            # Block was not set because stone cannot sustain tall grass
            self.assertEqual(world.get_block(0, 65, 0), AIR)


if __name__ == "__main__":
    unittest.main()
