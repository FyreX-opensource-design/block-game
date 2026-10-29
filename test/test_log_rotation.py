import unittest

from server.blocks import (
    BLOCKS,
    FACE_INDEX,
    LOG,
    canonical_id,
    placement_block,
    rotation_variant,
)
from server.items import canonical_item
from server.mesh import build_chunk_mesh
from server.recipes import craft_match
from server.world import Chunk, World


class LogRotationTests(unittest.TestCase):
    def test_log_has_axis_rotation_property(self):
        log_y = BLOCKS.get("log")
        self.assertIsNotNone(log_y)
        self.assertEqual(log_y.properties.get("rotation"), "axis")
        self.assertEqual(log_y.properties.get("axis"), "y")
        self.assertEqual(log_y.base_id, "log")
        self.assertFalse(log_y.is_variant)

        log_x = BLOCKS.get("log_x")
        self.assertIsNotNone(log_x)
        self.assertEqual(log_x.properties.get("rotation"), "axis")
        self.assertEqual(log_x.properties.get("axis"), "x")
        self.assertEqual(log_x.base_id, "log")
        self.assertTrue(log_x.is_variant)

        log_z = BLOCKS.get("log_z")
        self.assertIsNotNone(log_z)
        self.assertEqual(log_z.properties.get("rotation"), "axis")
        self.assertEqual(log_z.properties.get("axis"), "z")
        self.assertEqual(log_z.base_id, "log")
        self.assertTrue(log_z.is_variant)

    def test_placement_block_direction(self):
        # Vertical placement on top or bottom face
        self.assertEqual(placement_block("log", (0, 1, 0)), "log")
        self.assertEqual(placement_block("log", (0, -1, 0)), "log")

        # Horizontal placement on East / West faces -> X axis
        self.assertEqual(placement_block("log", (1, 0, 0)), "log_x")
        self.assertEqual(placement_block("log", (-1, 0, 0)), "log_x")

        # Horizontal placement on South / North faces -> Z axis
        self.assertEqual(placement_block("log", (0, 0, 1)), "log_z")
        self.assertEqual(placement_block("log", (0, 0, -1)), "log_z")

        # Non-rotating block stays unchanged
        self.assertEqual(placement_block("stone", (1, 0, 0)), "stone")
        self.assertEqual(placement_block("dirt", (0, 0, 1)), "dirt")

    def test_rotation_variant_helper(self):
        self.assertEqual(rotation_variant("log", "y"), "log")
        self.assertEqual(rotation_variant("log", "x"), "log_x")
        self.assertEqual(rotation_variant("log", "z"), "log_z")
        self.assertEqual(rotation_variant("stone", "x"), "stone")

    def test_canonical_id_and_canonical_item(self):
        self.assertEqual(canonical_id("log"), "log")
        self.assertEqual(canonical_id("log_x"), "log_x")
        self.assertEqual(canonical_id("log_z"), "log_z")
        self.assertEqual(canonical_id("log[axis=x]"), "log_x")
        self.assertEqual(canonical_id("log[axis=z]"), "log_z")
        self.assertEqual(canonical_id("log[axis=y]"), "log")

        self.assertEqual(canonical_item("log"), "log")
        self.assertEqual(canonical_item("log_x"), "log")
        self.assertEqual(canonical_item("log_z"), "log")
        from server.items import ITEMS
        self.assertNotIn("log_x", ITEMS)
        self.assertNotIn("log_z", ITEMS)
        self.assertEqual(BLOCKS["log_x"].base_id, "log")
        self.assertEqual(BLOCKS["log_z"].base_id, "log")

    def test_face_uv_assignment_for_axes(self):
        log_y = BLOCKS["log"]
        log_x = BLOCKS["log_x"]
        log_z = BLOCKS["log_z"]

        up_uv = log_y.face_uvs[FACE_INDEX["up"]]
        east_uv = log_y.face_uvs[FACE_INDEX["east"]]

        # For Y-axis: rings are up/down, bark is east/west/north/south
        self.assertEqual(log_y.face_uvs[FACE_INDEX["up"]], up_uv)
        self.assertEqual(log_y.face_uvs[FACE_INDEX["down"]], log_y.face_uvs[FACE_INDEX["down"]])
        self.assertEqual(log_y.face_uvs[FACE_INDEX["east"]], east_uv)

        # For X-axis: rings are east/west, bark on up/down/south/north
        self.assertEqual(log_x.face_uvs[FACE_INDEX["east"]], up_uv)
        self.assertEqual(log_x.face_uvs[FACE_INDEX["west"]], up_uv)
        self.assertNotEqual(log_x.face_uvs[FACE_INDEX["up"]], up_uv)

        # For Z-axis: rings are south/north, bark on up/down/east/west
        self.assertEqual(log_z.face_uvs[FACE_INDEX["south"]], up_uv)
        self.assertEqual(log_z.face_uvs[FACE_INDEX["north"]], up_uv)
        self.assertNotEqual(log_z.face_uvs[FACE_INDEX["up"]], up_uv)

    def test_rotated_log_meshing(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        from server.blocks import block_index
        chunk.data[1, 1, 1] = block_index("log_x")
        chunk.data[2, 2, 2] = block_index("log_z")
        world.chunks[(0, 0, 0)] = chunk

        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh)
        # 2 isolated blocks with 6 faces each = 12 faces = 48 vertices
        self.assertEqual(len(mesh.vertices), 12 * 4)

    def test_rotated_logs_can_craft_planks(self):
        res_x = craft_match(("log_x", "", "", ""))
        self.assertIsNotNone(res_x)
        self.assertEqual(res_x[0], "planks")
        self.assertEqual(res_x[1], 4)

        res_z = craft_match(("log_z", "", "", ""))
        self.assertIsNotNone(res_z)
        self.assertEqual(res_z[0], "planks")
        self.assertEqual(res_z[1], 4)

    def test_furnace_has_horizontal_rotation_variants(self):
        furnace = BLOCKS.get("furnace")
        self.assertIsNotNone(furnace)
        self.assertEqual(furnace.properties.get("rotation"), "horizontal")
        self.assertEqual(furnace.properties.get("facing"), "south")
        self.assertEqual(furnace.base_id, "furnace")
        self.assertFalse(furnace.is_variant)

        for facing in ("north", "east", "west"):
            var = BLOCKS.get(f"furnace_{facing}")
            self.assertIsNotNone(var, f"Missing furnace variant for {facing}")
            self.assertEqual(var.properties.get("rotation"), "horizontal")
            self.assertEqual(var.properties.get("facing"), facing)
            self.assertEqual(var.base_id, "furnace")
            self.assertTrue(var.is_variant)

        # Canonical IDs
        self.assertEqual(canonical_id("furnace_south"), "furnace")
        self.assertEqual(canonical_id("furnace[facing=south]"), "furnace")
        self.assertEqual(canonical_id("furnace[facing=north]"), "furnace_north")
        self.assertEqual(rotation_variant("furnace", "south"), "furnace")
        self.assertEqual(rotation_variant("furnace", "north"), "furnace_north")

    def test_furnace_placement_block_directional(self):
        import math

        # When player looks North (yaw = 0), placed furnace faces South towards player
        self.assertEqual(placement_block("furnace", (0, 1, 0), yaw=0.0), "furnace")

        # When player looks South (yaw = pi), placed furnace faces North towards player
        self.assertEqual(placement_block("furnace", (0, 1, 0), yaw=math.pi), "furnace_north")

        # When player looks East (yaw = 3*pi/2), placed furnace faces West towards player
        self.assertEqual(placement_block("furnace", (0, 1, 0), yaw=3 * math.pi / 2), "furnace_west")

        # When player looks West (yaw = pi/2), placed furnace faces East towards player
        self.assertEqual(placement_block("furnace", (0, 1, 0), yaw=math.pi / 2), "furnace_east")

    def test_furnace_face_uv_assignment(self):
        furnace_south = BLOCKS["furnace"]
        furnace_north = BLOCKS["furnace_north"]
        furnace_east = BLOCKS["furnace_east"]
        furnace_west = BLOCKS["furnace_west"]

        front_uv = furnace_south.face_uvs[FACE_INDEX["south"]]
        side_uv = furnace_south.face_uvs[FACE_INDEX["east"]]
        top_uv = furnace_south.face_uvs[FACE_INDEX["up"]]
        bottom_uv = furnace_south.face_uvs[FACE_INDEX["down"]]

        # South-facing has front on South
        self.assertEqual(furnace_south.face_uvs[FACE_INDEX["south"]], front_uv)
        self.assertEqual(furnace_south.face_uvs[FACE_INDEX["north"]], side_uv)

        # North-facing has front on North
        self.assertEqual(furnace_north.face_uvs[FACE_INDEX["north"]], front_uv)
        self.assertEqual(furnace_north.face_uvs[FACE_INDEX["south"]], side_uv)
        self.assertEqual(furnace_north.face_uvs[FACE_INDEX["up"]], top_uv)
        self.assertEqual(furnace_north.face_uvs[FACE_INDEX["down"]], bottom_uv)

        # East-facing has front on East
        self.assertEqual(furnace_east.face_uvs[FACE_INDEX["east"]], front_uv)
        self.assertEqual(furnace_east.face_uvs[FACE_INDEX["west"]], side_uv)

        # West-facing has front on West
        self.assertEqual(furnace_west.face_uvs[FACE_INDEX["west"]], front_uv)
        self.assertEqual(furnace_west.face_uvs[FACE_INDEX["east"]], side_uv)

    def test_furnace_variants_drop_furnace_item(self):
        from server.items import ITEMS
        from server.loot import get_block_drops

        pickaxe = ITEMS.get("wood_pickaxe")
        for variant in ("furnace", "furnace_north", "furnace_east", "furnace_west"):
            self.assertEqual(canonical_item(variant), "furnace")
            # With pickaxe, drops furnace item
            drops = get_block_drops(variant, pickaxe)
            self.assertEqual(drops, [("furnace", 1)])
            # Without tool (bare hands), drops nothing
            bare_drops = get_block_drops(variant, None)
            self.assertEqual(bare_drops, [])

    def test_furnace_variants_register_in_world(self):
        world = World(seed=1, caves=[])
        cx, cy, cz = world.chunk_key(10, 65, 10)
        world.load_chunk(cx, cy, cz)

        # Placing furnace_north creates furnace tile entity
        world.set_block(10, 65, 10, "furnace_north")
        furnace_obj = world.get_furnace(10, 65, 10)
        self.assertIsNotNone(furnace_obj)
        self.assertEqual(furnace_obj.x, 10)
        self.assertEqual(furnace_obj.y, 65)
        self.assertEqual(furnace_obj.z, 10)


if __name__ == "__main__":
    unittest.main()
