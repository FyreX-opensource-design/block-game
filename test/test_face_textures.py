import unittest
from server.blocks import (
    BLOCKS,
    FACE_INDEX,
    _resolve_texture,
    _load_registry,
    _VERTICAL_SLAB_FACE_MAPS,
    Block,
)


class TestFaceTextures(unittest.TestCase):
    def test_resolve_texture_cardinal_and_aliases(self):
        # Cardinal faces
        mapping = {
            "up": "tex_up",
            "down": "tex_down",
            "north": "tex_north",
            "south": "tex_south",
            "east": "tex_east",
            "west": "tex_west",
        }
        self.assertEqual(_resolve_texture("#top", mapping, "test_block", face_name="up"), "tex_up")
        self.assertEqual(_resolve_texture("#bottom", mapping, "test_block", face_name="down"), "tex_down")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="north"), "tex_north")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="south"), "tex_south")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="east"), "tex_east")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="west"), "tex_west")

    def test_resolve_texture_synonyms(self):
        # Synonyms: top, bottom, front, back, right, left
        mapping = {
            "top": "tex_top",
            "bottom": "tex_bottom",
            "front": "tex_front",
            "back": "tex_back",
            "right": "tex_right",
            "left": "tex_left",
        }
        self.assertEqual(_resolve_texture("#top", mapping, "test_block", face_name="up"), "tex_top")
        self.assertEqual(_resolve_texture("#bottom", mapping, "test_block", face_name="down"), "tex_bottom")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="south"), "tex_front")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="north"), "tex_back")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="east"), "tex_right")
        self.assertEqual(_resolve_texture("#side", mapping, "test_block", face_name="west"), "tex_left")

    def test_resolve_texture_fallbacks(self):
        # side fallback for cardinal horizontal faces
        mapping_side = {"side": "tex_side", "top": "tex_top", "bottom": "tex_bottom"}
        self.assertEqual(_resolve_texture("#side", mapping_side, "test_block", face_name="north"), "tex_side")
        self.assertEqual(_resolve_texture("#side", mapping_side, "test_block", face_name="east"), "tex_side")

        # Specific face overrides side
        mapping_override = {"side": "tex_side", "north": "special_north", "top": "tex_top", "bottom": "tex_bottom"}
        self.assertEqual(_resolve_texture("#side", mapping_override, "test_block", face_name="north"), "special_north")
        self.assertEqual(_resolve_texture("#side", mapping_override, "test_block", face_name="east"), "tex_side")

        # Bottom fallback to top
        mapping_no_bottom = {"side": "tex_side", "top": "tex_top"}
        self.assertEqual(_resolve_texture("#bottom", mapping_no_bottom, "test_block", face_name="down"), "tex_top")

        # Universal fallback to texture or all
        mapping_texture = {"texture": "tex_all"}
        self.assertEqual(_resolve_texture("#side", mapping_texture, "test_block", face_name="north"), "tex_all")
        self.assertEqual(_resolve_texture("#top", mapping_texture, "test_block", face_name="up"), "tex_all")

        mapping_all = {"all": "tex_all"}
        self.assertEqual(_resolve_texture("#side", mapping_all, "test_block", face_name="west"), "tex_all")

        # Missing token raises KeyError
        with self.assertRaises(KeyError):
            _resolve_texture("#unknown", {}, "test_block", face_name="up")

    def test_existing_blocks_loaded(self):
        # Crafting table has top, side, bottom
        crafting = BLOCKS["crafting_table"]
        self.assertIsNotNone(crafting.face_uvs[FACE_INDEX["up"]])
        self.assertIsNotNone(crafting.face_uvs[FACE_INDEX["down"]])
        self.assertIsNotNone(crafting.face_uvs[FACE_INDEX["north"]])

        # Furnace has top, bottom, side, front and horizontal rotation
        furnace_south = BLOCKS["furnace"]
        furnace_north = BLOCKS["furnace_north"]
        furnace_east = BLOCKS["furnace_east"]
        furnace_west = BLOCKS["furnace_west"]

        # Front is south on unrotated furnace
        front_uv = furnace_south.face_uvs[FACE_INDEX["south"]]
        side_uv = furnace_south.face_uvs[FACE_INDEX["east"]]

        # North-facing furnace has front on north
        self.assertEqual(furnace_north.face_uvs[FACE_INDEX["north"]], front_uv)
        self.assertEqual(furnace_north.face_uvs[FACE_INDEX["south"]], side_uv)

        # East-facing furnace has front on east
        self.assertEqual(furnace_east.face_uvs[FACE_INDEX["east"]], front_uv)

        # West-facing furnace has front on west
        self.assertEqual(furnace_west.face_uvs[FACE_INDEX["west"]], front_uv)

    def test_horizontal_rotation_all_distinct_faces(self):
        # Create a mock block with 6 distinct UVs: east(0), west(1), up(2), down(3), south(4), north(5)
        # Using simple identifier strings/tuples for UVs
        u_east = ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1))
        u_west = ((0.1, 0.0), (0.2, 0.0), (0.2, 0.1), (0.1, 0.1))
        u_up = ((0.2, 0.0), (0.3, 0.0), (0.3, 0.1), (0.2, 0.1))
        u_down = ((0.3, 0.0), (0.4, 0.0), (0.4, 0.1), (0.3, 0.1))
        u_south = ((0.4, 0.0), (0.5, 0.0), (0.5, 0.1), (0.4, 0.1))
        u_north = ((0.5, 0.0), (0.6, 0.0), (0.6, 0.1), (0.5, 0.1))

        base_uvs = [u_east, u_west, u_up, u_down, u_south, u_north]

        # Check rotation mappings
        north_uvs = [base_uvs[old] for old in _VERTICAL_SLAB_FACE_MAPS["north"]]
        east_uvs = [base_uvs[old] for old in _VERTICAL_SLAB_FACE_MAPS["east"]]
        west_uvs = [base_uvs[old] for old in _VERTICAL_SLAB_FACE_MAPS["west"]]

        # For north facing:
        # East face gets West UV
        self.assertEqual(north_uvs[FACE_INDEX["east"]], u_west)
        # West face gets East UV
        self.assertEqual(north_uvs[FACE_INDEX["west"]], u_east)
        # South face gets North UV
        self.assertEqual(north_uvs[FACE_INDEX["south"]], u_north)
        # North face gets South UV
        self.assertEqual(north_uvs[FACE_INDEX["north"]], u_south)
        # Up and Down stay the same
        self.assertEqual(north_uvs[FACE_INDEX["up"]], u_up)
        self.assertEqual(north_uvs[FACE_INDEX["down"]], u_down)

        # For east facing:
        # East face gets South (front) UV
        self.assertEqual(east_uvs[FACE_INDEX["east"]], u_south)
        # West face gets North (back) UV
        self.assertEqual(east_uvs[FACE_INDEX["west"]], u_north)
        # South face gets West (left) UV
        self.assertEqual(east_uvs[FACE_INDEX["south"]], u_west)
        # North face gets East (right) UV
        self.assertEqual(east_uvs[FACE_INDEX["north"]], u_east)

        # For west facing:
        # East face gets North (back) UV
        self.assertEqual(west_uvs[FACE_INDEX["east"]], u_north)
        # West face gets South (front) UV
        self.assertEqual(west_uvs[FACE_INDEX["west"]], u_south)
        # South face gets East (right) UV
        self.assertEqual(west_uvs[FACE_INDEX["south"]], u_east)
        # North face gets West (left) UV
        self.assertEqual(west_uvs[FACE_INDEX["north"]], u_west)


if __name__ == "__main__":
    unittest.main()

