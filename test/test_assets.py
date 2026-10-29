import unittest

from client.inventory import CreativeInventory
from server.blocks import (
    ASSETS,
    ATLAS,
    BLOCKS,
    BY_NAME,
    CREATIVE,
    DEFAULT_HOTBAR,
    TAB_ITEMS,
    TABS,
    DIRT,
    GRASS,
    HOTBAR_SLOTS,
    Block,
    _load_model,
    install,
    normalize_hotbar,
    uninstall,
)
from server.images import read_png
from server.stacks import stack_count, stack_id
from server.mesh import build_chunk_mesh
from server.world import Chunk, World


class AssetTests(unittest.TestCase):
    def test_blocks_come_from_the_model_files(self):
        grass = BY_NAME["grass"]
        self.assertEqual(grass.id, GRASS)
        self.assertEqual(grass.name, "Grass")
        self.assertTrue((ASSETS / "blocks" / "grass.json").is_file())
        self.assertTrue(grass.full_cube)
        self.assertFalse(grass.cutout)
        self.assertTrue(BY_NAME["leaves"].cutout)
        self.assertTrue(BY_NAME["leaves"].solid)
        self.assertEqual(BY_NAME["leaves"].opacity, 1.0)
        leaves = read_png(ASSETS / "textures" / "leaves.png")
        alpha = leaves[:, :, 3].reshape(-1)
        self.assertIn(0, alpha)
        self.assertIn(255, alpha)
        self.assertEqual(len(CREATIVE), len([b for b in BLOCKS.values() if not b.is_variant]))
        self.assertEqual([tab.id for tab in TABS], ["building", "natural", "tools", "materials", "inventory"])
        self.assertEqual(TABS[-1].screen, "player")
        self.assertEqual(BY_NAME["grass"].tabs, ("natural",))
        self.assertEqual(BY_NAME["stone"].tabs, ("building", "natural"))
        self.assertIn(BY_NAME["planks"].id, TAB_ITEMS["building"])
        self.assertNotIn(BY_NAME["leaves"].id, TAB_ITEMS["building"])
        self.assertEqual(DEFAULT_HOTBAR, ("",) * HOTBAR_SLOTS)
        self.assertIsInstance(grass.id, str)
        self.assertNotEqual(grass.face_uvs[2], grass.face_uvs[0])

    def test_inline_model_is_accepted(self):
        elements = _load_model({"elements": [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": {}}]})
        self.assertEqual(len(elements), 1)
        self.assertEqual(_load_model("cube")[0]["to"], [16, 16, 16])

    def test_side_texture_keeps_grass_above_dirt(self):
        image = read_png(ASSETS / "textures" / "grass_side.png")
        self.assertGreater(int(image[0, 0, 1]), int(image[0, 0, 0]))
        self.assertGreater(int(image[-1, 0, 0]), int(image[-1, 0, 1]))
        height, width = ATLAS.shape[:2]
        top_u, top_v = BY_NAME["grass"].face_uvs[0][3]
        bottom_u, bottom_v = BY_NAME["grass"].face_uvs[0][0]
        top = ATLAS[min(height - 1, int(top_v * height)), min(width - 1, int(top_u * width))]
        bottom = ATLAS[min(height - 1, int(bottom_v * height)), min(width - 1, int(bottom_u * width))]
        self.assertGreater(int(top[1]), int(top[0]))
        self.assertGreater(int(bottom[0]), int(bottom[1]))

    def test_hotbar_drops_unknown_blocks(self):
        slots = normalize_hotbar([GRASS, "missing", DIRT])
        self.assertEqual(stack_id(slots[0]), GRASS)
        self.assertEqual(stack_count(slots[0]), 1)
        self.assertEqual(stack_id(slots[1]), "")
        self.assertEqual(stack_id(slots[2]), DIRT)
        legacy = normalize_hotbar([1, 99])
        self.assertEqual(stack_id(legacy[0]), GRASS)
        self.assertEqual(stack_id(legacy[1]), "")
        counted = normalize_hotbar([{"id": DIRT, "count": 4}])
        self.assertEqual(stack_id(counted[0]), DIRT)
        self.assertEqual(stack_count(counted[0]), 4)
        self.assertEqual(len(slots), HOTBAR_SLOTS)
        self.assertTrue(all(stack_id(slot) == "" for slot in normalize_hotbar(None)))

    def test_creative_click_fills_a_hotbar_slot(self):
        inventory = CreativeInventory()
        hotbar = [""] * HOTBAR_SLOTS
        inventory.apply(f"item:{GRASS}", "left", hotbar, 0)
        self.assertEqual(inventory.held, GRASS)
        selected = inventory.apply("hotbar:4", "left", hotbar, 0)
        self.assertEqual(selected, 4)
        self.assertEqual(stack_id(hotbar[4]), GRASS)
        self.assertEqual(stack_count(hotbar[4]), 1)
        inventory.apply("hotbar:4", "right", hotbar, 4)
        self.assertEqual(stack_id(hotbar[4]), "")
        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0))
        actions = {hit.action for hit in frame.hits}
        self.assertEqual(inventory.tab, "building")
        for block_id in TAB_ITEMS["building"]:
            self.assertIn(f"item:{block_id}", actions)
        self.assertNotIn(f"item:{GRASS}", actions)
        inventory.scroll = 3
        inventory.apply("tab:natural", "left", hotbar, 0)
        self.assertEqual(inventory.scroll, 0)
        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0))
        actions = {hit.action for hit in frame.hits}
        self.assertIn(f"item:{GRASS}", actions)
        self.assertIn("tab:building", actions)
        self.assertEqual(frame.labels[0].text, "Natural Blocks")

    def test_shorter_model_lowers_the_top_face(self):
        uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        element = {
            "from": [0.0, 0.0, 0.0],
            "to": [16.0, 8.0, 16.0],
            "faces": {index: uvs for index in range(6)},
        }
        block = Block("slab", "Slab", True, False, [uvs] * 6, [element])
        install(block, 200)
        try:
            world = World()
            chunk = Chunk(0, 0, 0)
            chunk.data[3, 2, 4] = 200
            world.chunks[(0, 0, 0)] = chunk
            mesh = build_chunk_mesh(world, 0, 0, 0)
            self.assertEqual(len(mesh.vertices), 24)
            self.assertAlmostEqual(float(mesh.vertices[:, 1].max()), 3.5)
            self.assertAlmostEqual(float(mesh.vertices[:, 1].min()), 3.0)
        finally:
            uninstall("slab")
