import unittest
import numpy as np

from server.blocks import AIR, CHUNK_SIZE, GRASS, SECTION_COUNT, STONE, block_index, install, uninstall, Block
from server.mesh import build_chunk_mesh
from server.world import Chunk, World
from server.lighting import (
    CHANNEL_B,
    CHANNEL_G,
    CHANNEL_IR,
    CHANNEL_R,
    CHANNEL_UV,
    LIGHT_CHANNELS,
    LIGHT_MAX,
    LightingSystem,
    parse_light_levels,
)
from server.commands import CommandContext, CommandHandler


class LightingTests(unittest.TestCase):
    def test_parse_light_levels(self):
        # Scalar
        self.assertEqual(parse_light_levels(10), (10, 10, 10, 10, 10))
        # 3-channel RGB (UV and IR default to 0)
        self.assertEqual(parse_light_levels((15, 8, 2)), (15, 8, 2, 0, 0))
        # 5-channel RGB + UV + IR
        self.assertEqual(parse_light_levels((15, 10, 5, 3, 14)), (15, 10, 5, 3, 14))
        # Dict
        d = {"r": 12, "g": 6, "b": 0, "uv": 4, "ir": 15}
        self.assertEqual(parse_light_levels(d), (12, 6, 0, 4, 15))
        # Clamping
        self.assertEqual(parse_light_levels((-5, 20, 0, 100, -2)), (0, 15, 0, 15, 0))
        # None
        self.assertEqual(parse_light_levels(None), (0, 0, 0, 0, 0))

    def test_open_sky_has_maximum_skylight(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk
        world.lighting.compute_chunk_lighting(0, 0, 0)
        # In an empty chunk with open sky above, skylight is maximum (15)
        self.assertEqual(world.get_skylight(8, 15, 8), 15)
        self.assertEqual(world.get_skylight(8, 0, 8), 15)

    def test_solid_block_casts_shadow_and_attenuates_skylight(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk
        # Place a ceiling of stone from x=4..12, z=4..12 at y=10
        stone = block_index(STONE)
        chunk.data[10, 4:12, 4:12] = stone
        world.lighting.compute_chunk_lighting(0, 0, 0)

        # Above ceiling is direct sky (15)
        self.assertEqual(world.get_skylight(8, 12, 8), 15)
        # Right under center of ceiling (y=9, x=8, z=8), direct sunlight is blocked
        # Diffuse light propagates from edges, so center is shadowed
        shadow_level = world.get_skylight(8, 9, 8)
        self.assertLess(shadow_level, 15)
        self.assertGreaterEqual(shadow_level, 0)

    def test_blocklight_5_channels_propagation(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk

        # Set a blocklight source at (8, 8, 8): R=15, G=11, B=4, UV=2, IR=15
        world.set_blocklight(8, 8, 8, 15, 11, 4, 2, 15)

        # At source position:
        bl_source = world.get_blocklight(8, 8, 8)
        self.assertEqual(bl_source, (15, 11, 4, 2, 15))

        # 1 block away (8, 8, 9) in air:
        # All channels decrease by 1
        bl_1 = world.get_blocklight(8, 8, 9)
        self.assertEqual(bl_1, (14, 10, 3, 1, 14))

        # 2 blocks away (8, 8, 10):
        bl_2 = world.get_blocklight(8, 8, 10)
        self.assertEqual(bl_2, (13, 9, 2, 0, 13))

        # 5 blocks away (8, 8, 13):
        # B had 4, so it should be 0. UV had 2, so it should be 0.
        # R was 15 - 5 = 10. G was 11 - 5 = 6. IR was 15 - 5 = 10.
        bl_5 = world.get_blocklight(8, 8, 13)
        self.assertEqual(bl_5[CHANNEL_B], 0)
        self.assertEqual(bl_5[CHANNEL_UV], 0)
        self.assertEqual(bl_5[CHANNEL_R], 10)
        self.assertEqual(bl_5[CHANNEL_G], 6)
        self.assertEqual(bl_5[CHANNEL_IR], 10)

    def test_blocklight_blocked_by_solid_blocks(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk

        # Solid wall at x=9
        chunk.data[:, :, 9] = block_index(STONE)
        world.set_blocklight(8, 8, 8, 15, 15, 15, 15, 15)

        # Behind the solid stone wall at x=10: blocklight should not penetrate
        bl_behind = world.get_blocklight(10, 8, 8)
        self.assertEqual(bl_behind, (0, 0, 0, 0, 0))

    def test_furnace_lit_emission(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        chunk.data[5, 5, 5] = block_index("furnace")
        world.chunks[(0, 0, 0)] = chunk
        furnace = world.get_or_create_furnace(5, 5, 5)

        # Unlit furnace emits no light
        self.assertFalse(furnace.is_lit)
        world.lighting.compute_chunk_lighting(0, 0, 0)
        self.assertEqual(world.get_blocklight(5, 5, 5), (0, 0, 0, 0, 0))

        # Light the furnace
        furnace.burn_time = 10.0
        self.assertTrue(furnace.is_lit)
        world.lighting.on_block_changed(5, 5, 5)

        # Lit furnace emits high RGB, UV, and IR
        emit = world.get_blocklight(5, 5, 5)
        self.assertEqual(emit, (15, 11, 4, 2, 15))
        # Adjacent block receives lit furnace light
        adj = world.get_blocklight(5, 6, 5)
        self.assertEqual(adj, (14, 10, 3, 1, 14))

    def test_mesh_contains_lighting_data(self):
        world = World()
        chunk = Chunk(0, 0, 0)
        chunk.data[5, 5, 5] = block_index(STONE)
        world.chunks[(0, 0, 0)] = chunk
        world.set_blocklight(5, 6, 5, 15, 10, 5, 2, 14)

        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh)
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(mesh.vertices.shape[1], 9)  # Shape compatibility preserved

        # mesh.lights has 6 columns: skylight, R, G, B, UV, IR
        self.assertIsNotNone(mesh.lights)
        self.assertEqual(len(mesh.lights), 24)
        self.assertEqual(mesh.lights.shape[1], 6)

        # The top face of the block at (5,5,5) faces +Y (5,6,5)
        # where we placed the light source
        self.assertTrue(np.any(mesh.lights[:, 1] > 0))  # Red channel
        self.assertTrue(np.any(mesh.lights[:, 4] > 0))  # UV channel
        self.assertTrue(np.any(mesh.lights[:, 5] > 0))  # IR channel

    def test_commands_light_and_vision(self):
        from server.player import Player

        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk
        player = Player(world)
        player.pos = [5.0, 5.0, 5.0]
        handler = CommandHandler()
        ctx = CommandContext(world=world, player=player)

        # Query light
        handled = handler.execute(ctx, "/light")
        self.assertTrue(handled)
        self.assertTrue(any("Skylight" in r[0] and "Blocklight" in r[0] for r in ctx.responses))

        # Set light via command
        ctx.responses.clear()
        handler.execute(ctx, "/light set 15 10 5 3 12")
        self.assertEqual(world.get_blocklight(5, 5, 5), (15, 10, 5, 3, 12))

        # Clear light via command
        ctx.responses.clear()
        handler.execute(ctx, "/light clear")
        self.assertEqual(world.get_blocklight(5, 5, 5), (0, 0, 0, 0, 0))

        # Vision command
        ctx.responses.clear()
        handler.execute(ctx, "/vision thermal")
        self.assertTrue(any("Infrared / Thermal" in r[0] for r in ctx.responses))

    def test_multithreaded_lighting_computation(self):
        world = World()
        # Create 4 chunks in column
        for cy in range(4):
            world.chunks[(0, cy, 0)] = Chunk(0, cy, 0)

        world.lighting.compute_chunks_parallel([(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)])
        for cy in range(4):
            self.assertTrue(world.chunks[(0, cy, 0)].light_valid)
            self.assertEqual(world.chunks[(0, cy, 0)].skylight[15, 8, 8], 15)

    def test_cave_boundary_skylight_diffusion(self):
        world = World()
        c0 = Chunk(0, 0, 0)
        c1 = Chunk(1, 0, 0)
        world.chunks[(0, 0, 0)] = c0
        world.chunks[(1, 0, 0)] = c1

        stone = block_index(STONE)
        c0.data.fill(stone)
        c1.data.fill(stone)

        # Carve a horizontal tunnel crossing from chunk 0 into chunk 1 at y=5, z=8
        c0.data[5, 8, :] = 0
        c1.data[5, 8, :] = 0
        # Skylight entrance near edge in chunk 0
        c0.data[5:16, 8, 14] = 0

        world.lighting.ensure_chunk_lit(c0)
        world.lighting.ensure_chunk_lit(c1)

        # Chunk 0 edge
        edge_c0 = world.get_skylight(15, 5, 8)
        self.assertEqual(edge_c0, 14)
        # Across the chunk boundary in chunk 1 (x=16):
        # Light should seamlessly propagate into chunk 1!
        edge_c1 = world.get_skylight(16, 5, 8)
        self.assertEqual(edge_c1, 13)
        self.assertEqual(world.get_skylight(17, 5, 8), 12)
        self.assertEqual(world.get_skylight(18, 5, 8), 11)

    def test_ocean_vertical_skylight_propagation(self):
        world = World(seed=7)
        for cy in range(8):
            world.load_chunk(-1, cy, -1)

        # Query from bottom up
        for cy in range(4):
            world.lighting.ensure_chunk_lit(world.chunks[(-1, cy, -1)])

        # Water in chunk cy=3 at y=63 and y=62 must receive attenuated sunlight, not 0
        self.assertGreater(world.get_skylight(-5, 64, -5), 0)
        self.assertGreater(world.get_skylight(-5, 63, -5), 0)
        self.assertGreater(world.get_skylight(-5, 62, -5), 0)
        # Smooth attenuation: 64 >= 63 >= 62
        self.assertGreaterEqual(world.get_skylight(-5, 64, -5), world.get_skylight(-5, 63, -5))
        self.assertGreaterEqual(world.get_skylight(-5, 63, -5), world.get_skylight(-5, 62, -5))

    def test_ocean_and_cave_lighting_no_recursion(self):
        """Ensure lighting propagation across wide ocean / cave areas does not exceed recursion limits."""
        world = World(seed=0)
        cx, cy, cz = -11, 4, -9
        for dy in range(SECTION_COUNT - 1, -1, -1):
            for dz in range(-2, 3):
                for dx in range(-2, 3):
                    world.load_chunk(cx + dx, dy, cz + dz)

        for dy in range(SECTION_COUNT - 1, -1, -1):
            for dz in range(-2, 3):
                for dx in range(-2, 3):
                    world.lighting.ensure_chunk_lit(world.chunks[(cx + dx, dy, cz + dz)])

        ch = world.chunks[(cx, cy, cz)]
        self.assertTrue(getattr(ch, "light_valid", False))
        # Ensure light query succeeds without recursion error
        val = world.get_skylight(-175, 78, -138)
        self.assertIsInstance(val, int)

    def test_unloaded_neighbor_boundary_clamped_not_daylight(self):
        """Ensure boundary faces touching unloaded chunks do not falsely shine with 1.0 daylight."""
        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk
        stone = block_index(STONE)
        chunk.data.fill(stone)
        # Carve an internal hollow area (pitch dark underground)
        chunk.data[5:10, 5:10, 0:5] = 0
        from server.mesh import build_chunk_mesh
        mesh = build_chunk_mesh(world, 0, 0, 0)
        self.assertIsNotNone(mesh)
        # In this underground hollow cave touching unloaded neighbor (x < 0),
        # boundary face light should clamp to interior darkness (0.0), NOT 1.0 daylight!
        self.assertEqual(mesh.lights[:, 0].max(), 0.0)

    def test_ocean_surface_boundary_continuity(self):
        """Ensure ocean water surface across chunk boundaries is uniformly lit by sunlight."""
        world = World(seed=1)
        built, _ = world.sync(0, 4, 0)
        while built is not None:
            built, _ = world.sync(0, 4, 0)

        from server.mesh import build_chunk_mesh
        import numpy as np
        # Meshes of water in loaded chunks along Z
        for cz in (0, 1, 2, 3):
            mesh = build_chunk_mesh(world, 0, 4, cz)
            tl = getattr(mesh, "translucent_lights", None) if mesh else None
            if tl is not None and len(tl) > 0:
                tl_arr = np.array(tl)
                # Ocean surface skylight should be full daylight (1.0), not dark
                self.assertGreaterEqual(tl_arr[:, 0].mean(), 0.85)

    def test_section_boundary_block_change_lighting_stability(self):
        """Ensure breaking/placing blocks at y=63 and y=64 (section boundary) maintains valid lighting and does not black out chunks."""
        world = World(seed=20)
        # Load chunks around cave (-11, 4, -9)
        cx0, cz0 = -11, -9
        for cy in range(SECTION_COUNT - 1, -1, -1):
            for dcx in (-1, 0, 1):
                for dcz in (-1, 0, 1):
                    world.load_chunk(cx0 + dcx, cy, cz0 + dcz)

        from server.mesh import build_chunk_mesh
        # Initial mesh
        m3_init = build_chunk_mesh(world, cx0, 3, cz0)
        m4_init = build_chunk_mesh(world, cx0, 4, cz0)

        # Place block at boundary y=63
        x, z = cx0 * CHUNK_SIZE + 5, cz0 * CHUNK_SIZE + 5
        world.set_block(x, 63, z, STONE)
        m3_place63 = build_chunk_mesh(world, cx0, 3, cz0)
        m4_place63 = build_chunk_mesh(world, cx0, 4, cz0)

        # Chunks must remain valid and not pitch black from invalidation loops
        self.assertTrue(world.chunks[(cx0, 3, cz0)].light_valid)
        self.assertTrue(world.chunks[(cx0, 4, cz0)].light_valid)
        self.assertGreater(m4_place63.lights[:, 0].max(), 0.0)

        # Place block at boundary y=64
        world.set_block(x, 64, z, STONE)
        m3_place64 = build_chunk_mesh(world, cx0, 3, cz0)
        m4_place64 = build_chunk_mesh(world, cx0, 4, cz0)
        self.assertTrue(world.chunks[(cx0, 3, cz0)].light_valid)
        self.assertTrue(world.chunks[(cx0, 4, cz0)].light_valid)
        self.assertGreater(m4_place64.lights[:, 0].max(), 0.0)

        # Breaking block reverts cleanly
        world.set_block(x, 64, z, AIR)
        world.set_block(x, 63, z, AIR)
        m3_revert = build_chunk_mesh(world, cx0, 3, cz0)
        m4_revert = build_chunk_mesh(world, cx0, 4, cz0)
        self.assertTrue(world.chunks[(cx0, 3, cz0)].light_valid)
        self.assertTrue(world.chunks[(cx0, 4, cz0)].light_valid)
        self.assertGreater(m4_revert.lights[:, 0].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
