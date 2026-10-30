# Block Game - Feature Documentation

This document provides a comprehensive technical and design overview of all gameplay, generation, rendering, and simulation features implemented in Block Game.

---

## 1. Procedural World Generation

The world is procedurally generated using deterministic seed-based noise algorithms implemented in NumPy.

- **Coordinate System**:
  - `X`: East (+) / West (-)
  - `Y`: Up (+) / Down (-) (World height: 128 blocks, chunks are 16×16×16 blocks)
  - `Z`: South (+) / North (-)
- **Terrain Height**:
  - Computed using multi-octave Fractional Brownian Motion (fBm) noise.
  - Inland terrain blending combines base height, continuous mountain elevation offsets, and optional cliff-step plateaus.
  - Base ocean water level is fixed at `y = 64` (`WATER_LEVEL`).
- **Biomes**:
  - Climate noise (`climate_grid`) assigns biomes by climate bands (`ceiling` thresholds).
  - Biomes include: `ocean`, `beach`, `plains`, `forest`, `mountains`, `wooded_mountains`, `mountain_meadow`, `mountain_peaks`, and `mountain_cliffs`.
  - Surface layer (`surface`), subsurface layer (`soil`), and soil depth (`soil_depth`) are defined per biome.
- **Coastal Variation**:
  - Low-frequency coast noise varies the material and profile of shorelines.
  - Sandy beaches transition into gravel shores on some stretches of coast.
  - Other coastal stretches rise into steep stone cliffs that overlook the ocean.
  - The variation is applied only to dry sand-beach columns; seabeds and inland terrain are unaffected.
- **Underground Cave Systems**:
  - **Tunnels**: 3D dual-frequency worm/spaghetti noise carving continuous subterranean pathways (`assets/caves/tunnels.json`).
  - **Caverns**: Wide open underground chambers defined in Lua (`assets/caves/caverns.lua`).
  - Cave carvers respect water boundaries to prevent unroofing ocean floors.
- **Surface Lakes**:
  - Procedural surface water bodies carve depressions and fill with water exposed to the sky (`assets/features/water_lake.lua`).
- **Ore Distribution**:
  - Veins of `coal_ore`, `copper_ore`, `iron_ore`, and `gravel` generate throughout stone layers according to configured Y-min, Y-max, cluster sizes, and per-chunk attempts (`assets/ores/`).

---

## 2. Biome Grass Coloring & Smooth Blending

Grass surfaces dynamically adapt their color according to the local biome:

- **Per-Biome Color Palette**:
  - **Plains**: Lush natural green (`#83b749`)
  - **Forest**: Deep forest green (`#5da73b`)
  - **Mountain Meadow**: Vibrant alpine emerald green (`#6cdb3e`)
  - **Mountains**: Cool muted alpine sage (`#8ab689`)
  - **Wooded Mountains**: Evergreen pine green (`#689b52`)
  - **Mountain Peaks**: High-altitude slate green (`#78a783`)
  - **Mountain Cliffs**: Rugged stony green (`#6fa076`)
  - **Beach**: Sunny coastal yellow-green (`#a7cf5b`)
  - **Ocean**: Aquatic kelp green (`#8eb971`)
- **Smooth 3×3 Gradient Blending**:
  - Meshing samples climate values in an 18×18 column window around each 16×16 chunk and computes a box blur kernel.
  - Eliminates hard borders between biomes, creating smooth natural gradients.
- **Selective Texture Shader Tinting**:
  - In `WORLD_FRAG`, the shader isolates grass/foliage pixels (where green dominates over red and blue).
  - The grass blades/top are tinted to the biome color, while dirt bottoms and dirt/mud side lower sections stay natural without discoloration.
- **Blocks Supported**:
  - Normal `grass`, `muddy_grass`, `tall_grass`, and `fern`, plus any custom block declaring `"tint": "grass"`.

---

## 3. Dynamic Multi-Spectral Lighting & Day/Night Cycle

The lighting engine simulates realistic sunlight and multi-spectral block illumination:

- **6 Lighting Channels**:
  - `Skylight` (0–15): Ambient sunlight penetrating downward from the sky. Shadowed under solid overhangs.
  - `Blocklight RGB` (Channels 1–3, 0–15 each): Visible red, green, and blue light emitted by torches, furnaces, and glowing blocks.
  - `UV Light` (Channel 4, 0–15): Ultraviolet fluorescence light.
  - `IR Light` (Channel 5, 0–15): Infrared heat radiation emitted by heat sources (lava, burning furnaces, hot metals).
- **Light Propagation**:
  - Breadth-First Search (BFS) flood fill propagates skylight and blocklight with attenuation through air, transparent blocks, water, and leaves.
- **Day/Night & Celestial Movement**:
  - 24,000 tick cycle (1 second = 20 ticks by default).
  - Sun and moon rotate through the sky with synchronized celestial directional lighting.
  - Stars fade in during night and dusk.
  - Ambient sky color transitions seamlessly between dawn, daylight, dusk, and midnight.

---

## 4. Vision Modes

The player can toggle between 4 visual rendering spectrums (keys `V` or `L`, or via `/vision <mode>`):

1. **Normal Vision (`normal`)**: Standard realistic rendering with sunlight, shadows, ambient illumination, and warm blocklight.
2. **UV Vision (`uv`)**: Darkened scene where ultraviolet light causes materials to fluoresce in vivid purple and cyan.
3. **Infrared / Thermal Vision (`thermal`)**: FLIR false-color thermal imaging where temperatures and IR levels show as black/blue (cold) to magenta, orange, and bright white (hot).
4. **Full Spectrum / Debug (`spectrum`)**: Direct visualization of light buffer intensities (Skylight, RGB, UV, IR) for development and testing.

---

## 5. Fluid Simulation

Real-time cellular automaton fluid dynamics for `water`:

- **Water Levels & Spreading**:
  - Source blocks (`level = 8`) spread horizontally up to 7 steps with decaying height.
  - Fluid heights are meshed dynamically with sloped top quads.
- **Vertical Falling Columns**:
  - Water flowing over an edge forms vertical falling columns that hit the floor and spread outward.
- **Ocean Basins**:
  - Ocean surface water blocks are maintained at `WATER_LEVEL` (y=64) without draining.
  - Flowing water falling into ocean water correctly dissipates into the body of water.
- **Bucket Mechanics**:
  - Buckets (`wood_bucket`, `clay_bucket`, `iron_bucket`) can scoop water sources into water buckets and place fluids into the world.
  - Placing water against plants washes away the plant and drops its loot table.

---

## 6. Trees, Shapes & Leaf Decay

Procedural tree generation with realistic lifecycle dynamics:

- **Tree Shapes**:
  - `blob`: Classic rounded canopy (e.g. Standard `oak.json`).
  - `conifer`: Tiered conical evergreen foliage with exposed spire (e.g. `pine.json`).
  - `tall`: Tall columnar trunk with elevated canopy (e.g. `tall_oak.json`).
  - `branching`: Large spread tree with angled branches and foliage clusters (e.g. `fancy_oak.json`).
  - `shrub`: Low compact bush/shrub (e.g. `shrub.json`).
- **Leaf Decay Engine**:
  - Natural leaves (`leaves`) require connection to log blocks within a Manhattan/BFS distance of 4.
  - When trunk logs are chopped down, disconnected leaves randomly schedule decay timers (0.3s to 2.5s).
  - Decaying leaves drop sticks and saplings according to [`assets/loot/blocks/leaves.json`](file:///home/nathan/block-game/assets/loot/blocks/leaves.json) and cascade decay checks to neighbors.
- **Saplings**:
  - Oak and pine leaves occasionally drop their matching sapling (`oak_sapling`, `pine_sapling`).
  - Saplings can only be placed on soil blocks and are washed away by fluids like other plants.
  - After 15–30 minutes of staying placed, a sapling grows into a randomly chosen tree from the list in its [`assets/saplings/`](file:///home/nathan/block-game/assets/saplings) file.
  - Growth timers pause while the chunk is unloaded and are saved with the world, so reloading does not reset them.
- **Player Leaf Persistence**:
  - Leaves placed by players convert to `leaves_persistent`, making them completely immune to decay.

---

## 7. Directional Block Placement

Blocks rotate contextually based on placement angle and player orientation:

- **3-Axis Log Rotation**:
  - Placing logs against vertical or horizontal block faces aligns their axis (`x`, `y`, or `z`).
- **Cardinal Furnace Facing**:
  - Furnaces orient their front face (`north`, `south`, `east`, `west`) towards the player who placed them.
  - When actively smelting, the furnace front texture swaps to lit front with animated particle/glow emission.

---

## 8. Crafting, Smelting & Knapping

Multiple crafting tiers and interaction methods:

- **2×2 Inventory Crafting**: Basic grid accessible directly in player inventory (`E`).
- **3×3 Crafting Table**: Full crafting grid for tools, armor, and complex constructions.
- **Tag Ingredient Matching**: Recipes can specify tag ingredients (e.g. `"#logs"`, `"#leaves"`, `"#soil"`).
- **Furnace Smelting**:
  - Real-time fuel burn timers and smelting cook progression.
  - Fuel support for coal, charcoal, logs, planks, crafting tables, sticks, and wooden pickaxes.
- **Flint Knapping**:
  - Right-clicking flint against hard stone surfaces knaps it into `knapped_flint`.
- **In-World Block-Hit Recipes**:
  - Hitting specific blocks with specific items transforms or harvests materials without breaking the block in creative mode.

---

## 9. Survival Mechanics, Entities & AI

Full survival game loop with living creatures:

- **Health & Damage**:
  - 20 HP health pool with visual heart indicators.
  - Damage sources: Fall damage, drowning (oxygen bar), mob attacks, suffocation.
  - Invulnerability tick frames with red flash damage visualization.
- **Hunger & Saturation**:
  - Hunger bar with saturation buffer.
  - Sprinting and jumping deplete hunger; eating foods (e.g. `raw_porkchop`) restores hunger and health.
- **Entities & Mob AI**:
  - Wandering animals (e.g. `pig`) with multi-bone 3D models and animated walking legs.
  - Idle wandering, obstacle avoidance, panic sprint when attacked, and sound effects.
  - Loot table drops upon defeat.

---

## 10. HUD Readouts & In-Game Commands

- **On-Screen HUD**:
  - Current coordinates: `X`, `Y`, `Z`.
  - Current biome name (e.g. `Plains`, `Mountain Meadow`, `Wooded Mountains`).
  - Targeted block name and currently held item name/count.
  - Hotbar and container UI.
- **Chat & In-Game Commands**:
  - Press `T` for chat, `/` for command entry.
  - Commands include: `/help`, `/pos`, `/biome`, `/tp`, `/say`, `/seed`, `/light`, `/vision`, `/time`, `/spawn`.

