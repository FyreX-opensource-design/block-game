# Block Game - Asset Creation Guide (JSON & Lua)

Block Game features a data-driven content pipeline. Assets can be defined using standard **JSON** files or lightweight **Lua** scripts.

The Lua loader uses native C bindings (`liblua.so.5.5` via ctypes) to execute and extract table definitions directly into game dictionaries and lists.

---

## 1. Dual Format Support: JSON vs Lua

Any file in the asset directories can be authored either in `.json` or `.lua`:

| Format | File Extension | Syntax Style | Advantages |
| :--- | :--- | :--- | :--- |
| **JSON** | `.json` | Standard JSON object `{ ... }` | Strict schemas, ubiquitous tooling |
| **Lua** | `.lua` | `return { ... }` table | Inline math, comments, flexible syntax |

### Lua File Rule
A `.lua` file must return a single top-level table:
```lua
return {
  id = "my_asset",
  property = 123,
  tags = { "tag1", "tag2" },
}
```

---

## 2. Biomes (`assets/biomes/`)

Defines climate noise thresholds, terrain heights, surface/subsurface layers, trees, decorations, and grass colors.

### Fields
- `id` (string, required): Lowercase identifier matching filename.
- `ceiling` (float, required): Climate noise threshold (between 0.0 and 1.0). Biome bands climb in ascending order.
- `surface` (string): Block ID for surface layer (default `"grass"`).
- `soil` (string): Block ID for subsurface layers (default `"dirt"`). For sandy beaches this is `sand`, with `sandstone` below via `subsoil`.
- `soil_depth` (int): Number of soil layers below surface (default `3`).
- `subsoil` (string, optional): Block ID for deeper subsurface layers below the soil layers (default: the `soil` block itself).
- `subsoil_depth` (int, optional): Number of subsoil layers below the soil layers (default `0`). Sandy beaches use `sandstone` here.
- `scale` (float): Climate noise scale (must match across all biomes, typically `0.006`).
- `octaves` (int): Climate noise octaves (typically `2`).
- `seed_offset` (int): Offset added to world seed (typically `90`).
- `tree_chance` (float): Probability per surface block to spawn a tree (0.0 to 1.0).
- `trees` (list of strings): Tree IDs available in this biome (e.g. `["oak", "pine"]`).
- `decorations` (list): Decoration features or IDs (e.g. `["tall_grass", "fern", "rose"]`).
- `grass_color` (string or array): Hex color (`"#83b749"`) or `[r, g, b]` (0-255 or 0-1) for grass tinting.
- `height_offset` (float, optional): Added terrain elevation offset.
- `height_scale` (float, optional): Multiplier on terrain elevation noise.
- `cliff` (bool, optional): Enable cliff edge terrace effects.
- `cliff_step` (float, optional): Height step per cliff terrace.
- `variant_of` (string, optional): Turns this biome into a variant of another base biome. Variants share their parent's climate band and are selected by `variant_noise` instead. `ceiling`, `scale`, `octaves`, and `seed_offset` must match the parent.
- `variant_noise` (object, optional, required for variants): Noise used to choose where the variant appears within its parent's band.
  - `threshold` (float): Noise value at which the variant takes over (default `0.5`).
  - `above` (bool): Select where noise is `>= threshold` (default `true`), or `<= threshold` when `false`.
  - `scale` (float): Noise scale (defaults to the parent's `coast.noise.scale`, typically `0.018`).
  - `octaves` (int): Noise octaves (default `3`).
  - `seed_offset` (int): Offset added to the world seed (default `7331`).
  - `offset` (array of two floats): X/Z noise offset (default `[7.3, -11.9]`).
- `coast` (object, optional): Coastal shaping for non-inland biomes. Only biomes that define (or inherit) `coast` are treated as coastline.
  - `noise` (object): Low-frequency coastline noise. Accepts `scale`, `octaves`, `seed_offset`, and `offset` (same defaults as `variant_noise`).
  - `cliff` (object, optional): Rocky cliff headlands where the coast noise is below `threshold`.
    - `threshold` (float): Coast noise value below which cliffs appear (default `0.42`).
    - `blend` (float): Softness of the cliff transition (default `0.10`).
    - `power` (float): Steepness of the coastal profile (default `4.0`).
    - `uplift` (float): How far cliffs rise above the normal shelf (default `7.0`).
  - Cliff headland materials (`surface` / `soil`) live on the cliff biome variant, e.g. `cliff_beach.json`.

Biome variants do not appear in the top-level biome list. They are resolved per coordinate alongside their parent, so a gravel beach can share the beach climate band while replacing its surface and soil.

### Example: JSON (`plains.json`)
```json
{
  "id": "plains",
  "surface": "grass",
  "soil": "dirt",
  "soil_depth": 3,
  "scale": 0.006,
  "octaves": 2,
  "seed_offset": 90,
  "ceiling": 0.54,
  "tree_chance": 0.0022,
  "height_offset": 0,
  "height_scale": 1.0,
  "cliff": false,
  "trees": ["oak", "shrub"],
  "decorations": ["tall_grass", "dandelion", "rose"],
  "grass_color": "#83b749"
}
```

### Example: Lua (`forest.lua`)
```lua
return {
  id = "forest",
  surface = "grass",
  soil = "dirt",
  soil_depth = 3,
  scale = 0.006,
  octaves = 2,
  seed_offset = 90,
  ceiling = 0.64,
  tree_chance = 0.014,
  height_offset = 2,
  height_scale = 1.0,
  cliff = false,
  trees = { "oak", "fancy_oak", "tall_oak" },
  decorations = { "tall_grass", "fern", "dandelion", "rose" },
  grass_color = "#5da73b",
}
```

### Example: Variant (`gravel_beach.json`)
```json
{
  "id": "gravel_beach",
  "surface": "gravel",
  "soil": "gravel",
  "soil_depth": 3,
  "scale": 0.006,
  "octaves": 2,
  "seed_offset": 90,
  "ceiling": 0.36,
  "inland": false,
  "valid_spawn": false,
  "variant_of": "beach",
  "variant_noise": {
    "threshold": 0.60,
    "above": true
  },
  "tree_chance": 0.0,
  "trees": [],
  "grass_color": "#a7cf5b"
}
```

### Example: Variant (`cliff_beach.json`)
```json
{
  "id": "cliff_beach",
  "surface": "stone",
  "soil": "stone",
  "soil_depth": 3,
  "scale": 0.006,
  "octaves": 2,
  "seed_offset": 90,
  "ceiling": 0.36,
  "inland": false,
  "valid_spawn": false,
  "variant_of": "beach",
  "variant_noise": {
    "coast_cliff": true
  },
  "tree_chance": 0.0,
  "trees": [],
  "grass_color": "#a7cf5b"
}
```

The matching base biome can also describe coastline shape, such as the cliffs in `beach.json`:

```json
"coast": {
  "noise": { "scale": 0.018, "octaves": 3, "seed_offset": 7331, "offset": [7.3, -11.9] },
  "cliff": {
    "threshold": 0.42,
    "blend": 0.10,
    "power": 4.0,
    "uplift": 7.0
  }
}
```

Cliff headland materials are defined on the cliff biome variant (`surface` / `soil` above), while the base biome only describes the terrain profile.

---

## 3. Blocks (`assets/blocks/`)

Defines 3D blocks, hitboxes, models, hardness, and visual characteristics.

### Fields
- `id` (string, required): Unique block identifier.
- `lang` (string): Localization key (e.g. `"block.grass"`).
- `tab` (string or array): Creative inventory tabs (`"building"`, `"natural"`, etc.).
- `model` (string): Model file name in `assets/models/` (`"cube"`, `"cross"`, `"orientable"`).
- `solid` (bool): Whether entities collide with the block (default `true`).
- `render` (string): `"opaque"` (default) or `"cutout"` (alpha cutout for plants/leaves).
- `opacity` (float): Light opacity factor (1.0 = solid shadow, 0.0 = fully transparent).
- `hardness` (float): Break time multiplier (0.0 = instant, e.g. tall grass).
- `tool` (string): Effective harvest tool (`"shovel"`, `"pickaxe"`, `"axe"`, `"none"`).
- `mining_level` (int): Required tool tier (0 = hand/wood, 1 = stone, 2 = iron).
- `loot_table` (string): Loot table path (relative to `assets/loot/`).
- `tint` (string, optional): Set to `"grass"` to receive biome grass coloring.
- `light` / `emission` (int, list, or dict): Light emission levels:
  - Int `14` (emits white RGB blocklight 14).
  - Dict `{"r": 15, "g": 10, "b": 4, "uv": 2, "ir": 14}`.
- `properties` (dict): Additional runtime attributes:
  - `"rotation"`: `"axis"` (log rotation) or `"horizontal"` (furnace facing).
  - `"decayable"`: `true` (natural leaves decaying when unsupported).
  - `"plant"`: `true` (plant block requiring ground support).
  - `"fluid_breakable"`: `true` (washed away when hit by water).
- `textures` (dict): Texture mapping keys to files in `assets/textures/`:
  - Cube: `{"top": "...", "side": "...", "bottom": "..."}` or `{"all": "..."}`.
  - Cross: `{"cross": "..."}`.
  - Orientable: `{"top": "...", "bottom": "...", "front": "...", "side": "..."}`.
- `type` (string, optional): `"slab"` to register half-height slab variants (e.g. `sandstone_slab` generates `sandstone_slab_top` and waterlogged halves).
- `gravity` (bool): Whether the block falls as a falling entity (default `false`).

### Example (`sandstone.json`)
```json
{
  "id": "sandstone",
  "lang": "block.sandstone",
  "tab": "natural",
  "model": "cube",
  "hardness": 0.5,
  "tool": "shovel",
  "mining_level": 0,
  "loot_table": "blocks/sandstone",
  "gravity": false,
  "textures": {"top": "sandstone", "side": "sandstone", "bottom": "sandstone"}
}
```

### Example Slab (`sandstone_slab.json`)
```json
{
  "id": "sandstone_slab",
  "lang": "block.sandstone_slab",
  "type": "slab",
  "tab": ["building"],
  "model": "slab",
  "hardness": 1.0,
  "tool": "axe",
  "mining_level": 0,
  "loot_table": "blocks/sandstone_slab",
  "textures": {"top": "sandstone", "side": "sandstone", "bottom": "sandstone"}
}
```

### Example (`smooth_sandstone.json`)
```json
{
  "id": "smooth_sandstone",
  "lang": "block.smooth_sandstone",
  "tab": "building",
  "model": "cube",
  "hardness": 0.6,
  "tool": "pickaxe",
  "mining_level": 1,
  "loot_table": "blocks/smooth_sandstone",
  "gravity": false,
  "textures": {"top": "smooth_sandstone", "side": "smooth_sandstone", "bottom": "smooth_sandstone"}
}
```

### Example (`muddy_grass.json`)
```json
{
  "id": "muddy_grass",
  "lang": "block.muddy_grass",
  "tab": "natural",
  "model": "cube",
  "solid": true,
  "soil": true,
  "hardness": 0.6,
  "tool": "shovel",
  "mining_level": 0,
  "sound": "grass",
  "tint": "grass",
  "loot_table": "blocks/muddy_grass",
  "textures": {
    "top": "muddy_grass_top",
    "side": "muddy_grass_side",
    "bottom": "mud"
  }
}
```

---

## 4. Items (`assets/items/`)

Defines inventory items, hand models, and tool behaviors.

### Fields
- `id` (string, required): Unique item identifier.
- `lang` (string): Localization key (e.g. `"item.iron_pickaxe"`).
- `tab` (string or array): Creative tab category (`"tools"`, `"materials"`, etc.).
- `stack_size` (int): Max stack limit (default `64`, tools usually `1`).
- `tool_type` (string, optional): `"pickaxe"`, `"axe"`, `"shovel"`.
- `mining_level` (int, optional): Tool material tier.
- `mining_speed` (float, optional): Breaking speed multiplier.
- `durability` (int, optional): Uses before breaking.
- `food` (dict, optional): `{"nutrition": 4, "saturation": 0.6}` for edible items.

### Example (`raw_iron.json`)
```json
{
  "id": "raw_iron",
  "lang": "item.raw_iron",
  "tab": "materials",
  "stack_size": 64
}
```

---

## 5. Item & Block Tags (`assets/tags/`)

Tags group multiple blocks or items together under a single `#tag_name` identifier for crafting, loot, and world generation.

- Block tags: `assets/tags/blocks/<tag>.json`
- Item tags: `assets/tags/items/<tag>.json`

### Fields
- `values` (list of strings): Block/Item IDs or parent tags prefixed with `#`.

### Example: Block Tag (`assets/tags/blocks/logs.json`)
```json
{
  "values": [
    "log",
    "log_x",
    "log_z"
  ]
}
```

---

## 6. Recipes (`assets/recipes/`)

Defines shaped crafting, shapeless crafting, furnace smelting, and block-hit reactions.

### A. Shaped Recipe
```json
{
  "type": "crafting_shaped",
  "pattern": [
    "XX",
    "XX"
  ],
  "key": {
    "X": "planks"
  },
  "result": {
    "id": "crafting_table",
    "count": 1
  }
}
```
*Ingredients can use tags like `"X": "#logs"`.*

### B. Shapeless Recipe
```json
{
  "type": "crafting_shapeless",
  "ingredients": [
    "#logs"
  ],
  "result": {
    "id": "planks",
    "count": 4
  }
}
```

### C. Smelting Recipe
```json
{
  "type": "smelting",
  "ingredient": "copper_ore",
  "result": {
    "id": "copper_ingot",
    "count": 1
  },
  "cook_time": 10.0
}
```

### D. Block-Hit Recipe (In-World Right-Click / Strike)
```json
{
  "type": "block_hit",
  "block": "grass",
  "item": "water_bucket",
  "result_block": "muddy_grass",
  "result_item": "empty_bucket"
}
```

---

## 7. Trees (`assets/trees/`)

Defines procedural tree types, trunk heights, canopies, and branching.

### Fields
- `id` (string, required): Tree identifier.
- `trunk` (string): Trunk log block ID (default `"log"`).
- `leaves` (string): Foliage block ID (default `"leaves"`).
- `shape` (string): Foliage generator algorithm:
  - `"blob"`: Standard spherical canopy.
  - `"conifer"`: Tiered conical evergreen foliage.
  - `"tall"`: Compact elevated columnar canopy.
  - `"branching"`: Large canopy with extending diagonal branches.
  - `"shrub"`: Compact ground bush.
- `trunk_min` (int): Minimum trunk height.
- `trunk_max` (int): Maximum trunk height.
- `radius` (int): Foliage horizontal radius.
- `branch_chance` (float): Probability to generate branches (branching trees).
- `branch_count` (int): Maximum number of branches.
- `branch_length` (int): Length of angled branches.
- `valid_blocks` (list): Allowed soil blocks (supports tags like `"#soil"`).

### Example: Conifer / Pine (`pine.json`)
```json
{
  "id": "pine",
  "trunk": "log",
  "leaves": "leaves",
  "shape": "conifer",
  "trunk_min": 7,
  "trunk_max": 11,
  "radius": 2,
  "valid_blocks": ["#soil"]
}
```

### Saplings (`assets/saplings/`)
Saplings are placeable plant blocks that grow into a tree after a random delay. Each
sapling file lists the tree assets it may become:

- `id` (string, required): Unique sapling identifier.
- `block` (string): Block ID placed in the world (defaults to `id`).
- `trees` (list, required): Tree IDs from `assets/trees/` to choose from when growing.
- `min_time` / `max_time` (numbers): Growth delay in seconds (defaults to 900–1800,
  i.e. 15–30 minutes).

```json
{
  "id": "oak_sapling",
  "block": "oak_sapling",
  "trees": ["oak", "fancy_oak", "tall_oak"],
  "min_time": 900,
  "max_time": 1800
}
```

Saplings are placed like other plants: they require a soil block below and are washed
away by fluids. Leaf loot tables can drop their matching sapling with a `chance` entry.
Growth timers only advance while the sapling's chunk is loaded, so leaving the area
pauses growth instead of skipping it. Remaining timers are saved in the world folder
and restored when the world is loaded again.

---

## 8. Caves (`assets/caves/`) & Features (`assets/features/`)

Caves and environmental features generate during initial chunk creation.

### Example: Caverns in Lua (`assets/caves/caverns.lua`)
```lua
return {
  id = "caverns",
  scale_xz = 0.022,
  scale_y = 0.038,
  octaves = 2,
  threshold = 0.62,
  y_min = 4,
  y_max = 50,
  seed_offset = 2048,
}
```

### Example: Water Lake in Lua (`assets/features/water_lake.lua`)
```lua
return {
  id = "water_lake",
  chance = 0.002,
  radius_min = 4,
  radius_max = 8,
  depth_min = 2,
  depth_max = 4,
  fluid = "water",
  barrier = "clay",
  biomes = { "plains", "forest", "mountain_meadow" },
}
```

---

## 9. Loot Tables (`assets/loot/`)

Defines item drops when blocks are broken or entities are killed.

### Example: Leaves Decay Loot (`assets/loot/blocks/leaves.json`)
```json
{
  "type": "block",
  "pools": [
    {
      "rolls": 1,
      "entries": [
        {
          "type": "item",
          "name": "stick",
          "count": 1,
          "chance": 0.2
        }
      ]
    }
  ]
}
```

---

## 10. Fuels (`assets/fuels/`)

Defines burn durations for furnace smelting.

### Example (`coal.json`)
```json
{
  "item": "coal",
  "burn_time": 80.0
}
```

---

## 11. Ores (`assets/ores/`)

Defines subterranean ore vein placement.

### Example (`iron.json`)
```json
{
  "id": "iron",
  "block": "iron_ore",
  "host": "stone",
  "vein_size": 8,
  "veins_per_chunk": 12,
  "y_min": 4,
  "y_max": 64
}
```

