# Block Game

A fast, highly-moddable voxel sandbox game and engine implemented in Python, ModernGL, and Pyglet. Features procedurally generated multi-biome worlds, a 6-channel multi-spectral dynamic lighting engine, cellular automaton fluids, survival mechanics, and client-server multiplayer.

---

## Quick Start

### 1. Requirements & Setup
Make sure you have Python 3.10+ and the required dependencies installed:

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Singleplayer
Launch the client directly to create or resume singleplayer worlds:

```bash
python3 play.py
```

### 3. Multiplayer
- **Run a Dedicated Server**:
  ```bash
  python3 -m server.network --port 25565 --world multiplayer --gamemode survival
  ```
- **Run the Split-Screen Multiplayer Test Harness**:
  ```bash
  ./run_mp_test.sh
  ```

### 4. Running the Test Suite
```bash
pytest
```

---

## Controls & Keybindings

| Key / Input | Action |
| :--- | :--- |
| `W`, `A`, `S`, `D` | Move (Walk / Strafe) |
| `Space` | Jump / Swim upward |
| `Left Shift` | Sneak (prevent falling off edges / bypass container open) |
| `Left Mouse Button` | Mine block / Attack entity |
| `Right Mouse Button` | Place block / Interact (open Crafting Table, Furnace, use Buckets) |
| `1` – `9` | Select hotbar slot |
| `E` | Open Inventory / Crafting Grid |
| `T` | Open In-Game Chat |
| `/` | Open Chat with command prompt |
| `V` or `L` | Cycle Vision Mode (Normal → UV → Thermal/IR → Full Spectrum) |
| `Esc` | Pause Menu / Resume Game |

---

## Detailed Documentation

Comprehensive documentation is organized in the [`docs/`](file:///home/nathan/block-game/docs) directory:

- 🎮 **[Features Documentation](docs/FEATURES.md)**:
  Deep dive into procedural terrain generation, multi-biome climate bands, cave carvers, surface lakes, dynamic 6-channel lighting (Skylight, RGB, UV, IR), fluid dynamics, tree shapes & leaf decay, flint knapping, block-hit crafting, and survival AI.

- 🛠️ **[Asset Creation Guide (JSON & Lua)](docs/ASSET_CREATION.md)**:
  Full specification and examples for authoring game content in either **JSON** or **Lua**:
  - Biomes (`assets/biomes/`)
  - Blocks (`assets/blocks/`)
  - Items (`assets/items/`)
  - Tags (`assets/tags/`)
  - Recipes (`assets/recipes/`)
  - Trees (`assets/trees/`)
  - Caves (`assets/caves/`)
  - Environmental Features (`assets/features/`)
  - Loot Tables (`assets/loot/`)
  - Ores (`assets/ores/`)
  - Fuels (`assets/fuels/`)

- 🌐 **[Multiplayer Guide & Server Setup](docs/MULTIPLAYER.md)**:
  Instructions for launching dedicated servers, CLI configuration flags, direct connecting, server list management, automated local split-screen testing, port forwarding, and in-game commands (`/tp`, `/biome`, `/time`, `/light`, `/spawn`, etc.).

