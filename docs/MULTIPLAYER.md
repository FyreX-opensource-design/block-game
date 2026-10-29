# Block Game - Multiplayer Guide & Server Setup

This guide explains how to host a dedicated multiplayer server, configure world options, connect clients, and perform automated local split-screen testing.

---

## 1. Network Architecture Overview

Block Game uses an authoritative client-server architecture built on standard TCP sockets with asynchronous non-blocking event loops:

- **Server (`server/network.py`)**:
  - The [`MultiplayerServer`](file:///home/nathan/block-game/server/network.py) manages connected clients, chunk generation, world simulation (fluids, leaf decay, mobs, day/night cycles), entity synchronization, and chat routing.
  - Chunks are compressed and streamed on demand as players move across chunk boundaries.
  - Delta updates broadcast block placements, block breaks, entity positions, health, damage events, and chat.
- **Client (`client/network.py`)**:
  - The [`NetworkClient`](file:///home/nathan/block-game/client/network.py) communicates with the server in a background thread, maintaining interpolation buffers for smooth remote player movement and nametag rendering.

---

## 2. Launching a Dedicated Server

You can run the server in headless dedicated mode from the command line:

```bash
# Basic start (binds to 0.0.0.0:25565, world name "multiplayer", survival mode)
python3 -m server.network

# With custom parameters
python3 -m server.network --host 0.0.0.0 --port 25565 --world survival_world --seed 42 --gamemode survival
```

### Server CLI Flags
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--host` | `0.0.0.0` | IP address to bind to (`0.0.0.0` listens on all network interfaces; `127.0.0.1` listens locally only). |
| `--port` | `25565` | TCP port number to listen on. |
| `--world` | `multiplayer` | Name of the world folder (stored under `saves/`). |
| `--seed` | `None` | Integer seed for deterministic terrain generation (random if omitted). |
| `--gamemode` | `survival` | Default game mode for joining players: `survival` or `creative`. |
| `--saves-dir` | `saves/` | Custom directory path for world saves and databases. |

### Stopping the Server
Press `Ctrl+C` in the server terminal to cleanly flush all modified chunks, save player inventories, and disconnect clients.

---

## 3. Joining a Server from the Client

1. Launch the game client:
   ```bash
   python3 play.py
   ```
2. In the Main Menu, select **Multiplayer**.
3. Choose one of the connection methods:
   - **Direct Connect**:
     - Type the server address: e.g. `127.0.0.1:25565` (local machine) or `192.168.1.50:25565` (LAN).
     - Click **Connect** to immediately join.
   - **Server List / Add Server**:
     - Click **Add Server**.
     - Enter a **Server Name** and **Address**.
     - Saved servers persist across restarts in `saves/servers.json`.
     - Select a server from the list and click **Join Server**.

---

## 4. Automated Split-Screen Testing (`run_mp_test.py`)

For developers testing multiplayer interactions (combat, block syncing, lighting, player models, animations) on a single machine, the repository includes an automated testing harness:

```bash
# Using the helper shell script:
./run_mp_test.sh

# Or using Python directly:
python3 run_mp_test.py
```

### What `run_mp_test.py` does:
1. Spawns a dedicated server on port `25565` with a fresh test world (`saves/mp_test/`).
2. Launches **Client 1** (`Player1`) positioned on the left side of your screen.
3. Launches **Client 2** (`Player2`) positioned on the right side of your screen.
4. Spawns both players in close proximity so they can immediately see each other's 3D player models, animations, held items, and nametags.
5. Closing either client window or pressing `Ctrl+C` cleanly shuts down all processes.

---

## 5. In-Game Chat & Server Commands

Players can type `T` to open chat or `/` to enter commands:

| Command | Usage | Description |
| :--- | :--- | :--- |
| `/help` | `/help [command]` | Lists all available commands or shows detailed usage for a command. |
| `/biome` | `/biome` | Displays the player's current biome name and ID. |
| `/pos` | `/pos` | Prints player coordinates (`X`, `Y`, `Z`). |
| `/seed` | `/seed` | Prints the server's world generation seed. |
| `/tp` | `/tp <x> <y> <z>` | Teleports the player to the specified coordinates. |
| `/say` | `/say <message>` | Broadcasts a server announcement in gold text. |
| `/time` | `/time <set <day\|night\|noon\|midnight\|ticks> \| query>` | Changes or queries current world time. |
| `/light` | `/light [levels \| set <r> <g> <b> [uv] [ir] \| clear]` | Queries or sets skylight and blocklight at the player's position. |
| `/vision` | `/vision <normal \| uv \| thermal \| spectrum>` | Switches spectral vision mode. |
| `/spawn` | `/spawn <mob_type> [x y z]` | Spawns a mob entity (e.g. `/spawn pig`). |

---

## 6. Port Forwarding & Firewall

To allow external players over the internet to connect:

1. **Firewall**: Ensure incoming TCP traffic on your selected port (default `25565`) is allowed:
   ```bash
   # On Linux UFW:
   sudo ufw allow 25565/tcp
   ```
2. **Router Port Forwarding**:
   - Access your router's admin panel (typically `192.168.1.1` or `192.168.0.1`).
   - Forward external port `25565` (TCP) to your host machine's internal local IP address.
   - Share your public IP address with players (`curl ifconfig.me`).

