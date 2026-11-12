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

The preferred way to host a dedicated server is directly through `play.py`:

```bash
# Basic start (binds to 0.0.0.0:25565, world "multiplayer", survival mode)
./.venv/bin/python3 play.py --server

# With custom parameters
./.venv/bin/python3 play.py --server --port 25565 --world survival_world --seed 42 --gamemode survival

# Alternatively via the server module:
./.venv/bin/python3 -m server.network --port 25565
```

When started, the server automatically detects your local network interface IPs and displays them in the console so other devices know what address to connect to.

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

### Quick Connect from Command Line (Preferred)
```bash
# Connect to a server on the same machine:
./.venv/bin/python3 play.py -m
# or
./.venv/bin/python3 play.py --connect

# Connect to a server on another device (replace with the host's actual IP):
./.venv/bin/python3 play.py --connect 192.168.1.50:25565
```

### Joining Through the Main Menu GUI
1. Launch the game client:
   ```bash
   ./.venv/bin/python3 play.py
   ```
2. In the Main Menu, select **Multiplayer**.
3. Choose one of the connection methods:
   - **Local Server**: By default, "Local Server" (`127.0.0.1:25565`) is selected. Click **Join Server** to join a server running on your own machine.
   - **Direct Connect**:
     - Type the server address: e.g. `127.0.0.1:25565` (local machine) or the host's LAN IP (e.g. `192.168.1.50:25565`).
     - Click **Join** to immediately connect.
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
| `/gamemode` | `/gamemode <creative \| survival> [player]` | Sets the game mode for a player (defaults to yourself, or specify a player by name). |

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

---

## 7. LAN & Mobile Hotspot Troubleshooting

### Error 101 (`Network is unreachable`)
- **Broadcast Address**: In IPv4, addresses ending in `.255` (e.g. `10.46.100.255`) are subnet broadcast addresses, not host IPs. Connecting to a `.255` address will fail with Error 101. Run `hostname -I` or `ip route get 1.1.1.1` to find your host's actual IP address.
- **Local Machine**: Always use `127.0.0.1:25565` (or `./.venv/bin/python3 play.py -m`) when connecting to a server running on the same machine.

### Error 113 (`No route to host`) / Mobile Hotspot AP Isolation
Many mobile phones (Android / iOS) enforce **AP Isolation** on their Wi-Fi hotspots, blocking tethered devices from talking directly to each other. Furthermore, school or public Wi-Fi networks block client-to-client traffic.

**Recommended Solution — Host directly from your Linux laptop**:
1. On the host laptop, broadcast your own Wi-Fi hotspot:
   ```bash
   nmcli device wifi hotspot ssid BlockGame password 12345678
   ```
2. Have the second device connect to the `BlockGame` Wi-Fi with password `12345678`.
3. Start the server on the laptop (`./.venv/bin/python3 play.py --server`).
4. On the second device, connect to the laptop's hotspot IP (usually `10.42.0.1:25565`):
   ```bash
   ./.venv/bin/python3 play.py --connect 10.42.0.1:25565
   ```
*(When finished, stop the hotspot with `nmcli connection down Hotspot`)*.


