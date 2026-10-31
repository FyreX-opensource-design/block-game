import shlex


class CommandContext:
    def __init__(self, sender="Player", player=None, world=None, server=None, network_client=None):
        self.sender = sender
        self.player = player
        self.world = world
        self.server = server
        self.network_client = network_client
        self.responses = []  # list of (text, color)

    def reply(self, text, color=(255, 215, 0, 255)):
        self.responses.append((str(text), color))

    def error(self, text):
        self.responses.append((str(text), (255, 100, 100, 255)))

    def broadcast(self, text, color=(255, 215, 0, 255)):
        if self.server is not None:
            self.server.broadcast({"type": "chat", "sender": "", "text": str(text), "color": list(color)})
        else:
            self.reply(text, color)


class CommandHandler:
    def __init__(self):
        self.commands = {}
        self._register_defaults()

    def register(self, name, handler, description="", usage=""):
        name = name.lower().lstrip("/")
        self.commands[name] = {
            "name": name,
            "handler": handler,
            "description": description,
            "usage": usage or f"/{name}",
        }

    def execute(self, context, command_line):
        line = command_line.strip()
        if not line.startswith("/"):
            return False
        parts = line[1:].strip().split()
        if not parts:
            return False
        cmd_name = parts[0].lower()
        args = parts[1:]

        cmd = self.commands.get(cmd_name)
        if cmd is None:
            context.error(f"Unknown command: /{cmd_name}. Type /help for a list of commands.")
            return True

        try:
            cmd["handler"](context, args)
        except Exception as exc:
            context.error(f"Error executing /{cmd_name}: {exc}")
        return True

    def _register_defaults(self):
        self.register(
            "help",
            self._cmd_help,
            "Lists all available commands or usage for a specific command.",
            usage="/help [command]",
        )
        self.register(
            "seed",
            self._cmd_seed,
            "Displays the current world seed.",
            usage="/seed",
        )
        self.register(
            "pos",
            self._cmd_pos,
            "Shows your current coordinates.",
            usage="/pos",
        )
        self.register(
            "biome",
            self._cmd_biome,
            "Shows your current biome.",
            usage="/biome",
        )
        self.register(
            "tp",
            self._cmd_tp,
            "Teleport to coordinates (x y z).",
            usage="/tp <x> <y> <z>",
        )
        self.register(
            "say",
            self._cmd_say,
            "Broadcast an announcement to all players.",
            usage="/say <message>",
        )
        self.register(
            "light",
            self._cmd_light,
            "Query or set skylight and blocklight (RGB, UV, IR) levels.",
            usage="/light [levels | set <r> <g> <b> [uv] [ir] | clear]",
        )
        self.register(
            "vision",
            self._cmd_vision,
            "Switch vision mode (normal, uv, thermal/ir, spectrum/debug).",
            usage="/vision <normal | uv | thermal | spectrum>",
        )
        self.register(
            "time",
            self._cmd_time,
            "Query or set world time (day, night, noon, midnight, or ticks 0-24000).",
            usage="/time <set <day|night|noon|midnight|number> | query>",
        )
        self.register(
            "spawn",
            self._cmd_spawn,
            "Spawn a mob entity (e.g. /spawn pig).",
            usage="/spawn <mob_type> [x y z]",
        )
        self.register(
            "setworldspawn",
            self._cmd_setworldspawn,
            "Set the world spawn point to your current position or specified coordinates.",
            usage="/setworldspawn [x [y] z]",
        )
        self.register(
            "worldspawn",
            self._cmd_worldspawn,
            "Query or set the world spawn point.",
            usage="/worldspawn [query | set [x [y] z]]",
        )
        self.register(
            "gamemode",
            self._cmd_gamemode,
            "Sets your game mode to creative or survival, or another player's to the given mode.",
            usage="/gamemode <creative | survival> [player]",
        )

    def _cmd_help(self, ctx, args):
        if args:
            target = args[0].lower().lstrip("/")
            cmd = self.commands.get(target)
            if cmd is None:
                ctx.error(f"Unknown command: /{target}")
            else:
                ctx.reply(f"{cmd['usage']} - {cmd['description']}")
            return

        ctx.reply("--- Available Commands ---", (180, 220, 255, 255))
        for name in sorted(self.commands.keys()):
            cmd = self.commands[name]
            ctx.reply(f"  {cmd['usage']} - {cmd['description']}")

    def _cmd_seed(self, ctx, args):
        if ctx.world is not None:
            ctx.reply(f"World Seed: {ctx.world.seed}")
        elif ctx.server is not None and ctx.server.world is not None:
            ctx.reply(f"World Seed: {ctx.server.world.seed}")
        else:
            ctx.error("World not available.")

    def _cmd_pos(self, ctx, args):
        if ctx.player is not None and getattr(ctx.player, "pos", None) is not None:
            p = ctx.player.pos
            ctx.reply(f"Position: X: {p[0]:.2f}, Y: {p[1]:.2f}, Z: {p[2]:.2f}")
        else:
            ctx.error("Player position not available.")

    def _cmd_biome(self, ctx, args):
        world = ctx.world or (ctx.server.world if ctx.server is not None else None)
        if ctx.player is not None and getattr(ctx.player, "pos", None) is not None and world is not None and getattr(world, "biomes", None):
            from server.biomes import biome_at
            p = ctx.player.pos
            b = biome_at(int(p[0]), int(p[2]), getattr(world, "seed", 0), world.biomes)
            name = getattr(b, "name", None) or b.id.replace("_", " ").title()
            ctx.reply(f"Biome: {name} ({b.id})")
        else:
            ctx.error("Biome information not available.")

    def _cmd_tp(self, ctx, args):
        if len(args) < 3:
            ctx.error("Usage: /tp <x> <y> <z>")
            return
        try:
            x, y, z = float(args[0]), float(args[1]), float(args[2])
        except ValueError:
            ctx.error("Coordinates must be numbers.")
            return

        if ctx.player is not None:
            ctx.player.pos = [x, y, z]
            ctx.player.vel = [0.0, 0.0, 0.0]
            ctx.reply(f"Teleported to {x:.1f}, {y:.1f}, {z:.1f}")
        else:
            ctx.error("Player not available.")

    def _cmd_say(self, ctx, args):
        if not args:
            ctx.error("Usage: /say <message>")
            return
        msg = " ".join(args)
        prefix = f"[{ctx.sender}] " if ctx.sender else "[Server] "
        ctx.broadcast(f"{prefix}{msg}", (255, 180, 100, 255))

    def _cmd_light(self, ctx, args):
        world = ctx.world or (ctx.server.world if ctx.server else None)
        if world is None:
            ctx.error("World not available.")
            return

        if not args or args[0].lower() in ("get", "levels", "status"):
            if ctx.player is not None and getattr(ctx.player, "pos", None) is not None:
                x, y, z = int(ctx.player.pos[0]), int(ctx.player.pos[1]), int(ctx.player.pos[2])
            else:
                x, y, z = 0, 64, 0
            info = world.get_light(x, y, z)
            sky = info["sky"]
            r, g, b, uv, ir = info["block"]
            ctx.reply(
                f"Light at ({x}, {y}, {z}): Skylight: {sky}/15 | Blocklight RGB: ({r}, {g}, {b}) UV: {uv} IR: {ir}",
                (255, 230, 150, 255),
            )
            return

        sub = args[0].lower()
        if sub == "set":
            if len(args) < 4:
                ctx.error("Usage: /light set <r> <g> <b> [uv] [ir]")
                return
            try:
                r = int(args[1])
                g = int(args[2])
                b = int(args[3])
                uv = int(args[4]) if len(args) > 4 else 0
                ir = int(args[5]) if len(args) > 5 else 0
            except ValueError:
                ctx.error("Light levels must be integers (0..15).")
                return
            if ctx.player is not None and getattr(ctx.player, "pos", None) is not None:
                x, y, z = int(ctx.player.pos[0]), int(ctx.player.pos[1]), int(ctx.player.pos[2])
            else:
                x, y, z = 0, 64, 0
            world.set_blocklight(x, y, z, r, g, b, uv, ir)
            ctx.reply(f"Set blocklight at ({x}, {y}, {z}) to RGB:({r},{g},{b}) UV:{uv} IR:{ir}")
            return

        if sub == "clear":
            if ctx.player is not None and getattr(ctx.player, "pos", None) is not None:
                x, y, z = int(ctx.player.pos[0]), int(ctx.player.pos[1]), int(ctx.player.pos[2])
            else:
                x, y, z = 0, 64, 0
            if getattr(world, "lighting", None) is not None:
                world.lighting.clear_blocklight(x, y, z)
            ctx.reply(f"Cleared light source at ({x}, {y}, {z})")
            return

        ctx.error("Usage: /light [levels | set <r> <g> <b> [uv] [ir] | clear]")

    def _cmd_vision(self, ctx, args):
        mode_map = {
            "normal": 0,
            "0": 0,
            "uv": 1,
            "ultraviolet": 1,
            "1": 1,
            "ir": 2,
            "infrared": 2,
            "thermal": 2,
            "2": 2,
            "spectrum": 3,
            "debug": 3,
            "3": 3,
        }
        mode_names = ["Normal", "UV Vision", "Infrared / Thermal", "Full Spectrum"]
        if not args:
            ctx.reply("Usage: /vision <normal | uv | thermal | spectrum>")
            return
        arg = args[0].lower()
        if arg not in mode_map:
            ctx.error(f"Unknown vision mode: {arg}. Choose from: normal, uv, thermal, spectrum")
            return
        mode = mode_map[arg]
        app = getattr(ctx, "app", None)
        if app is not None:
            app.vision_mode = mode
            if hasattr(app, "renderer") and app.renderer is not None:
                app.renderer.vision_mode = mode
        ctx.reply(f"Vision Mode set to: {mode_names[mode]}")

    def _cmd_time(self, ctx, args):
        world = ctx.world
        if world is None and getattr(ctx, "server", None) is not None:
            world = ctx.server.world

        if not args or args[0].lower() == "query":
            if world is not None:
                ticks = int(getattr(world, "day_time", 0)) % 24000
                level = round(world.get_sky_light_level(), 1) if hasattr(world, "get_sky_light_level") else 15.0
                ctx.reply(f"World Time: {ticks} ticks (Skylight level: {level}/15)")
            else:
                ctx.error("World not available.")
            return

        sub = args[0].lower()
        if sub == "set":
            if len(args) < 2:
                ctx.error("Usage: /time set <day|night|noon|midnight|sunrise|sunset|number>")
                return
            target = args[1].lower()
            if target == "day":
                new_time = 1000.0
            elif target == "noon":
                new_time = 6000.0
            elif target == "sunset":
                new_time = 12000.0
            elif target == "night":
                new_time = 13000.0
            elif target == "midnight":
                new_time = 18000.0
            elif target == "sunrise":
                new_time = 23000.0
            else:
                try:
                    new_time = float(target) % 24000.0
                except ValueError:
                    ctx.error(f"Invalid time: {target}. Must be day, night, noon, midnight, or a number.")
                    return

            if world is not None:
                world.day_time = new_time
                ctx.reply(f"Set time to {int(new_time)} ticks.")
                app = getattr(ctx, "app", None)
                if app is not None and getattr(app, "_persist_world", None):
                    app._persist_world()
                server = getattr(ctx, "server", None)
                if server is not None and hasattr(server, "broadcast"):
                    server.broadcast({"type": "time_update", "day_time": new_time})
            else:
                ctx.error("World not available.")
        else:
            ctx.error("Usage: /time <set <day|night|noon|midnight|number> | query>")

    def _cmd_spawn(self, ctx, args):
        if not args:
            ctx.error("Usage: /spawn <mob_type> [x y z]")
            return
        mob_type = args[0].lower()
        world = ctx.world or (ctx.server.world if ctx.server else None)
        if world is None:
            ctx.error("World not available.")
            return

        if len(args) >= 4:
            try:
                x, y, z = float(args[1]), float(args[2]), float(args[3])
            except ValueError:
                ctx.error("Coordinates must be numbers.")
                return
        elif ctx.player is not None and getattr(ctx.player, "pos", None) is not None:
            x, y, z = ctx.player.pos[0], ctx.player.pos[1], ctx.player.pos[2]
        else:
            ctx.error("Specify x y z coordinates or run as player.")
            return

        mob = world.spawn_mob(mob_type, pos=[x, y, z])
        if ctx.server is not None:
            ctx.server.broadcast(
                {
                    "type": "spawn_mob",
                    "id": mob.id,
                    "mob_type": mob.mob_type,
                    "pos": mob.pos,
                    "vel": mob.vel,
                    "yaw": mob.yaw,
                    "pitch": mob.pitch,
                    "health": mob.health,
                }
            )
        ctx.reply(f"Spawned {mob_type} at {x:.1f}, {y:.1f}, {z:.1f} (id: {mob.id})")

    def _cmd_setworldspawn(self, ctx, args):
        world = ctx.world or (ctx.server.world if ctx.server else None)
        if world is None:
            ctx.error("World not available.")
            return

        if not args:
            if ctx.player is not None and getattr(ctx.player, "pos", None) is not None:
                x, y, z = float(ctx.player.pos[0]), float(ctx.player.pos[1]), float(ctx.player.pos[2])
                sp = world.set_spawn_point(x, y, z)
            else:
                ctx.error("Usage: /setworldspawn <x> [y] <z>")
                return
        elif len(args) == 2:
            try:
                x, z = float(args[0]), float(args[1])
            except ValueError:
                ctx.error("Coordinates must be numbers.")
                return
            sp = world.set_spawn_point(x, None, z)
        elif len(args) >= 3:
            try:
                x, y, z = float(args[0]), float(args[1]), float(args[2])
            except ValueError:
                ctx.error("Coordinates must be numbers.")
                return
            sp = world.set_spawn_point(x, y, z)
        else:
            ctx.error("Usage: /setworldspawn [x [y] z]")
            return

        app = getattr(ctx, "app", None)
        if app is not None and getattr(app, "_persist_world", None):
            app._persist_world()
        server = getattr(ctx, "server", None)
        if server is not None and getattr(server, "world_info", None) is not None and getattr(server, "catalog", None) is not None:
            server.world_info.spawn = list(sp)
            server.catalog.write(server.world_info)

        ctx.reply(f"Set world spawn point to ({sp[0]:.1f}, {sp[1]:.1f}, {sp[2]:.1f})")

    def _cmd_worldspawn(self, ctx, args):
        world = ctx.world or (ctx.server.world if ctx.server else None)
        if world is None:
            ctx.error("World not available.")
            return
        if not args or args[0].lower() in ("query", "get"):
            sp = world.get_spawn_point()
            ctx.reply(f"World spawn point is ({sp[0]:.1f}, {sp[1]:.1f}, {sp[2]:.1f})")
            return
        if args[0].lower() == "set":
            self._cmd_setworldspawn(ctx, args[1:])
            return
        self._cmd_setworldspawn(ctx, args)

    def _cmd_gamemode(self, ctx, args):
        if not args:
            ctx.error("Usage: /gamemode <creative | survival> [player]")
            return
        mode = str(args[0]).lower()
        if mode not in ("creative", "survival"):
            ctx.error("Game mode must be creative or survival.")
            return

        player = ctx.player
        if len(args) > 1:
            server = ctx.server
            clients = getattr(server, "clients", None) if server is not None else None
            target_client = None
            if clients:
                wanted = str(args[1]).lower()
                for client in clients.values():
                    if getattr(client, "alive", True) and str(getattr(client, "username", "")).lower() == wanted:
                        target_client = client
                        break
            if target_client is None or not getattr(target_client, "uuid", None):
                ctx.error(f"Player not found: {args[1]}")
                return
            player = ctx.world.players.get(getattr(target_client, "uuid", "")) if ctx.world is not None else None
            if player is None:
                ctx.error(f"Player not found: {args[1]}")
                return

        if player is None or getattr(player, "uuid", None) is None:
            ctx.error("Player not available.")
            return

        player.gamemode = mode

        client = None
        server = ctx.server
        if server is not None:
            for candidate in list(getattr(server, "clients", {}).values()) + list(getattr(server, "pending_clients", [])):
                if getattr(candidate, "uuid", None) == getattr(player, "uuid", None) and getattr(candidate, "alive", True):
                    client = candidate
                    break
        if client is not None:
            client.send({"type": "gamemode", "gamemode": mode})

        if ctx.player is player:
            ctx.reply(f"Game mode set to {mode}.")
        else:
            ctx.reply(f"Game mode set to {mode} for {getattr(player, 'uuid', args[1])}.")
