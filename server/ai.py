import math
import random


def is_position_hazardous(world, px: float, py: float, pz: float, max_fall: int = 4, already_in_water: bool = False, mob=None) -> bool:
    """Check if stepping to (px, py, pz) would result in a fall > max_fall blocks or entering fluid."""
    if world is None:
        return False
    from server.fluid import is_water

    bx = math.floor(px)
    by = math.floor(py)
    bz = math.floor(pz)

    blocks = getattr(world, "blocks_movement", world.is_solid)

    # For climbing mobs (e.g. spiders), navigating ceilings and scaling walls is permitted
    if mob is not None and getattr(mob, "can_climb", False):
        if not already_in_water and (is_water(world.get_block(bx, by, bz)) or is_water(world.get_block(bx, by + 1, bz))):
            return True
        from server.blocks import is_stalagmite
        if is_stalagmite(world.get_block(bx, by, bz)):
            return True
        if getattr(mob, "on_ceiling", False):
            # Ceiling crawl: safe as long as the ceiling above is solid
            check_ceil_y = math.floor(py + getattr(mob, "height", 0.9) + 0.3)
            return not blocks(bx, check_ceil_y, bz)
        # On walls or ground: climbing mobs do not avoid drop falls because they can climb walls
        return False

    # If the target cell itself is solid (a 1-block step up), check 1 block higher
    check_y = by
    if blocks(bx, by, bz):
        if blocks(bx, by + 1, bz):
            # 2-block high wall - blocked, not a fall/fluid hazard per se, but impassable
            return False
        check_y = by + 1

    # Check if the air/body space at check_y has fluid or sharp stalagmites
    if not already_in_water:
        if is_water(world.get_block(bx, check_y, bz)) or is_water(world.get_block(bx, check_y + 1, bz)):
            return True

    from server.blocks import is_stalagmite
    if is_stalagmite(world.get_block(bx, check_y, bz)):
        return True

    # Trace downward from check_y - 1 down to check_y - 1 - max_fall
    # A drop of 0 blocks means ground is at check_y - 1.
    # A drop of `d` blocks means ground is at check_y - 1 - d.
    # So drops of 0, 1, 2, ..., max_fall have ground at check_y - 1 - d for d in 0..max_fall.
    found_ground = False
    for drop in range(0, max_fall + 1):
        gy = check_y - 1 - drop
        blk = world.get_block(bx, gy, bz)
        if not already_in_water and is_water(blk):
            return True
        if is_stalagmite(blk):
            return True
        if blocks(bx, gy, bz):
            found_ground = True
            break

    if not found_ground:
        # No solid ground within max_fall blocks -> fall > max_fall blocks!
        return True

    return False


def is_direction_safe(mob, yaw: float, world=None, lookahead: float = 1.0, max_fall: int = 4) -> bool:
    """Return True if moving along `yaw` from mob's current position avoids fluids and falls > max_fall."""
    if world is None:
        world = getattr(mob, "world", None)
    if world is None:
        return True

    already_in_water = mob.in_water(world) if hasattr(mob, "in_water") else False
    fx = -math.sin(yaw)
    fz = -math.cos(yaw)
    py = mob.pos[1]

    # Check intermediate and lookahead points along the heading
    steps = [0.6, lookahead] if lookahead > 0.6 else [lookahead]
    for dist in steps:
        tx = mob.pos[0] + fx * dist
        tz = mob.pos[2] + fz * dist
        if is_position_hazardous(world, tx, py, tz, max_fall=max_fall, already_in_water=already_in_water, mob=mob):
            return False
    return True


def pick_safe_yaw(mob, preferred_yaw: float, world=None, lookahead: float = 1.2, max_fall: int = 4):
    """Try `preferred_yaw` first; if hazardous, sample alternative headings and return a safe one (or None)."""
    if world is None:
        world = getattr(mob, "world", None)
    if world is None or is_direction_safe(mob, preferred_yaw, world=world, lookahead=lookahead, max_fall=max_fall):
        return preferred_yaw

    # Try offsets fanning out from preferred_yaw
    offsets = [0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.1, -2.1, 2.6, -2.6, math.pi]
    random.shuffle(offsets)
    # Sort slightly to prefer smaller turns when possible, with randomness
    for offset in offsets:
        cand = (preferred_yaw + offset + math.pi) % (2 * math.pi) - math.pi
        if is_direction_safe(mob, cand, world=world, lookahead=lookahead, max_fall=max_fall):
            return cand
    return None


class Goal:
    def __init__(self, mob):
        self.mob = mob

    def can_start(self) -> bool:
        return False

    def can_continue(self) -> bool:
        return self.can_start()

    def start(self):
        pass

    def stop(self):
        pass

    def tick(self, dt: float):
        pass


class GoalSelector:
    def __init__(self):
        self.goals = []  # list of (priority, goal)
        self.active_goal = None

    def add_goal(self, priority: int, goal: Goal):
        self.goals.append((priority, goal))
        self.goals.sort(key=lambda item: item[0])

    def tick(self, dt: float):
        # 1. Check if active goal can continue
        if self.active_goal is not None:
            if not self.active_goal.can_continue():
                self.active_goal.stop()
                self.active_goal = None

        # 2. Check in priority order (lowest number = highest priority)
        best_goal = None
        for priority, goal in self.goals:
            if self.active_goal is not None:
                active_priority = next(p for p, g in self.goals if g is self.active_goal)
                if priority >= active_priority:
                    # Current active goal has higher or equal priority; don't preempt
                    break
            if goal.can_start():
                best_goal = goal
                break

        if best_goal is not None and best_goal is not self.active_goal:
            if self.active_goal is not None:
                self.active_goal.stop()
            self.active_goal = best_goal
            self.active_goal.start()

        if self.active_goal is not None:
            self.active_goal.tick(dt)


class PanicGoal(Goal):
    """Triggered when the mob takes damage. Causes the mob to flee rapidly."""

    def __init__(self, mob, speed_mult=2.2):
        super().__init__(mob)
        self.speed_mult = speed_mult
        self.repath_timer = 0.0
        self.target_yaw = 0.0

    def can_start(self) -> bool:
        return bool(self.mob.panic_timer > 0)

    def can_continue(self) -> bool:
        return bool(self.mob.panic_timer > 0)

    def start(self):
        self._pick_flee_direction()

    def stop(self):
        self.mob.desired_speed = 0.0

    def _pick_flee_direction(self):
        self.mob.desired_speed = getattr(self.mob, "run_speed", 2.8)
        if self.mob.panic_source is not None:
            sx, sy, sz = self.mob.panic_source
            dx = self.mob.pos[0] - sx
            dz = self.mob.pos[2] - sz
            if math.hypot(dx, dz) > 1e-4:
                base_angle = math.atan2(-dx, -dz)
            else:
                base_angle = random.uniform(-math.pi, math.pi)
            # Add random jitter to flee heading (-45 to +45 deg)
            preferred = base_angle + random.uniform(-0.75, 0.75)
        else:
            preferred = random.uniform(-math.pi, math.pi)

        safe_yaw = pick_safe_yaw(self.mob, preferred)
        if safe_yaw is not None:
            self.target_yaw = safe_yaw
        else:
            self.target_yaw = preferred
            self.mob.desired_speed = 0.0

        self.repath_timer = random.uniform(0.8, 1.4)

    def tick(self, dt: float):
        self.repath_timer -= dt
        if self.repath_timer <= 0 or not is_direction_safe(self.mob, self.target_yaw):
            self._pick_flee_direction()

        self.mob.target_yaw = self.target_yaw
        if is_direction_safe(self.mob, self.target_yaw):
            self.mob.desired_speed = getattr(self.mob, "run_speed", 2.8)
        else:
            self.mob.desired_speed = 0.0


class EatGrassGoal(Goal):
    """Goal for a sheared sheep to eat a 'grass' block (not 'muddy_grass'), turning it into 'dirt' and regrowing wool."""

    def __init__(self, mob):
        super().__init__(mob)
        self.eat_timer = 0.0
        self.eating = False

    def _find_edible_grass_under(self):
        world = getattr(self.mob, "world", None)
        if world is None:
            return None
        bx = math.floor(self.mob.pos[0])
        by = math.floor(self.mob.pos[1] - 0.1)
        bz = math.floor(self.mob.pos[2])
        # Only eat actual 'grass', explicitly NOT 'muddy_grass'
        if world.get_block(bx, by, bz) == "grass":
            return (bx, by, bz)
        return None

    def _find_nearby_grass(self, radius: int = 4):
        world = getattr(self.mob, "world", None)
        if world is None:
            return None
        bx = math.floor(self.mob.pos[0])
        by = math.floor(self.mob.pos[1] - 0.1)
        bz = math.floor(self.mob.pos[2])
        best = None
        best_dist = 999999.0
        for dy in (-1, 0, 1):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx, ty, tz = bx + dx, by + dy, bz + dz
                    if world.get_block(tx, ty, tz) == "grass":
                        # Must have non-solid air/plant above it so sheep can stand on it
                        above = world.get_block(tx, ty + 1, tz)
                        if not world.is_solid(tx, ty + 1, tz) and above not in ("water",):
                            d = dx * dx + dz * dz + abs(dy) * 2
                            if d < best_dist:
                                best_dist = d
                                best = (tx + 0.5, ty + 1.0, tz + 0.5)
        return best

    def can_start(self) -> bool:
        if getattr(self.mob, "panic_timer", 0.0) > 0:
            return False
        if not getattr(self.mob, "sheared", False):
            return False
        if getattr(self.mob, "regrow_cooldown", 0.0) > 0:
            return False
        if self._find_edible_grass_under() is not None:
            return True
        if self._find_nearby_grass() is not None:
            return True
        return False

    def can_continue(self) -> bool:
        if getattr(self.mob, "panic_timer", 0.0) > 0:
            return False
        if not getattr(self.mob, "sheared", False):
            return False
        if self.eating:
            return self.eat_timer > 0 and self._find_edible_grass_under() is not None
        return self.can_start()

    def start(self):
        if self._find_edible_grass_under() is not None:
            self.eating = True
            self.eat_timer = 0.8
            self.mob.desired_speed = 0.0
            self.mob.pitch = -0.8  # Head bowed down eating grass
        else:
            self.eating = False
            self.eat_timer = 0.0

    def stop(self):
        self.eating = False
        self.eat_timer = 0.0
        self.mob.desired_speed = 0.0
        self.mob.pitch = 0.0

    def tick(self, dt: float):
        world = getattr(self.mob, "world", None)
        if world is None:
            return

        under = self._find_edible_grass_under()
        if under is not None and not self.eating:
            self.eating = True
            self.eat_timer = 0.8
            self.mob.desired_speed = 0.0
            self.mob.pitch = -0.8

        if self.eating:
            self.mob.desired_speed = 0.0
            self.mob.pitch = -0.8
            self.eat_timer -= dt
            if self.eat_timer <= 0:
                if under is not None and world.get_block(*under) == "grass":
                    world.set_block(under[0], under[1], under[2], "dirt")
                    self.mob.sheared = False
                    self.mob.regrow_cooldown = 0.0
                self.eating = False
                self.mob.pitch = 0.0
            return

        # Walk toward nearby grass block
        target = self._find_nearby_grass()
        if target is None:
            self.mob.desired_speed = 0.0
            return
        dx = target[0] - self.mob.pos[0]
        dz = target[2] - self.mob.pos[2]
        if math.hypot(dx, dz) > 0.1:
            yaw = math.atan2(-dx, -dz)
            safe_yaw = pick_safe_yaw(self.mob, yaw, world=world)
            if safe_yaw is not None:
                self.mob.target_yaw = safe_yaw
                self.mob.desired_speed = getattr(self.mob, "walk_speed", 1.2)
            else:
                self.mob.desired_speed = 0.0


class WanderGoal(Goal):
    """Idle wandering: pauses for random intervals, then walks in a random direction."""

    def __init__(self, mob):
        super().__init__(mob)
        self.state = "idle"  # "idle" or "walk"
        self.timer = random.uniform(1.0, 3.0)
        self.target_yaw = getattr(mob, "yaw", 0.0)
        self.stuck_timer = 0.0

    def can_start(self) -> bool:
        return getattr(self.mob, "panic_timer", 0.0) <= 0

    def can_continue(self) -> bool:
        return getattr(self.mob, "panic_timer", 0.0) <= 0

    def start(self):
        self.state = "idle"
        self.timer = random.uniform(1.5, 4.0)
        self.mob.desired_speed = 0.0

    def stop(self):
        self.mob.desired_speed = 0.0

    def tick(self, dt: float):
        self.timer -= dt

        if self.state == "idle":
            self.mob.desired_speed = 0.0
            if self.timer <= 0:
                # Switch to walking
                turn = random.uniform(-1.8, 1.8)
                preferred_yaw = (self.mob.yaw + turn + math.pi) % (2 * math.pi) - math.pi
                safe_yaw = pick_safe_yaw(self.mob, preferred_yaw)
                if safe_yaw is not None:
                    self.state = "walk"
                    self.timer = random.uniform(2.0, 4.5)
                    self.mob.desired_speed = getattr(self.mob, "walk_speed", 1.2)
                    self.target_yaw = safe_yaw
                    self.mob.target_yaw = self.target_yaw
                    self.stuck_timer = 0.0
                else:
                    self.timer = random.uniform(1.0, 2.5)

        elif self.state == "walk":
            if not is_direction_safe(self.mob, self.target_yaw):
                safe_yaw = pick_safe_yaw(self.mob, self.target_yaw)
                if safe_yaw is not None:
                    self.target_yaw = safe_yaw
                    self.mob.target_yaw = self.target_yaw
                else:
                    self.state = "idle"
                    self.timer = random.uniform(1.5, 3.5)
                    self.mob.desired_speed = 0.0
                    return

            self.mob.desired_speed = getattr(self.mob, "walk_speed", 1.2)
            self.mob.target_yaw = self.target_yaw

            # Check if stuck against a wall
            hspeed = math.hypot(self.mob.vel[0], self.mob.vel[2])
            if hspeed < 0.1 and self.mob.desired_speed > 0:
                self.stuck_timer += dt
                if self.stuck_timer > 0.6:
                    # Switch to idle early to pick new direction next time
                    self.state = "idle"
                    self.timer = random.uniform(1.5, 3.5)
                    self.mob.desired_speed = 0.0
            else:
                self.stuck_timer = max(0.0, self.stuck_timer - dt)

            if self.timer <= 0:
                self.state = "idle"
                self.timer = random.uniform(2.0, 5.0)
                self.mob.desired_speed = 0.0


class ZombieAttackGoal(Goal):
    """Hostile attack goal for Zombie:

    - Detects closest alive survival player within target_range (16 blocks).
    - Chases the player at mob.run_speed.
    - Attacks when within melee reach (e.g. 1.5 blocks), applying damage, knockback, and hurt_time.
    """

    def __init__(self, mob, target_range=16.0, reach=1.5, damage=3.0, cooldown=1.0):
        super().__init__(mob)
        self.target_range = float(target_range)
        self.reach = float(reach)
        self.damage = float(damage)
        self.cooldown = float(cooldown)
        self.attack_timer = 0.0
        self.target = None
        self.repath_timer = 0.0

    def _get_target_candidates(self):
        world = getattr(self.mob, "world", None)
        if world is None:
            return []
        players = []
        if hasattr(world, "players") and world.players:
            players.extend(world.players.values())
        single = getattr(world, "_singleplayer_player", None)
        if single is not None and single not in players:
            players.append(single)
        return players

    def _find_closest_player(self):
        candidates = self._get_target_candidates()
        best_p = None
        best_dist_sq = self.target_range * self.target_range
        mx, my, mz = self.mob.pos
        for p in candidates:
            if not getattr(p, "alive", True) or getattr(p, "dead", False):
                continue
            if not getattr(p, "survival", True):
                continue
            px, py, pz = p.pos
            dx = px - mx
            dy = py - my
            dz = pz - mz
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq <= best_dist_sq:
                best_dist_sq = dist_sq
                best_p = p
        return best_p

    def can_start(self) -> bool:
        self.target = self._find_closest_player()
        return self.target is not None

    def can_continue(self) -> bool:
        if self.target is None:
            return False
        if not getattr(self.target, "alive", True) or getattr(self.target, "dead", False):
            return False
        if not getattr(self.target, "survival", True):
            return False
        mx, my, mz = self.mob.pos
        px, py, pz = self.target.pos
        dist_sq = (px - mx) ** 2 + (py - my) ** 2 + (pz - mz) ** 2
        return dist_sq <= (self.target_range * self.target_range)

    def start(self):
        self.repath_timer = 0.0

    def stop(self):
        self.target = None
        self.mob.desired_speed = 0.0

    def tick(self, dt: float):
        if self.attack_timer > 0:
            self.attack_timer = max(0.0, self.attack_timer - dt)

        # Re-evaluate target or track current target
        if self.target is None or not getattr(self.target, "alive", True):
            self.target = self._find_closest_player()
            if self.target is None:
                self.mob.desired_speed = 0.0
                return

        mx, my, mz = self.mob.pos
        px, py, pz = self.target.pos
        dx = px - mx
        dy = py - my
        dz = pz - mz
        dist_xz = math.hypot(dx, dz)
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Chase towards player
        self.repath_timer -= dt
        if self.repath_timer <= 0:
            self.repath_timer = 0.1
            if dist_xz > 0.05:
                desired_yaw = math.atan2(-dx, -dz)
                world = getattr(self.mob, "world", None)
                safe_yaw = pick_safe_yaw(self.mob, desired_yaw, world=world)
                if safe_yaw is not None:
                    self.mob.target_yaw = safe_yaw
                    self.mob.desired_speed = getattr(self.mob, "run_speed", 2.4)
                else:
                    self.mob.target_yaw = desired_yaw
                    self.mob.desired_speed = 0.0

        # Attack if in melee range
        if dist_3d <= self.reach or (dist_xz <= self.reach and abs(dy) <= 2.0):
            if self.attack_timer <= 0:
                self.attack_timer = self.cooldown
                self._execute_attack(self.target)

    def _execute_attack(self, target):
        target.take_damage(self.damage)
        mx, my, mz = self.mob.pos
        px, py, pz = target.pos
        dx = px - mx
        dz = pz - mz
        klen = math.hypot(dx, dz) or 1.0
        kdx = dx / klen
        kdz = dz / klen
        kb_speed = 4.5
        new_vel = [kdx * kb_speed, 3.2, kdz * kb_speed]

        if hasattr(target, "vel"):
            target.vel = list(new_vel)
        target.hurt_time = 0.4

        # Multiplayer sync if running on server
        world = getattr(self.mob, "world", None)
        server = getattr(world, "server", None)
        if server is not None:
            target_uuid = getattr(target, "uuid", None)
            target_client = server.clients.get(target_uuid)
            if target_client is not None:
                target_client.vel = list(new_vel)
                target_client.send({
                    "type": "player_knockback",
                    "vel": new_vel,
                })
                server._send_player_health(target_client, target)
                server.broadcast({
                    "type": "player_hurt",
                    "uuid": target_uuid,
                    "hurt_time": target.hurt_time,
                })
                if getattr(target, "dead", False):
                    server._respawn_client_player(target_client, target, cause="slain by Zombie")


class SpiderAttackGoal(Goal):
    """Hostile attack goal for Spider:
    - Only targets the player in low skylight (<= 10) and low infrared light (<= 10).
    - If either skylight or infrared light is > 10, the spider does not target the player.
    - Chases the player at mob.run_speed, climbing walls and ceilings to pursue.
    - Attacks when within melee reach, applying damage, knockback, and hurt_time.
    """

    def __init__(self, mob, target_range=16.0, reach=1.8, damage=2.5, cooldown=1.0):
        super().__init__(mob)
        self.target_range = float(target_range)
        self.reach = float(reach)
        self.damage = float(damage)
        self.cooldown = float(cooldown)
        self.attack_timer = 0.0
        self.target = None
        self.repath_timer = 0.0

    def _get_target_candidates(self):
        world = getattr(self.mob, "world", None)
        if world is None:
            return []
        players = []
        if hasattr(world, "players") and world.players:
            players.extend(world.players.values())
        single = getattr(world, "_singleplayer_player", None)
        if single is not None and single not in players:
            players.append(single)
        return players

    def _find_closest_player(self):
        if hasattr(self.mob, "can_target_player") and not self.mob.can_target_player():
            return None
        candidates = self._get_target_candidates()
        best_p = None
        best_dist_sq = self.target_range * self.target_range
        mx, my, mz = self.mob.pos
        for p in candidates:
            if not getattr(p, "alive", True) or getattr(p, "dead", False):
                continue
            if not getattr(p, "survival", True):
                continue
            if hasattr(self.mob, "can_target_player") and not self.mob.can_target_player(player=p):
                continue
            px, py, pz = p.pos
            dx = px - mx
            dy = py - my
            dz = pz - mz
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq <= best_dist_sq:
                best_dist_sq = dist_sq
                best_p = p
        return best_p

    def can_start(self) -> bool:
        if hasattr(self.mob, "can_target_player") and not self.mob.can_target_player():
            return False
        self.target = self._find_closest_player()
        return self.target is not None

    def can_continue(self) -> bool:
        if self.target is None:
            return False
        if hasattr(self.mob, "can_target_player") and not self.mob.can_target_player(player=self.target):
            return False
        if not getattr(self.target, "alive", True) or getattr(self.target, "dead", False):
            return False
        if not getattr(self.target, "survival", True):
            return False
        mx, my, mz = self.mob.pos
        px, py, pz = self.target.pos
        dist_sq = (px - mx) ** 2 + (py - my) ** 2 + (pz - mz) ** 2
        return dist_sq <= (self.target_range * self.target_range)

    def start(self):
        self.repath_timer = 0.0

    def stop(self):
        self.target = None
        self.mob.desired_speed = 0.0

    def tick(self, dt: float):
        if self.attack_timer > 0:
            self.attack_timer = max(0.0, self.attack_timer - dt)

        if self.target is None or not getattr(self.target, "alive", True):
            self.target = self._find_closest_player()
            if self.target is None:
                self.mob.desired_speed = 0.0
                return

        if hasattr(self.mob, "can_target_player") and not self.mob.can_target_player(player=self.target):
            self.stop()
            return

        mx, my, mz = self.mob.pos
        px, py, pz = self.target.pos
        dx = px - mx
        dy = py - my
        dz = pz - mz
        dist_xz = math.hypot(dx, dz)
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Repath towards target
        self.repath_timer -= dt
        if self.repath_timer <= 0:
            self.repath_timer = 0.1
            if dist_xz > 0.05:
                desired_yaw = math.atan2(-dx, -dz)
                self.mob.target_yaw = desired_yaw
                self.mob.desired_speed = getattr(self.mob, "run_speed", 3.0)

        # If on ceiling and target is below, can descend/drop
        if getattr(self.mob, "on_ceiling", False) and dy < -2.0 and dist_xz < 2.0:
            self.mob.on_ceiling = False
            self.mob.vel[1] = -2.0

        # Melee attack
        if dist_3d <= self.reach or (dist_xz <= self.reach and abs(dy) <= 2.0):
            if self.attack_timer <= 0:
                self.attack_timer = self.cooldown
                self._execute_attack(self.target)

    def _execute_attack(self, target):
        target.take_damage(self.damage)
        mx, my, mz = self.mob.pos
        px, py, pz = target.pos
        dx = px - mx
        dz = pz - mz
        klen = math.hypot(dx, dz) or 1.0
        kdx = dx / klen
        kdz = dz / klen
        kb_speed = 4.0
        new_vel = [kdx * kb_speed, 2.5, kdz * kb_speed]

        if hasattr(target, "vel"):
            target.vel = list(new_vel)
        target.hurt_time = 0.4

        # Multiplayer sync if running on server
        world = getattr(self.mob, "world", None)
        server = getattr(world, "server", None)
        if server is not None:
            target_uuid = getattr(target, "uuid", None)
            target_client = server.clients.get(target_uuid)
            if target_client is not None:
                target_client.vel = list(new_vel)
                target_client.send({
                    "type": "player_knockback",
                    "vel": new_vel,
                })
                server._send_player_health(target_client, target)
                server.broadcast({
                    "type": "player_hurt",
                    "uuid": target_uuid,
                    "hurt_time": target.hurt_time,
                })
                if getattr(target, "dead", False):
                    server._respawn_client_player(target_client, target, cause="slain by Spider")


