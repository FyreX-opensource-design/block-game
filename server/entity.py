import math
import random

GRAVITY = 20.0
TERMINAL_VELOCITY = -25.0
DESPAWN_TIME = 300.0  # 5 minutes


def _has_collision_boxes(world):
    return isinstance(getattr(world, "chunks", None), dict) and hasattr(world, "get_collision_boxes")


def _has_box_collision(world):
    return isinstance(getattr(world, "chunks", None), dict) and hasattr(world, "collides_box")


class ItemEntity:
    _next_id = 1

    def __init__(
        self,
        item,
        count=1,
        pos=None,
        vel=None,
        despawn_timer=DESPAWN_TIME,
        pickup_delay=0.0,
        entity_id=None,
    ):
        if entity_id is None:
            self.id = ItemEntity._next_id
            ItemEntity._next_id += 1
        else:
            self.id = int(entity_id)
            if self.id >= ItemEntity._next_id:
                ItemEntity._next_id = self.id + 1

        self.item = str(item)
        self.count = max(1, int(count))
        self.pos = list(pos) if pos is not None else [0.0, 0.0, 0.0]
        self.vel = list(vel) if vel is not None else [0.0, 0.0, 0.0]
        self.despawn_timer = float(despawn_timer)
        self.pickup_delay = float(pickup_delay)
        self.on_ground = False
        self.age = 0.0
        self.radius = 0.15
        self.height = 0.25

    def in_water(self, world):
        if world is None:
            return False
        from server.fluid import is_water

        min_x = math.floor(self.pos[0] - self.radius)
        max_x = math.floor(self.pos[0] + self.radius)
        min_y = math.floor(self.pos[1])
        max_y = math.floor(self.pos[1] + self.height)
        min_z = math.floor(self.pos[2] - self.radius)
        max_z = math.floor(self.pos[2] + self.radius)
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                for x in range(min_x, max_x + 1):
                    if is_water(world.get_block(x, y, z)):
                        return True
        return False

    def update(self, dt, world=None):
        self.age += dt
        self.despawn_timer -= dt
        if self.pickup_delay > 0:
            self.pickup_delay = max(0.0, self.pickup_delay - dt)

        water = self.in_water(world) if world is not None else False
        flow_x, flow_y, flow_z = 0.0, 0.0, 0.0
        if water and world is not None:
            from server.fluid import get_entity_fluid_flow

            min_x = math.floor(self.pos[0] - self.radius)
            max_x = math.floor(self.pos[0] + self.radius)
            min_y = math.floor(self.pos[1])
            max_y = math.floor(self.pos[1] + self.height)
            min_z = math.floor(self.pos[2] - self.radius)
            max_z = math.floor(self.pos[2] + self.radius)
            flow_x, flow_y, flow_z = get_entity_fluid_flow(world, min_x, min_y, min_z, max_x, max_y, max_z)

        if water:
            if flow_y < 0:
                self.vel[1] = max(-6.0, self.vel[1] + flow_y * 10.0 * dt)
            else:
                self.vel[1] = min(1.5, self.vel[1] + 8.0 * dt)

            water_drag = math.exp(-3.0 * dt)
            self.vel[0] *= water_drag
            self.vel[2] *= water_drag
            if flow_x != 0.0 or flow_z != 0.0:
                self.vel[0] += flow_x * 4.5 * dt
                self.vel[2] += flow_z * 4.5 * dt
            if abs(self.vel[0]) < 0.005 and flow_x == 0.0:
                self.vel[0] = 0.0
            if abs(self.vel[2]) < 0.005 and flow_z == 0.0:
                self.vel[2] = 0.0
        else:
            # Apply gravity
            if not self.on_ground:
                self.vel[1] = max(TERMINAL_VELOCITY, self.vel[1] - GRAVITY * dt)

            # Friction
            if self.on_ground:
                friction = math.exp(-12.0 * dt)
                self.vel[0] *= friction
                self.vel[2] *= friction
                if abs(self.vel[0]) < 0.01:
                    self.vel[0] = 0.0
                if abs(self.vel[2]) < 0.01:
                    self.vel[2] = 0.0
            else:
                air_drag = math.exp(-1.5 * dt)
                self.vel[0] *= air_drag
                self.vel[2] *= air_drag

        # Move with collision against world
        if world is not None:
            self._move(dt, world)
            if self.on_ground and hasattr(world, "notify_neighbors_falling"):
                by = math.floor(self.pos[1] - 0.05)
                bx = math.floor(self.pos[0])
                bz = math.floor(self.pos[2])
                world.notify_neighbors_falling(bx, by, bz)
        else:
            self.pos[0] += self.vel[0] * dt
            self.pos[1] += self.vel[1] * dt
            self.pos[2] += self.vel[2] * dt

    def _move(self, dt, world):
        # Y movement
        dy = self.vel[1] * dt
        if dy != 0:
            new_y = self.pos[1] + dy
            if dy < 0:
                # Falling down
                check_y = math.floor(new_y)
                min_x = math.floor(self.pos[0] - self.radius)
                max_x = math.floor(self.pos[0] + self.radius)
                min_z = math.floor(self.pos[2] - self.radius)
                max_z = math.floor(self.pos[2] + self.radius)
                hit = False
                highest_y = None
                for x in range(min_x, max_x + 1):
                    for z in range(min_z, max_z + 1):
                        boxes = (
                            world.get_collision_boxes(x, check_y, z)
                            if _has_collision_boxes(world)
                            else (
                                [(float(x), float(check_y), float(z), float(x + 1), float(check_y + 1), float(z + 1))]
                                if world.is_solid(x, check_y, z)
                                else []
                            )
                        )
                        for bx0, by0, bz0, bx1, by1, bz1 in boxes:
                            if new_y < by1:
                                hit = True
                                if highest_y is None or by1 > highest_y:
                                    highest_y = by1
                if hit and highest_y is not None:
                    self.pos[1] = highest_y
                    self.vel[1] = 0.0
                    self.on_ground = True
                else:
                    self.pos[1] = new_y
                    self.on_ground = False
            else:
                # Rising up
                check_y = math.floor(new_y + self.height)
                min_x = math.floor(self.pos[0] - self.radius)
                max_x = math.floor(self.pos[0] + self.radius)
                min_z = math.floor(self.pos[2] - self.radius)
                max_z = math.floor(self.pos[2] + self.radius)
                hit = False
                lowest_y = None
                for x in range(min_x, max_x + 1):
                    for z in range(min_z, max_z + 1):
                        boxes = (
                            world.get_collision_boxes(x, check_y, z)
                            if _has_collision_boxes(world)
                            else (
                                [(float(x), float(check_y), float(z), float(x + 1), float(check_y + 1), float(z + 1))]
                                if world.is_solid(x, check_y, z)
                                else []
                            )
                        )
                        for bx0, by0, bz0, bx1, by1, bz1 in boxes:
                            if (new_y + self.height) > by0:
                                hit = True
                                if lowest_y is None or by0 < lowest_y:
                                    lowest_y = by0
                if hit and lowest_y is not None:
                    self.pos[1] = lowest_y - self.height - 1e-4
                    self.vel[1] = 0.0
                else:
                    self.pos[1] = new_y
                    self.on_ground = False
        else:
            # Check if ground underneath is still solid
            if _has_box_collision(world):
                self.on_ground = world.collides_box(
                    self.pos[0] - self.radius,
                    self.pos[1] - 0.05,
                    self.pos[2] - self.radius,
                    self.pos[0] + self.radius,
                    self.pos[1],
                    self.pos[2] + self.radius,
                )
            else:
                check_y = math.floor(self.pos[1] - 0.05)
                min_x = math.floor(self.pos[0] - self.radius)
                max_x = math.floor(self.pos[0] + self.radius)
                min_z = math.floor(self.pos[2] - self.radius)
                max_z = math.floor(self.pos[2] + self.radius)
                has_ground = False
                for x in range(min_x, max_x + 1):
                    for z in range(min_z, max_z + 1):
                        if world.is_solid(x, check_y, z):
                            has_ground = True
                            break
                    if has_ground:
                        break
                self.on_ground = has_ground

        # X movement
        dx = self.vel[0] * dt
        if dx != 0:
            new_x = self.pos[0] + dx
            if _has_box_collision(world):
                hit = world.collides_box(
                    new_x - self.radius,
                    self.pos[1],
                    self.pos[2] - self.radius,
                    new_x + self.radius,
                    self.pos[1] + self.height,
                    self.pos[2] + self.radius,
                )
            else:
                check_x = math.floor(new_x + math.copysign(self.radius, dx))
                min_y = math.floor(self.pos[1])
                max_y = math.floor(self.pos[1] + self.height)
                min_z = math.floor(self.pos[2] - self.radius)
                max_z = math.floor(self.pos[2] + self.radius)
                hit = False
                for y in range(min_y, max_y + 1):
                    for z in range(min_z, max_z + 1):
                        if world.is_solid(check_x, y, z):
                            hit = True
                            break
                    if hit:
                        break
            if hit:
                self.vel[0] = 0.0
            else:
                self.pos[0] = new_x

        # Z movement
        dz = self.vel[2] * dt
        if dz != 0:
            new_z = self.pos[2] + dz
            if _has_box_collision(world):
                hit = world.collides_box(
                    self.pos[0] - self.radius,
                    self.pos[1],
                    new_z - self.radius,
                    self.pos[0] + self.radius,
                    self.pos[1] + self.height,
                    new_z + self.radius,
                )
            else:
                check_z = math.floor(new_z + math.copysign(self.radius, dz))
                min_y = math.floor(self.pos[1])
                max_y = math.floor(self.pos[1] + self.height)
                min_x = math.floor(self.pos[0] - self.radius)
                max_x = math.floor(self.pos[0] + self.radius)
                hit = False
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        if world.is_solid(x, y, check_z):
                            hit = True
                            break
                    if hit:
                        break
            if hit:
                self.vel[2] = 0.0
            else:
                self.pos[2] = new_z

    def try_merge(self, other):
        from server.items import stack_limit

        if self.item != other.item or self.id == other.id or other.count <= 0:
            return False
        limit = stack_limit(self.item)
        if self.count >= limit:
            return False
        dist_sq = (
            (self.pos[0] - other.pos[0]) ** 2
            + (self.pos[1] - other.pos[1]) ** 2
            + (self.pos[2] - other.pos[2]) ** 2
        )
        if dist_sq < 0.65 * 0.65:
            transfer = min(limit - self.count, other.count)
            self.count += transfer
            other.count -= transfer
            self.despawn_timer = max(self.despawn_timer, other.despawn_timer)
            return True
        return False

    def to_dict(self):
        return {
            "id": self.id,
            "item": self.item,
            "count": self.count,
            "pos": [round(v, 4) for v in self.pos],
            "vel": [round(v, 4) for v in self.vel],
            "despawn_timer": round(self.despawn_timer, 2),
            "pickup_delay": round(self.pickup_delay, 2),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            item=data["item"],
            count=data.get("count", 1),
            pos=data.get("pos"),
            vel=data.get("vel"),
            despawn_timer=data.get("despawn_timer", DESPAWN_TIME),
            pickup_delay=data.get("pickup_delay", 0.0),
            entity_id=data.get("id"),
        )


class FallingBlockEntity:
    def __init__(
        self,
        block_id,
        pos=None,
        vel=None,
        time_falling=0.0,
        drop_on_fail=True,
        entity_id=None,
    ):
        if entity_id is None:
            self.id = ItemEntity._next_id
            ItemEntity._next_id += 1
        else:
            self.id = int(entity_id)
            if self.id >= ItemEntity._next_id:
                ItemEntity._next_id = self.id + 1

        self.block_id = str(block_id)
        self.item = ""
        # Position is center of block horizontally, bottom of block vertically
        self.pos = list(pos) if pos is not None else [0.5, 0.0, 0.5]
        self.vel = list(vel) if vel is not None else [0.0, 0.0, 0.0]
        self.time_falling = float(time_falling)
        self.drop_on_fail = bool(drop_on_fail)
        self.alive = True
        self.count = 1
        self.pickup_delay = 999999.0
        self.despawn_timer = 60.0

    @property
    def is_falling_block(self):
        return True

    def update(self, dt, world=None):
        if not self.alive:
            return

        self.time_falling += dt
        self.despawn_timer -= dt
        if self.despawn_timer <= 0:
            self.alive = False
            return

        # Acceleration downwards
        self.vel[1] = max(TERMINAL_VELOCITY, self.vel[1] - GRAVITY * dt)

        # Move vertically
        dy = self.vel[1] * dt
        new_y = self.pos[1] + dy

        if world is None:
            self.pos[1] = new_y
            return

        bx = math.floor(self.pos[0])
        bz = math.floor(self.pos[2])

        # Check landing:
        target_y = math.floor(new_y)
        if target_y <= 0:
            place_y = max(1, target_y)
            self._land(world, bx, place_y, bz)
            return

        if world.is_solid(bx, target_y, bz):
            place_y = target_y + 1
            self._land(world, bx, place_y, bz)
        else:
            self.pos[1] = new_y

    def _land(self, world, bx, by, bz):
        from server.blocks import BLOCKS

        self.alive = False
        if by <= 0 or by >= 16 * 16:
            return

        cur_id = world.get_block(bx, by, bz)
        cur_block = BLOCKS.get(cur_id)

        # If landing spot is air, fluid, or non-solid, place block here
        if cur_id in ("", "air", None) or (cur_block is not None and not cur_block.solid):
            world.set_block(bx, by, bz, self.block_id)
        else:
            above_id = world.get_block(bx, by + 1, bz)
            above_block = BLOCKS.get(above_id)
            if above_id in ("", "air", None) or (above_block is not None and not above_block.solid):
                world.set_block(bx, by + 1, bz, self.block_id)
            elif self.drop_on_fail:
                world.spawn_item(self.block_id, 1, pos=[bx + 0.5, by + 0.5, bz + 0.5])

    def to_dict(self):
        return {
            "type": "falling_block",
            "id": self.id,
            "block": self.block_id,
            "pos": [round(v, 4) for v in self.pos],
            "vel": [round(v, 4) for v in self.vel],
            "time": round(self.time_falling, 2),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            block_id=data.get("block", data.get("item", "sand")),
            pos=data.get("pos"),
            vel=data.get("vel"),
            time_falling=data.get("time", 0.0),
            entity_id=data.get("id"),
        )


class Mob:
    """Base class for living mobs."""

    def __init__(
        self,
        mob_type="pig",
        pos=None,
        vel=None,
        yaw=0.0,
        pitch=0.0,
        health=10.0,
        max_health=10.0,
        entity_id=None,
    ):
        if entity_id is None:
            self.id = ItemEntity._next_id
            ItemEntity._next_id += 1
        else:
            self.id = int(entity_id)
            if self.id >= ItemEntity._next_id:
                ItemEntity._next_id = self.id + 1

        self.mob_type = str(mob_type)
        self.type = self.mob_type
        self.pos = list(pos) if pos is not None else [0.5, 0.0, 0.5]
        self.vel = list(vel) if vel is not None else [0.0, 0.0, 0.0]
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.target_yaw = self.yaw
        self.health = float(health)
        self.max_health = float(max_health)
        self.hurt_time = 0.0
        self.panic_timer = 0.0
        self.panic_source = None
        self.desired_speed = 0.0
        self.walk_speed = 1.2
        self.run_speed = 2.8
        self.walk_time = 0.0
        self.idle_time = 0.0
        self.radius = 0.45
        self.height = 0.9
        self.alive = True
        self.on_ground = False
        self.is_remote = False
        self.target_pos = None
        self.target_yaw = self.yaw
        self.target_pitch = self.pitch
        self.count = 1
        self.despawn_timer = 999999.0
        self.loot_table = f"entities/{mob_type}"
        self.goals = None

    @property
    def is_mob(self):
        return True

    def in_panic(self):
        return self.panic_timer > 0

    def take_damage(self, amount, attacker=None, knockback_dir=None, world=None):
        if self.hurt_time > 0 or not self.alive:
            return False

        self.health -= float(amount)
        self.hurt_time = 0.4
        self.panic_timer = 5.0

        if attacker is not None and hasattr(attacker, "pos"):
            self.panic_source = [attacker.pos[0], attacker.pos[1], attacker.pos[2]]
        elif knockback_dir is not None:
            self.panic_source = [self.pos[0] - knockback_dir[0], self.pos[1], self.pos[2] - knockback_dir[1]]

        # Apply knockback
        if knockback_dir is not None:
            kdx, kdz = knockback_dir[0], knockback_dir[1]
        elif attacker is not None and hasattr(attacker, "pos"):
            kdx = self.pos[0] - attacker.pos[0]
            kdz = self.pos[2] - attacker.pos[2]
        else:
            kdx = random.uniform(-1, 1)
            kdz = random.uniform(-1, 1)

        klen = math.hypot(kdx, kdz) or 1.0
        self.vel[0] = (kdx / klen) * 5.5
        self.vel[2] = (kdz / klen) * 5.5
        self.vel[1] = 4.0
        self.on_ground = False

        if self.health <= 0:
            self.health = 0.0
            self.alive = False
            self.on_death(world=world, killer=attacker)

        return True

    def on_death(self, world=None, killer=None):
        if world is not None and self.loot_table:
            from server.loot import get_mob_drops

            drops = get_mob_drops(self, killer=killer)
            for drop_id, count in drops:
                if count > 0:
                    world.spawn_item(
                        drop_id,
                        count=count,
                        pos=[self.pos[0], self.pos[1] + 0.3, self.pos[2]],
                    )

    def collides_at(self, px, py, pz, world):
        if world is None:
            return False
        if _has_box_collision(world):
            return world.collides_box(
                px - self.radius,
                py,
                pz - self.radius,
                px + self.radius,
                py + self.height,
                pz + self.radius,
            )
        min_x = math.floor(px - self.radius)
        max_x = math.floor(px + self.radius - 1e-6)
        min_y = math.floor(py)
        max_y = math.floor(py + self.height - 1e-4)
        min_z = math.floor(pz - self.radius)
        max_z = math.floor(pz + self.radius - 1e-6)
        blocks = getattr(world, "blocks_movement", world.is_solid)
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                for x in range(min_x, max_x + 1):
                    if blocks(x, y, z):
                        return True
        return False

    def sweep(self, axis, delta, world):
        if delta == 0 or world is None:
            return False
        max_step = 0.35
        left = delta
        while abs(left) > 1e-8:
            step = math.copysign(min(abs(left), max_step), left)
            start = self.pos[axis]
            self.pos[axis] = start + step
            if not self.collides_at(*self.pos, world):
                left -= step
                continue
            lo = 0.0
            hi = step
            for _ in range(12):
                mid = (lo + hi) * 0.5
                self.pos[axis] = start + mid
                if self.collides_at(*self.pos, world):
                    hi = mid
                else:
                    lo = mid
            self.pos[axis] = start + lo - math.copysign(1e-4, step)
            self.vel[axis] = 0.0
            return True
        return False

    def step_up(self, dx, dz, world):
        if world is None:
            return False
        start_x, start_y, start_z = self.pos
        self.pos[1] = start_y + 1.05
        if self.collides_at(*self.pos, world):
            self.pos[1] = start_y
            return False
        self.sweep(0, dx, world)
        self.sweep(2, dz, world)
        moved = math.hypot(self.pos[0] - start_x, self.pos[2] - start_z)
        if moved < max(0.02, math.hypot(dx, dz) * 0.35):
            self.pos[0], self.pos[1], self.pos[2] = start_x, start_y, start_z
            return False
        self.sweep(1, -1.1, world)
        return True

    def move_horizontal(self, dx, dz, world, avoid_hazards=False):
        if world is None:
            self.pos[0] += dx
            self.pos[2] += dz
            return
        start_x, start_y, start_z = self.pos

        if avoid_hazards and self.on_ground and self.hurt_time <= 0 and not self.in_water(world):
            from server.ai import is_position_hazardous

            # Look slightly ahead along the movement vector so the mob stops before stepping into water or off a >4 block cliff
            dist = math.hypot(dx, dz)
            if dist > 1e-6:
                ux, uz = dx / dist, dz / dist
                check_x = start_x + dx + ux * self.radius
                check_z = start_z + dz + uz * self.radius
                if is_position_hazardous(world, check_x, start_y, check_z, max_fall=4, already_in_water=False) or is_position_hazardous(
                    world, start_x + dx, start_y, start_z + dz, max_fall=4, already_in_water=False
                ):
                    self.vel[0] = 0.0
                    self.vel[2] = 0.0
                    self.desired_speed = 0.0
                    return

        hit_x = self.sweep(0, dx, world)
        hit_z = self.sweep(2, dz, world)
        if not hit_x and not hit_z:
            return
        moved = math.hypot(self.pos[0] - start_x, self.pos[2] - start_z)
        want = math.hypot(dx, dz)
        if want < 1e-6 or moved > want * 0.85:
            return
        self.pos[0], self.pos[1], self.pos[2] = start_x, start_y, start_z
        if not self.step_up(dx, dz, world):
            self.sweep(0, dx, world)
            self.sweep(2, dz, world)

    def in_water(self, world):
        if world is None:
            return False
        from server.fluid import is_water

        min_x = math.floor(self.pos[0] - self.radius)
        max_x = math.floor(self.pos[0] + self.radius - 1e-6)
        min_y = math.floor(self.pos[1])
        max_y = math.floor(self.pos[1] + self.height - 1e-4)
        min_z = math.floor(self.pos[2] - self.radius)
        max_z = math.floor(self.pos[2] + self.radius - 1e-6)
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                for x in range(min_x, max_x + 1):
                    if is_water(world.get_block(x, y, z)):
                        return True
        return False

    def update(self, dt, world=None):
        if not self.alive:
            return
        if world is not None:
            self.world = world

        dt = min(dt, 0.05)
        if self.hurt_time > 0:
            self.hurt_time = max(0.0, self.hurt_time - dt)
        if self.panic_timer > 0:
            self.panic_timer = max(0.0, self.panic_timer - dt)
        if getattr(self, "regrow_cooldown", 0.0) > 0:
            self.regrow_cooldown = max(0.0, self.regrow_cooldown - dt)

        if getattr(self, "is_remote", False):
            target_pos = getattr(self, "target_pos", None)
            if target_pos is not None:
                dx = target_pos[0] - self.pos[0]
                dy = target_pos[1] - self.pos[1]
                dz = target_pos[2] - self.pos[2]
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq > 9.0:
                    self.pos[0] = target_pos[0]
                    self.pos[1] = target_pos[1]
                    self.pos[2] = target_pos[2]
                elif dist_sq > 1e-6:
                    blend = min(1.0, 15.0 * dt)
                    self.pos[0] += dx * blend
                    self.pos[1] += dy * blend
                    self.pos[2] += dz * blend
                else:
                    self.pos[0] = target_pos[0]
                    self.pos[1] = target_pos[1]
                    self.pos[2] = target_pos[2]

            target_yaw = getattr(self, "target_yaw", None)
            if target_yaw is not None:
                yaw_diff = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += yaw_diff * min(1.0, 15.0 * dt)

            target_pitch = getattr(self, "target_pitch", None)
            if target_pitch is not None:
                self.pitch += (target_pitch - self.pitch) * min(1.0, 15.0 * dt)

            hspeed = math.hypot(self.vel[0], self.vel[2])
            if hspeed > 0.05:
                self.walk_time += dt * hspeed * 2.5
            return

        # Tick AI goals
        if self.goals is not None:
            self.goals.tick(dt)

        # Smooth yaw turning towards target_yaw
        yaw_diff = (self.target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        turn_speed = 6.0 if self.in_panic() else 3.5
        self.yaw += max(-turn_speed * dt, min(turn_speed * dt, yaw_diff))

        water = self.in_water(world) if world else False
        flow_x, flow_y, flow_z = 0.0, 0.0, 0.0
        if water and world is not None:
            from server.fluid import get_entity_fluid_flow

            min_x = math.floor(self.pos[0] - self.radius)
            max_x = math.floor(self.pos[0] + self.radius - 1e-6)
            min_y = math.floor(self.pos[1])
            max_y = math.floor(self.pos[1] + self.height - 1e-4)
            min_z = math.floor(self.pos[2] - self.radius)
            max_z = math.floor(self.pos[2] + self.radius - 1e-6)
            flow_x, flow_y, flow_z = get_entity_fluid_flow(world, min_x, min_y, min_z, max_x, max_y, max_z)

        # Desired horizontal movement
        if self.desired_speed > 0:
            fx = -math.sin(self.yaw)
            fz = -math.cos(self.yaw)
            want_vx = fx * self.desired_speed
            want_vz = fz * self.desired_speed
            blend = min(1.0, 10.0 * dt) if (self.on_ground and not water) else min(1.0, 3.0 * dt)
            self.vel[0] += (want_vx - self.vel[0]) * blend
            self.vel[2] += (want_vz - self.vel[2]) * blend
        elif not water:
            friction = math.exp(-10.0 * dt) if self.on_ground else math.exp(-1.5 * dt)
            self.vel[0] *= friction
            self.vel[2] *= friction
            if abs(self.vel[0]) < 0.01:
                self.vel[0] = 0.0
            if abs(self.vel[2]) < 0.01:
                self.vel[2] = 0.0

        # Gravity / Buoyancy
        if water:
            if flow_y < 0:
                self.vel[1] = max(-6.0, self.vel[1] + flow_y * 12.0 * dt)
            else:
                self.vel[1] = min(self.vel[1] + 10.0 * dt, 2.0)
            self.vel[0] *= math.exp(-4.0 * dt)
            self.vel[2] *= math.exp(-4.0 * dt)
            if flow_x != 0.0 or flow_z != 0.0:
                self.vel[0] += flow_x * 4.5 * dt
                self.vel[2] += flow_z * 4.5 * dt
            if abs(self.vel[0]) < 0.01 and flow_x == 0.0:
                self.vel[0] = 0.0
            if abs(self.vel[2]) < 0.01 and flow_z == 0.0:
                self.vel[2] = 0.0
        else:
            self.vel[1] = max(TERMINAL_VELOCITY, self.vel[1] - GRAVITY * dt)

        # Collision & movement
        if world is not None:
            # Horizontal move
            dx = self.vel[0] * dt
            dz = self.vel[2] * dt
            if dx != 0 or dz != 0:
                self.move_horizontal(dx, dz, world, avoid_hazards=True)

            # Vertical move
            dy = self.vel[1] * dt
            if dy != 0:
                hit_y = self.sweep(1, dy, world)
                if hit_y:
                    if dy < 0:
                        self.on_ground = True
                        self.vel[1] = 0.0
                    else:
                        self.vel[1] = 0.0
                else:
                    self.on_ground = False
            else:
                self.on_ground = self.collides_at(self.pos[0], self.pos[1] - 0.05, self.pos[2], world)
        else:
            self.pos[0] += self.vel[0] * dt
            self.pos[1] += self.vel[1] * dt
            self.pos[2] += self.vel[2] * dt

        # Walk animation tracking
        hspeed = math.hypot(self.vel[0], self.vel[2])
        if self.on_ground and hspeed > 0.05:
            self.walk_time += dt * hspeed * 2.5

        # Void despawn
        if self.pos[1] < -32.0:
            self.alive = False

    def to_dict(self):
        return {
            "type": self.mob_type,
            "id": self.id,
            "pos": [round(v, 4) for v in self.pos],
            "vel": [round(v, 4) for v in self.vel],
            "yaw": round(self.yaw, 4),
            "pitch": round(self.pitch, 4),
            "health": round(self.health, 2),
            "hurt_time": round(self.hurt_time, 2),
        }

    @classmethod
    def from_dict(cls, data):
        mob_type = data.get("type", data.get("mob_type", "pig"))
        if mob_type == "pig":
            return Pig.from_dict(data)
        if mob_type == "sheep":
            return Sheep.from_dict(data)
        mob = cls(
            mob_type=mob_type,
            pos=data.get("pos"),
            vel=data.get("vel"),
            yaw=data.get("yaw", 0.0),
            pitch=data.get("pitch", 0.0),
            health=data.get("health", 10.0),
            entity_id=data.get("id"),
        )
        mob.hurt_time = float(data.get("hurt_time", 0.0))
        return mob


class Pig(Mob):
    """Pig mob entity with wander and panic goals."""

    def __init__(
        self,
        pos=None,
        vel=None,
        yaw=0.0,
        pitch=0.0,
        health=10.0,
        max_health=10.0,
        entity_id=None,
    ):
        super().__init__(
            mob_type="pig",
            pos=pos,
            vel=vel,
            yaw=yaw,
            pitch=pitch,
            health=health,
            max_health=max_health,
            entity_id=entity_id,
        )
        self.walk_speed = 1.2
        self.run_speed = 2.8
        self.radius = 0.45
        self.height = 0.9

        # Initialize Goal system
        from server.ai import GoalSelector, PanicGoal, WanderGoal

        self.goals = GoalSelector()
        self.goals.add_goal(1, PanicGoal(self, speed_mult=2.3))
        self.goals.add_goal(2, WanderGoal(self))

    @classmethod
    def from_dict(cls, data):
        pig = cls(
            pos=data.get("pos"),
            vel=data.get("vel"),
            yaw=data.get("yaw", 0.0),
            pitch=data.get("pitch", 0.0),
            health=data.get("health", 10.0),
            entity_id=data.get("id"),
        )
        pig.hurt_time = float(data.get("hurt_time", 0.0))
        return pig


SHEEP_WOOL_BLOCKS = {
    "white": "wool",
    "red": "red_wool",
    "yellow": "yellow_wool",
    "orange": "orange_wool",
    "green": "green_wool",
    "blue": "blue_wool",
    "black": "black_wool",
}

DYE_TO_SHEEP_COLOR = {
    "white_dye": "white",
    "red_dye": "red",
    "yellow_dye": "yellow",
    "orange_dye": "orange",
    "green_dye": "green",
    "blue_dye": "blue",
    "black_dye": "black",
}


class Sheep(Mob):
    """Sheep mob entity with panic, eat grass (regrow wool), and wander goals."""

    def __init__(
        self,
        pos=None,
        vel=None,
        yaw=0.0,
        pitch=0.0,
        health=8.0,
        max_health=8.0,
        entity_id=None,
        sheared=False,
        color="white",
        regrow_cooldown=0.0,
    ):
        super().__init__(
            mob_type="sheep",
            pos=pos,
            vel=vel,
            yaw=yaw,
            pitch=pitch,
            health=health,
            max_health=max_health,
            entity_id=entity_id,
        )
        self.walk_speed = 1.2
        self.run_speed = 2.8
        self.radius = 0.45
        self.height = 1.1
        self.sheared = bool(sheared)
        self.color = str(color) if color in SHEEP_WOOL_BLOCKS else "white"
        self.regrow_cooldown = float(regrow_cooldown)

        from server.ai import EatGrassGoal, GoalSelector, PanicGoal, WanderGoal

        self.goals = GoalSelector()
        self.goals.add_goal(1, PanicGoal(self, speed_mult=2.3))
        self.goals.add_goal(2, EatGrassGoal(self))
        self.goals.add_goal(3, WanderGoal(self))

    @property
    def wool_block(self):
        return SHEEP_WOOL_BLOCKS.get(self.color, "wool")

    def shear(self, world=None, rng=None):
        """Shear the sheep for 1-3 wool blocks if not already sheared.

        Returns list of (wool_block_id, count) drops, or empty list if unable to shear.
        """
        if not self.alive or self.sheared:
            return []
        rng = rng or random
        self.sheared = True
        self.regrow_cooldown = rng.uniform(5.0, 15.0)
        count = rng.randint(1, 3)
        wool_id = self.wool_block
        if world is not None:
            for _ in range(count):
                vx = rng.uniform(-1.2, 1.2)
                vy = rng.uniform(2.0, 3.5)
                vz = rng.uniform(-1.2, 1.2)
                world.spawn_item(
                    wool_id,
                    count=1,
                    pos=[self.pos[0], self.pos[1] + 0.5, self.pos[2]],
                    vel=[vx, vy, vz],
                    pickup_delay=0.3,
                )
        return [(wool_id, count)]

    def dye(self, dye_item_id):
        """Dye the sheep's wool with a dye item."""
        if not self.alive:
            return False
        new_color = DYE_TO_SHEEP_COLOR.get(str(dye_item_id))
        if not new_color or new_color == self.color:
            return False
        self.color = new_color
        return True

    def on_death(self, world=None, killer=None):
        if world is not None:
            if not self.sheared:
                world.spawn_item(
                    self.wool_block,
                    count=1,
                    pos=[self.pos[0], self.pos[1] + 0.3, self.pos[2]],
                )

    def to_dict(self):
        d = super().to_dict()
        d["sheared"] = self.sheared
        d["color"] = self.color
        d["regrow_cooldown"] = round(self.regrow_cooldown, 2)
        return d

    @classmethod
    def from_dict(cls, data):
        sheep = cls(
            pos=data.get("pos"),
            vel=data.get("vel"),
            yaw=data.get("yaw", 0.0),
            pitch=data.get("pitch", 0.0),
            health=data.get("health", 8.0),
            entity_id=data.get("id"),
            sheared=data.get("sheared", False),
            color=data.get("color", "white"),
            regrow_cooldown=data.get("regrow_cooldown", 0.0),
        )
        sheep.hurt_time = float(data.get("hurt_time", 0.0))
        return sheep


def entity_from_dict(data):
    etype = data.get("type")
    if etype == "falling_block" or "block" in data:
        return FallingBlockEntity.from_dict(data)
    elif etype == "pig" or data.get("mob_type") == "pig":
        return Pig.from_dict(data)
    elif etype == "sheep" or data.get("mob_type") == "sheep":
        return Sheep.from_dict(data)
    elif etype == "mob":
        return Mob.from_dict(data)
    return ItemEntity.from_dict(data)



