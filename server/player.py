import math

GRAVITY = 28.0
JUMP = 8.5
WALK = 4.6
SPRINT = 7.2
HEIGHT = 1.8
RADIUS = 0.3


def look_direction(yaw, pitch):
    cp = math.cos(pitch)
    return (-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp)


class Input:
    def __init__(self):
        self.forward = False
        self.back = False
        self.left = False
        self.right = False
        self.jump = False
        self.sprint = False
        self.sneak = False
        self.down = False


class Player:
    def __init__(self, world):
        self.world = world
        self.pos = [0.5, 30.0, 0.5]
        self.vel = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0
        self.on_ground = False
        self.radius = RADIUS
        self.height = HEIGHT
        self.eye = 1.62
        self.input = Input()
        self.actual_speed = None
        self.uuid = ""
        self.skin = None

    def apply_identity(self, player_id, skin_rgba):
        self.uuid = str(player_id)
        self.skin = skin_rgba

    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, val):
        self.pos[0] = float(val)

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, val):
        self.pos[1] = float(val)

    @property
    def z(self):
        return self.pos[2]

    @z.setter
    def z(self, val):
        self.pos[2] = float(val)

    def spawn(self, x, z):
        ground = self.world.column_top(x, z)
        self.pos[0] = x + 0.5
        self.pos[1] = ground + 1
        self.pos[2] = z + 0.5
        self.vel = [0.0, 0.0, 0.0]
        self.on_ground = True

    def collides_at(self, px, py, pz):
        if isinstance(getattr(self.world, "chunks", None), dict) and hasattr(self.world, "collides_box"):
            return self.world.collides_box(
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
        blocks = self.world.blocks_movement
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                for x in range(min_x, max_x + 1):
                    if blocks(x, y, z):
                        return True
        return False

    def sweep(self, axis, delta):
        if delta == 0:
            return False
        max_step = 0.35
        left = delta
        while abs(left) > 1e-8:
            step = math.copysign(min(abs(left), max_step), left)
            start = self.pos[axis]
            self.pos[axis] = start + step
            if not self.collides_at(*self.pos):
                left -= step
                continue
            lo = 0.0
            hi = step
            for _ in range(14):
                mid = (lo + hi) * 0.5
                self.pos[axis] = start + mid
                if self.collides_at(*self.pos):
                    hi = mid
                else:
                    lo = mid
            self.pos[axis] = start + lo - math.copysign(1e-4, step)
            self.vel[axis] = 0.0
            return True
        return False

    def step_up(self, dx, dz):
        start_x, start_y, start_z = self.pos
        self.pos[1] = start_y + 1
        if self.collides_at(*self.pos):
            self.pos[1] = start_y
            return False
        self.sweep(0, dx)
        self.sweep(2, dz)
        moved = math.hypot(self.pos[0] - start_x, self.pos[2] - start_z)
        if moved < max(0.02, math.hypot(dx, dz) * 0.35):
            self.pos[0], self.pos[1], self.pos[2] = start_x, start_y, start_z
            return False
        self.sweep(1, -1.05)
        return True

    def move_horizontal(self, dx, dz):
        start_x, start_y, start_z = self.pos
        hit_x = self.sweep(0, dx)
        hit_z = self.sweep(2, dz)
        if not hit_x and not hit_z:
            return
        moved = math.hypot(self.pos[0] - start_x, self.pos[2] - start_z)
        want = math.hypot(dx, dz)
        if want < 1e-6 or moved > want * 0.85:
            return
        self.pos[0], self.pos[1], self.pos[2] = start_x, start_y, start_z
        if not self.step_up(dx, dz):
            self.sweep(0, dx)
            self.sweep(2, dz)

    def in_water(self):
        if self.world is None:
            return False
        min_x = math.floor(self.pos[0] - self.radius)
        max_x = math.floor(self.pos[0] + self.radius - 1e-6)
        min_y = math.floor(self.pos[1])
        max_y = math.floor(self.pos[1] + self.height - 1e-4)
        min_z = math.floor(self.pos[2] - self.radius)
        max_z = math.floor(self.pos[2] + self.radius - 1e-6)
        from server.fluid import is_water

        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                for x in range(min_x, max_x + 1):
                    if is_water(self.world.get_block(x, y, z)):
                        return True
        return False

    def update(self, dt):
        inp = self.input
        ix = (-1 if inp.left else 0) + (1 if inp.right else 0)
        iz = (1 if inp.forward else 0) + (-1 if inp.back else 0)
        length = math.hypot(ix, iz)
        if length > 0:
            ix /= length
            iz /= length
        yaw = self.yaw
        fx, fz = -math.sin(yaw), -math.cos(yaw)
        rx, rz = math.cos(yaw), -math.sin(yaw)

        in_water = self.in_water()

        if in_water:
            speed = (SPRINT if inp.sprint else WALK) * 0.65
            self.vel[0] = (rx * ix + fx * iz) * speed
            self.vel[2] = (rz * ix + fz * iz) * speed

            flow_y = 0.0
            if self.world is not None:
                from server.fluid import get_entity_fluid_flow
                min_x = math.floor(self.pos[0] - self.radius)
                max_x = math.floor(self.pos[0] + self.radius - 1e-6)
                min_y = math.floor(self.pos[1])
                max_y = math.floor(self.pos[1] + self.height - 1e-4)
                min_z = math.floor(self.pos[2] - self.radius)
                max_z = math.floor(self.pos[2] + self.radius - 1e-6)
                flow_x, flow_y, flow_z = get_entity_fluid_flow(self.world, min_x, min_y, min_z, max_x, max_y, max_z)
                if flow_x != 0.0 or flow_z != 0.0:
                    flow_push = 1.4
                    self.vel[0] += flow_x * flow_push
                    self.vel[2] += flow_z * flow_push
                if flow_y < 0 and not inp.jump and not (inp.forward and self.pitch > 0.15):
                    self.vel[1] = max(-6.0, self.vel[1] + flow_y * 12.0 * dt)

            is_down = getattr(inp, "down", False) or getattr(inp, "sneak", False)
            if inp.jump:
                head_y = math.floor(self.pos[1] + self.eye)
                from server.fluid import is_water

                head_in_water = is_water(self.world.get_block(math.floor(self.pos[0]), head_y, math.floor(self.pos[2])))
                if not head_in_water:
                    self.vel[1] = max(self.vel[1], 5.5)
                else:
                    self.vel[1] = min(self.vel[1] + 18.0 * dt, 4.0) if self.vel[1] >= 0 else min(self.vel[1] + 28.0 * dt, 4.0)
                self.on_ground = False
            elif is_down:
                self.vel[1] = max(self.vel[1] - 18.0 * dt, -4.0)
            elif inp.forward and abs(self.pitch) > 0.15:
                target_vy = math.sin(self.pitch) * speed
                self.vel[1] += (target_vy - self.vel[1]) * min(1.0, 5.0 * dt)
                self.vel[1] = max(-4.0, min(4.0, self.vel[1]))
            else:
                if self.vel[1] > 0:
                    self.vel[1] = max(0.0, self.vel[1] - 10.0 * dt)
                else:
                    min_down = -6.0 if flow_y < 0 else -2.0
                    self.vel[1] = max(min_down, self.vel[1] - (8.0 if flow_y < 0 else 4.0) * dt)

            dx = self.vel[0] * dt
            dy = self.vel[1] * dt
            dz = self.vel[2] * dt

            start_x, start_z = self.pos[0], self.pos[2]
            if self.on_ground and dy <= 0:
                if not self.step_up(dx, dz):
                    self.sweep(0, dx)
                    self.sweep(2, dz)
            else:
                self.sweep(0, dx)
                self.sweep(2, dz)

            hit_vert = self.sweep(1, dy)
            actual_moved = math.hypot(self.pos[0] - start_x, self.pos[2] - start_z)
            self.actual_speed = actual_moved / max(dt, 0.001)

            self.pos[1] -= 0.06
            grounded = self.collides_at(*self.pos)
            self.pos[1] += 0.06
            self.on_ground = grounded or (hit_vert and dy < 0)
            if self.on_ground and self.vel[1] < 0:
                self.vel[1] = 0.0
        else:
            speed = SPRINT if inp.sprint else WALK
            self.vel[0] = (rx * ix + fx * iz) * speed
            self.vel[2] = (rz * ix + fz * iz) * speed

            jumping = inp.jump and self.on_ground
            if jumping:
                self.vel[1] = JUMP
                self.on_ground = False
            elif self.on_ground and self.vel[1] < 0:
                self.vel[1] = 0.0
            self.vel[1] -= GRAVITY * dt
            if self.vel[1] < -40:
                self.vel[1] = -40

            dx = self.vel[0] * dt
            dy = self.vel[1] * dt
            dz = self.vel[2] * dt
            start_x, start_z = self.pos[0], self.pos[2]
            if self.on_ground and not jumping:
                self.move_horizontal(dx, dz)
            else:
                self.sweep(0, dx)
                self.sweep(2, dz)
            hit_ground = self.sweep(1, dy) and dy < 0
            actual_moved = math.hypot(self.pos[0] - start_x, self.pos[2] - start_z)
            self.actual_speed = actual_moved / max(dt, 0.001)

            self.pos[1] -= 0.06
            grounded = self.collides_at(*self.pos)
            self.pos[1] += 0.06
            self.on_ground = grounded or hit_ground
            if self.on_ground and self.vel[1] < 0:
                self.vel[1] = 0.0

        if self.on_ground and self.world is not None:
            by = math.floor(self.pos[1] - 0.05)
            min_x = math.floor(self.pos[0] - self.radius)
            max_x = math.floor(self.pos[0] + self.radius - 1e-6)
            min_z = math.floor(self.pos[2] - self.radius)
            max_z = math.floor(self.pos[2] + self.radius - 1e-6)
            for x in range(min_x, max_x + 1):
                for z in range(min_z, max_z + 1):
                    if hasattr(self.world, "notify_neighbors_falling"):
                        self.world.notify_neighbors_falling(x, by, z)

    def overlaps_block(self, x, y, z):
        return (
            self.pos[0] - self.radius < x + 1
            and self.pos[0] + self.radius > x
            and self.pos[1] < y + 1
            and self.pos[1] + self.height > y
            and self.pos[2] - self.radius < z + 1
            and self.pos[2] + self.radius > z
        )
