import math
import random


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
            self.target_yaw = base_angle + random.uniform(-0.75, 0.75)
        else:
            self.target_yaw = random.uniform(-math.pi, math.pi)

        self.repath_timer = random.uniform(0.8, 1.4)

    def tick(self, dt: float):
        self.repath_timer -= dt
        if self.repath_timer <= 0:
            self._pick_flee_direction()

        self.mob.target_yaw = self.target_yaw
        self.mob.desired_speed = getattr(self.mob, "run_speed", 2.8)


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
                self.state = "walk"
                self.timer = random.uniform(2.0, 4.5)
                self.mob.desired_speed = getattr(self.mob, "walk_speed", 1.2)
                # Pick a random direction, slightly biased forward/turning
                turn = random.uniform(-1.8, 1.8)
                self.target_yaw = (self.mob.yaw + turn + math.pi) % (2 * math.pi) - math.pi
                self.mob.target_yaw = self.target_yaw
                self.stuck_timer = 0.0

        elif self.state == "walk":
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
