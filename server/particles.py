"""Short-lived world particles.

Hearts rise off livestock that are ready to breed, and a small burst marks
a birth. The simulation is independent of the renderer: positions, velocity,
and lifetime live here, and the client draws whatever is still alive.
"""

import random


HEART_INTERVAL = 0.28
BIRTH_HEARTS = 7


class Particle:
    def __init__(self, kind, x, y, z, vx, vy, vz, life, size):
        self.kind = kind
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.age = 0.0
        self.life = float(life)
        self.size = float(size)

    @property
    def pos(self):
        return (self.x, self.y, self.z)


class ParticleSystem:
    def __init__(self):
        self.particles = []
        self._love_acc = {}
        self._seen = set()

    def spawn(self, kind, pos, count=1, spread=0.12, life=1.15, size=0.32):
        for _ in range(max(1, int(count))):
            self.particles.append(
                Particle(
                    kind,
                    float(pos[0]) + random.uniform(-spread, spread),
                    float(pos[1]) + random.uniform(0.0, spread),
                    float(pos[2]) + random.uniform(-spread, spread),
                    random.uniform(-0.15, 0.15),
                    0.85,
                    random.uniform(-0.15, 0.15),
                    life,
                    size,
                )
            )

    def update(self, dt):
        dt = max(0.0, float(dt))
        alive = []
        for particle in self.particles:
            particle.age += dt
            if particle.age >= particle.life:
                continue
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.z += particle.vz * dt
            alive.append(particle)
        self.particles = alive

    def emit_mob_effects(self, entities, dt):
        """Hearts while an adult is in love, and one burst when a baby is born."""
        dt = max(0.0, float(dt))
        alive_ids = set()
        for entity in entities:
            if not getattr(entity, "is_mob", False) or not getattr(entity, "alive", True):
                continue
            eid = entity.id
            alive_ids.add(eid)
            if getattr(entity, "love_time", 0.0) > 0.0 and not getattr(entity, "is_baby", False):
                acc = self._love_acc.get(eid, 0.0) + dt
                while acc >= HEART_INTERVAL:
                    acc -= HEART_INTERVAL
                    self.spawn(
                        "heart",
                        (entity.pos[0], entity.pos[1] + entity.height + 0.25, entity.pos[2]),
                        count=1,
                        spread=0.12,
                    )
                self._love_acc[eid] = acc
            else:
                self._love_acc.pop(eid, None)
            if eid not in self._seen:
                grow = float(getattr(entity, "grow_time", 0.0) or 0.0)
                if getattr(entity, "is_baby", False) and grow > 0.0 and entity.age <= -(grow - 2.0):
                    self.spawn(
                        "heart",
                        (entity.pos[0], entity.pos[1] + entity.height + 0.2, entity.pos[2]),
                        count=BIRTH_HEARTS,
                        spread=0.28,
                    )
                self._seen.add(eid)
        self._seen.intersection_update(alive_ids)
        for eid in list(self._love_acc):
            if eid not in alive_ids:
                del self._love_acc[eid]
