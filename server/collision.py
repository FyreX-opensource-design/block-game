"""Collision between living bodies.

Players and mobs block each other. Dropped items, arrows, and falling blocks
stay passable so they can be walked over and picked up. A body you are already
inside of does not stop a step that moves you away from it, and after the move
any leftover overlap is pushed apart.
"""


def iter_solid_bodies(world, ignore):
    """Yield mobs and players that should stop movement, other than ``ignore``."""
    if world is None:
        return
    from server.player import Player

    entities = getattr(world, "entities", None)
    if isinstance(entities, (list, tuple)):
        for entity in entities:
            if entity is ignore or getattr(entity, "is_mob", False) is not True:
                continue
            if getattr(entity, "alive", True) is not True:
                continue
            yield entity
    players = getattr(world, "players", None)
    if isinstance(players, dict):
        for player in players.values():
            if player is ignore or not isinstance(player, Player):
                continue
            if getattr(player, "dead", False):
                continue
            yield player
    remotes = getattr(world, "remote_players", None)
    if isinstance(remotes, dict):
        remote_values = remotes.values()
    elif isinstance(remotes, (list, tuple)):
        remote_values = remotes
    else:
        remote_values = ()
    seen = set(id(player) for player in (players.values() if isinstance(players, dict) else ()))
    for player in remote_values:
        if player is ignore or id(player) in seen or not isinstance(player, Player):
            continue
        if getattr(player, "dead", False):
            continue
        yield player


def overlaps_body(px, py, pz, radius, height, other, margin=0.001):
    other_radius = float(getattr(other, "radius", 0.3) or 0.3)
    other_height = float(getattr(other, "height", 1.8) or 1.8)
    ox, oy, oz = other.pos
    return (
        abs(px - ox) < radius + other_radius - margin
        and abs(pz - oz) < radius + other_radius - margin
        and py < oy + other_height - margin
        and py + height > oy + margin
    )


def hits_body(body, px, py, pz, world):
    """True when the candidate position newly overlaps a living body."""
    radius = float(body.radius)
    height = float(body.height)
    ox, oy, oz = body.pos
    for other in iter_solid_bodies(world, body):
        if not overlaps_body(px, py, pz, radius, height, other):
            continue
        if overlaps_body(ox, oy, oz, radius, height, other):
            old = (ox - other.pos[0]) ** 2 + (oz - other.pos[2]) ** 2
            new = (px - other.pos[0]) ** 2 + (pz - other.pos[2]) ** 2
            if new > old + 1e-10:
                continue
        return True
    return False


def _terrain_blocked(body):
    checker = getattr(body, "collides_blocks", None)
    if checker is None:
        return False
    return checker(body.pos[0], body.pos[1], body.pos[2])


def separate_from_bodies(body, world):
    """Push ``body`` out of any living body it currently overlaps."""
    if world is None or getattr(body, "is_remote", False):
        return
    radius = float(body.radius)
    height = float(body.height)
    for _ in range(4):
        other = None
        for candidate in iter_solid_bodies(world, body):
            if overlaps_body(body.pos[0], body.pos[1], body.pos[2], radius, height, candidate):
                other = candidate
                break
        if other is None:
            return
        dx = body.pos[0] - other.pos[0]
        dz = body.pos[2] - other.pos[2]
        limit = radius + float(other.radius)
        pen_x = limit - abs(dx)
        pen_z = limit - abs(dz)
        if pen_x <= 0.0 or pen_z <= 0.0:
            return
        if pen_x <= pen_z:
            axis = 0
            pen = pen_x
            sign = 1.0 if dx > 0.0 or (dx == 0.0 and dz == 0.0) else -1.0
        else:
            axis = 2
            pen = pen_z
            sign = 1.0 if dz > 0.0 else -1.0
        body.pos[axis] += sign * (pen + 0.01)
        if _terrain_blocked(body):
            body.pos[axis] -= sign * (pen + 0.01)
            other_axis = 2 if axis == 0 else 0
            other_sign = 1.0 if (dz if other_axis == 2 else dx) >= 0.0 else -1.0
            other_pen = pen_z if other_axis == 2 else pen_x
            body.pos[other_axis] += other_sign * (other_pen + 0.01)
            if _terrain_blocked(body):
                body.pos[other_axis] -= other_sign * (other_pen + 0.01)
                return
