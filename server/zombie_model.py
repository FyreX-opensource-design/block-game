import math

import numpy as np

from server.player_model import (
    PART_ORDER,
    build_player_meshes,
    model_matrix,
    transform_vertices,
)


def build_zombie_meshes():
    return build_player_meshes()


class ZombieMeshes:
    def __init__(self):
        self.parts = build_zombie_meshes()

    def posed_vertices(self, position, yaw, poses, parts=None):
        names = PART_ORDER if parts is None else parts
        chunks = []
        index_chunks = []
        base = 0
        for name in names:
            vertices, indices = self.parts[name]
            pitch, roll, _yaw = poses.get(name, (0.0, 0.0, 0.0))
            matrix = model_matrix(position, yaw, name, pitch=pitch, roll=roll)
            transformed = transform_vertices(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)


def zombie_pose(time, speed, yaw, pitch=0.0):
    """Calculate poses for zombie parts: outstretched arms, leg swing, head tracking."""
    if speed < 0.08:
        leg_swing = 0.0
    else:
        leg_swing = math.sin(time * 10.0) * min(1.0, speed * 1.6) * 0.7

    # Arms outstretched forward (pitch ~ +pi/2), with subtle swaying
    arm_sway = math.sin(time * 2.5) * 0.06
    arm_pitch = math.pi * 0.5 + arm_sway

    return {
        "head": (pitch, 0.0, 0.0),
        "hat": (pitch, 0.0, 0.0),
        "body": (0.0, 0.0, 0.0),
        "arm_r": (arm_pitch, 0.0, 0.0),
        "arm_l": (arm_pitch, 0.0, 0.0),
        "leg_r": (-leg_swing, 0.0, 0.0),
        "leg_l": (leg_swing, 0.0, 0.0),
    }


def skeleton_pose(time, speed, yaw, pitch=0.0, pull=0.0):
    """Arms hang while walking and rise into a bow draw."""
    pull = max(0.0, min(1.0, float(pull or 0.0)))
    if speed < 0.08:
        leg_swing = 0.0
    else:
        leg_swing = math.sin(time * 10.0) * min(1.0, speed * 1.6) * 0.7
    aim = 1.15 * pull
    walk_arm = 0.0 if pull > 0.05 else leg_swing
    return {
        "head": (pitch, 0.0, 0.0),
        "hat": (pitch, 0.0, 0.0),
        "body": (0.0, 0.0, 0.0),
        "arm_r": (aim - walk_arm, 0.0, 0.0),
        "arm_l": (aim * 0.85 + walk_arm, 0.2 * pull, 0.0),
        "leg_r": (-leg_swing, 0.0, 0.0),
        "leg_l": (leg_swing, 0.0, 0.0),
    }

