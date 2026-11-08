"""Sub-cell (sub-voxel) data structure and utilities for half-block resolution.

A SubCell represents a 1m x 1m x 1m block space subdivided into an 8-octant
(2 x 2 x 2) grid of 0.5m x 0.5m x 0.5m sub-voxels.

Octant indexing (ox, oy, oz in {0, 1}):
  index = oy * 4 + oz * 2 + ox
  - ox: 0 = West half [0.0, 0.5], 1 = East half [0.5, 1.0]
  - oy: 0 = Bottom half [0.0, 0.5], 1 = Top half [0.5, 1.0]
  - oz: 0 = North half [0.0, 0.5], 1 = South half [0.5, 1.0]
"""

from typing import Dict, List, Optional, Tuple

from server.blocks import is_sub_cell

SUB_CELL_BLOCK = "sub_cell"

# Octant index constants
OCTANT_COORDS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0: Bottom North West
    (1, 0, 0),  # 1: Bottom North East
    (0, 0, 1),  # 2: Bottom South West
    (1, 0, 1),  # 3: Bottom South East
    (0, 1, 0),  # 4: Top North West
    (1, 1, 0),  # 5: Top North East
    (0, 1, 1),  # 6: Top South West
    (1, 1, 1),  # 7: Top South East
)


def octant_index(ox: int, oy: int, oz: int) -> int:
    """Convert (ox, oy, oz) in {0, 1} to an index 0..7."""
    return (int(oy) & 1) * 4 + (int(oz) & 1) * 2 + (int(ox) & 1)


def full_block_for(block_or_id: str) -> str:
    """Resolve a slab, vertical slab, or variant ID to its underlying full block ID."""
    if not block_or_id:
        return ""
    from server.blocks import BLOCKS

    b_id = block_or_id if isinstance(block_or_id, str) else getattr(block_or_id, "id", "")
    if "_slab" in b_id or "_vertical" in b_id:
        candidate = (
            b_id.replace("_vertical_slab", "")
            .replace("_vertical", "")
            .replace("_slab", "")
            .replace("_top", "")
        )
        if "_waterlogged" in candidate:
            candidate = candidate.split("_waterlogged")[0]
        for facing in ("_north", "_south", "_east", "_west"):
            if candidate.endswith(facing):
                candidate = candidate[: -len(facing)]
        return candidate if candidate in BLOCKS else b_id
    return b_id


class SubCell:
    """Represents the 8 octants within a 1x1x1m block cell."""

    __slots__ = ("octants", "waterlogged")

    def __init__(self, octants: Optional[List[str]] = None, waterlogged: Optional[List[bool]] = None):
        if octants is None:
            self.octants = [""] * 8
        else:
            self.octants = [full_block_for(o) if o else "" for o in octants]
            if len(self.octants) < 8:
                self.octants.extend([""] * (8 - len(self.octants)))
            elif len(self.octants) > 8:
                self.octants = self.octants[:8]

        if waterlogged is None:
            self.waterlogged = [False] * 8
        else:
            self.waterlogged = list(waterlogged)
            if len(self.waterlogged) < 8:
                self.waterlogged.extend([False] * (8 - len(self.waterlogged)))
            elif len(self.waterlogged) > 8:
                self.waterlogged = self.waterlogged[:8]

    def copy(self) -> "SubCell":
        return SubCell(list(self.octants), list(self.waterlogged))

    def is_empty(self) -> bool:
        """Return True if all 8 octants are empty."""
        return not any(self.octants)

    def is_full(self) -> bool:
        """Return True if all 8 octants are filled with the same non-empty material."""
        first = self.octants[0]
        if not first:
            return False
        return all(o == first for o in self.octants)

    def get_octant(self, ox: int, oy: int, oz: int) -> str:
        return self.octants[octant_index(ox, oy, oz)]

    def set_octant(self, ox: int, oy: int, oz: int, block_id: str, waterlogged: bool = False) -> None:
        idx = octant_index(ox, oy, oz)
        self.octants[idx] = full_block_for(block_id) if block_id else ""
        self.waterlogged[idx] = bool(waterlogged)

    def count_occupied(self) -> int:
        return sum(1 for o in self.octants if o)

    def occupied_indices(self) -> List[int]:
        return [i for i, o in enumerate(self.octants) if o]

    def to_dict(self) -> dict:
        return {
            "octants": list(self.octants),
            "waterlogged": list(self.waterlogged),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubCell":
        return cls(data.get("octants"), data.get("waterlogged"))

    # Helper factories for standard geometric shapes

    @classmethod
    def from_horizontal_slab(cls, base_id: str, half: str = "bottom", waterlogged: bool = False) -> "SubCell":
        """Create a SubCell representing a horizontal bottom or top slab."""
        cell = cls()
        oy = 0 if half == "bottom" else 1
        for oz in (0, 1):
            for ox in (0, 1):
                cell.set_octant(ox, oy, oz, base_id, waterlogged)
        return cell

    @classmethod
    def from_vertical_slab(cls, base_id: str, facing: str = "south", waterlogged: bool = False) -> "SubCell":
        """Create a SubCell representing a vertical slab with the given facing."""
        cell = cls()
        facing = facing.lower()
        if facing == "north":
            # Solid half is north (oz=0)
            for oy in (0, 1):
                for ox in (0, 1):
                    cell.set_octant(ox, oy, 0, base_id, waterlogged)
        elif facing == "south":
            # Solid half is south (oz=1)
            for oy in (0, 1):
                for ox in (0, 1):
                    cell.set_octant(ox, oy, 1, base_id, waterlogged)
        elif facing == "west":
            # Solid half is west (ox=0)
            for oy in (0, 1):
                for oz in (0, 1):
                    cell.set_octant(0, oy, oz, base_id, waterlogged)
        elif facing == "east":
            # Solid half is east (ox=1)
            for oy in (0, 1):
                for oz in (0, 1):
                    cell.set_octant(1, oy, oz, base_id, waterlogged)
        return cell

    def bounding_boxes(self, x: float, y: float, z: float) -> List[Tuple[float, float, float, float, float, float]]:
        """Return world-space AABBs for all occupied octants in this cell."""
        boxes: List[Tuple[float, float, float, float, float, float]] = []
        for i, block_id in enumerate(self.octants):
            if not block_id:
                continue
            ox, oy, oz = OCTANT_COORDS[i]
            bx0 = x + ox * 0.5
            by0 = y + oy * 0.5
            bz0 = z + oz * 0.5
            boxes.append((bx0, by0, bz0, bx0 + 0.5, by0 + 0.5, bz0 + 0.5))
        return boxes


def calculate_sub_cell_placement(
    world,
    hit: dict,
    held_block_id: str,
    player_yaw: float = 0.0,
) -> Optional[Tuple[int, int, int, SubCell]]:
    """Determine the resulting SubCell when placing held_block_id against hit.

    Returns (target_x, target_y, target_z, new_sub_cell) or None if placement is invalid.
    """
    from server.blocks import (
        AIR,
        SUB_CELL_BLOCK,
        get_slab_info,
        get_vertical_slab_info,
        is_slab,
        is_sub_cell,
        is_vertical_slab,
        player_facing_from_yaw,
    )

    held_slab = get_slab_info(held_block_id)
    held_vslab = get_vertical_slab_info(held_block_id)

    if not held_slab and not held_vslab:
        return None
    elif held_slab:
        base_mat = held_slab["base_id"]
    else:
        base_mat = held_vslab["base_id"]

    hx, hy, hz = hit["x"], hit["y"], hit["z"]
    nx, ny, nz = hit["nx"], hit["ny"], hit["nz"]
    hit_x = hit.get("hit_x", hx + 0.5)
    hit_y = hit.get("hit_y", hy + 0.5)
    hit_z = hit.get("hit_z", hz + 0.5)
    local_y = hit.get("local_y", hit_y - hy)
    target_block = world.get_block(hx, hy, hz)

    target_is_sub = target_block == SUB_CELL_BLOCK or is_sub_cell(target_block)
    target_sub = world.get_sub_cell(hx, hy, hz) if target_is_sub else None

    if target_sub is not None:
        octant = hit.get("octant")
        if octant is None:
            ox = 1 if (hit_x - hx) >= 0.5 else 0
            oy = 1 if local_y >= 0.5 else 0
            oz = 1 if (hit_z - hz) >= 0.5 else 0
            octant = (ox, oy, oz)
        ox, oy, oz = octant
        nox = ox + nx
        noy = oy + ny
        noz = oz + nz

        if 0 <= nox <= 1 and 0 <= noy <= 1 and 0 <= noz <= 1:
            cell_x, cell_y, cell_z = hx, hy, hz
            sub = target_sub.copy()
            target_oy = noy
        else:
            cell_x = hx + nx
            cell_y = hy + ny
            cell_z = hz + nz
            existing_dest = world.get_sub_cell(cell_x, cell_y, cell_z)
            if existing_dest is not None:
                sub = existing_dest.copy()
            else:
                dest_block = world.get_block(cell_x, cell_y, cell_z)
                if dest_block not in ("", AIR, None):
                    d_slab = get_slab_info(dest_block)
                    d_vslab = get_vertical_slab_info(dest_block)
                    if d_slab and not d_slab["double"]:
                        sub = SubCell.from_horizontal_slab(d_slab["base_id"], d_slab["half"])
                    elif d_vslab:
                        sub = SubCell.from_vertical_slab(d_vslab["base_id"], d_vslab["facing"])
                    else:
                        return None
                else:
                    if not held_slab and not held_vslab:
                        return None
                    sub = SubCell()
            target_oy = noy % 2
    else:
        target_slab = get_slab_info(target_block)
        target_vslab = get_vertical_slab_info(target_block)

        placed_in_target = False
        if target_slab and not target_slab["double"]:
            if target_slab["half"] == "bottom" and held_slab and (ny == 1 or (ny == 0 and local_y >= 0.5)):
                cell_x, cell_y, cell_z = hx, hy, hz
                sub = SubCell.from_horizontal_slab(target_slab["base_id"], "bottom")
                target_oy = 1
                placed_in_target = True
            elif target_slab["half"] == "top" and (ny == -1 or (ny == 0 and local_y < 0.5)):
                cell_x, cell_y, cell_z = hx, hy, hz
                sub = SubCell.from_horizontal_slab(target_slab["base_id"], "top")
                target_oy = 0
                placed_in_target = True

        if not placed_in_target and target_vslab:
            from server.blocks import FACING_NORMALS, OPPOSITE_FACING

            back_normal = FACING_NORMALS.get(OPPOSITE_FACING.get(target_vslab["facing"], ""), (0, 0, 0))
            if (nx, ny, nz) == back_normal:
                cell_x, cell_y, cell_z = hx, hy, hz
                sub = SubCell.from_vertical_slab(target_vslab["base_id"], target_vslab["facing"])
                target_oy = 1 if local_y >= 0.5 else 0
                placed_in_target = True

        if not placed_in_target:
            cell_x = hx + nx
            cell_y = hy + ny
            cell_z = hz + nz
            existing_dest = world.get_sub_cell(cell_x, cell_y, cell_z)
            if existing_dest is not None:
                sub = existing_dest.copy()
            else:
                dest_block = world.get_block(cell_x, cell_y, cell_z)
                if dest_block not in ("", AIR, None):
                    d_slab = get_slab_info(dest_block)
                    d_vslab = get_vertical_slab_info(dest_block)
                    if d_slab and not d_slab["double"]:
                        sub = SubCell.from_horizontal_slab(d_slab["base_id"], d_slab["half"])
                    elif d_vslab:
                        sub = SubCell.from_vertical_slab(d_vslab["base_id"], d_vslab["facing"])
                    else:
                        return None
                else:
                    if not held_slab and not held_vslab:
                        return None
                    sub = SubCell()

            if ny == 1:
                target_oy = 0
            elif ny == -1:
                target_oy = 1
            else:
                target_oy = 1 if local_y >= 0.5 else 0

    from server.blocks import has_half_offset_top, is_offset_block, is_offset_torch

    above_block = world.get_block(cell_x, cell_y + 1, cell_z)
    if (has_half_offset_top(above_block) or is_offset_block(above_block) or is_offset_torch(above_block)) and target_oy == 1:
        return None

    needed_octants = []
    if held_slab:
        for cx in (0, 1):
            for cz in (0, 1):
                needed_octants.append((cx, target_oy, cz))
    elif held_vslab:
        facing = player_facing_from_yaw(player_yaw)
        if facing == "north":
            for cy in (0, 1):
                for cx in (0, 1):
                    needed_octants.append((cx, cy, 0))
        elif facing == "south":
            for cy in (0, 1):
                for cx in (0, 1):
                    needed_octants.append((cx, cy, 1))
        elif facing == "west":
            for cy in (0, 1):
                for cz in (0, 1):
                    needed_octants.append((0, cy, cz))
        else:  # east
            for cy in (0, 1):
                for cz in (0, 1):
                    needed_octants.append((1, cy, cz))

    if not needed_octants:
        return None

    conflict = False
    for ox, oy, oz in needed_octants:
        curr = sub.get_octant(ox, oy, oz)
        if curr != "":
            conflict = True
            break

    if conflict:
        if held_vslab:
            if any(sub.get_octant(cx, 1, cz) != "" for cx in (0, 1) for cz in (0, 1)) and all(
                sub.get_octant(ox, 0, oz) == "" for ox, oy, oz in needed_octants
            ):
                needed_octants = list(set((ox, 0, oz) for ox, oy, oz in needed_octants))
                conflict = False
            elif any(oz == 0 for _, _, oz in needed_octants) and all(sub.get_octant(cx, cy, 1) == "" for cx in (0, 1) for cy in (0, 1)):
                needed_octants = [(cx, cy, 1) for cx in (0, 1) for cy in (0, 1)]
                conflict = False
            elif any(oz == 1 for _, _, oz in needed_octants) and all(sub.get_octant(cx, cy, 0) == "" for cx in (0, 1) for cy in (0, 1)):
                needed_octants = [(cx, cy, 0) for cx in (0, 1) for cy in (0, 1)]
                conflict = False
            elif any(ox == 0 for ox, _, _ in needed_octants) and all(sub.get_octant(1, cy, cz) == "" for cy in (0, 1) for cz in (0, 1)):
                needed_octants = [(1, cy, cz) for cy in (0, 1) for cz in (0, 1)]
                conflict = False
            elif any(ox == 1 for ox, _, _ in needed_octants) and all(sub.get_octant(0, cy, cz) == "" for cy in (0, 1) for cz in (0, 1)):
                needed_octants = [(0, cy, cz) for cy in (0, 1) for cz in (0, 1)]
                conflict = False

    if conflict:
        return None

    for ox, oy, oz in needed_octants:
        sub.set_octant(ox, oy, oz, base_mat)

    return cell_x, cell_y, cell_z, sub


def calculate_horizontal_offset_placement(
    world,
    hit: dict,
    held_block_id: str,
    player_yaw: float = 0.0,
) -> Optional[List[Tuple[int, int, int, SubCell]]]:
    """Calculate sub-cell updates when placing a block horizontally offset against a vertical slab or sub-cell.

    Returns a list of (x, y, z, sub_cell) tuples to apply to world, or None.
    """
    from server.blocks import (
        AIR,
        FACING_NORMALS,
        OPPOSITE_FACING,
        SUB_CELL_BLOCK,
        get_slab_info,
        get_vertical_slab_info,
        is_slab,
        is_sub_cell,
        is_vertical_slab,
        player_facing_from_yaw,
    )

    held_slab = get_slab_info(held_block_id)
    held_vslab = get_vertical_slab_info(held_block_id)
    base_mat = full_block_for(held_block_id)

    hx, hy, hz = hit["x"], hit["y"], hit["z"]
    nx, ny, nz = hit["nx"], hit["ny"], hit["nz"]
    target_block = world.get_block(hx, hy, hz)

    target_is_sub = target_block == SUB_CELL_BLOCK or is_sub_cell(target_block)
    target_vslab = get_vertical_slab_info(target_block)

    # 1. Full block placed against back face of a vertical slab
    if not held_slab and not held_vslab and target_vslab is not None:
        facing = target_vslab["facing"]
        back_normal = FACING_NORMALS.get(OPPOSITE_FACING.get(facing, ""), (0, 0, 0))
        if (nx, ny, nz) == back_normal:
            sub1 = (
                world.get_sub_cell(hx, hy, hz).copy()
                if target_is_sub
                else SubCell.from_vertical_slab(target_vslab["base_id"], facing)
            )
            cell2_pos = (hx + nx, hy + ny, hz + nz)
            blk2 = world.get_block(*cell2_pos)
            sub2 = world.get_sub_cell(*cell2_pos)
            if sub2 is not None:
                sub2 = sub2.copy()
            elif blk2 in ("", AIR, None):
                sub2 = SubCell()
            else:
                return None

            if nz != 0:
                oz1 = 1 if nz > 0 else 0
                oz2 = 0 if nz > 0 else 1
                octs1 = [(ox, oy, oz1) for ox in (0, 1) for oy in (0, 1)]
                octs2 = [(ox, oy, oz2) for ox in (0, 1) for oy in (0, 1)]
            elif nx != 0:
                ox1 = 1 if nx > 0 else 0
                ox2 = 0 if nx > 0 else 1
                octs1 = [(ox1, oy, oz) for oy in (0, 1) for oz in (0, 1)]
                octs2 = [(ox2, oy, oz) for oy in (0, 1) for oz in (0, 1)]
            else:
                return None

            if any(sub1.get_octant(ox, oy, oz) != "" for ox, oy, oz in octs1):
                return None
            if any(sub2.get_octant(ox, oy, oz) != "" for ox, oy, oz in octs2):
                return None

            for ox, oy, oz in octs1:
                sub1.set_octant(ox, oy, oz, base_mat)
            for ox, oy, oz in octs2:
                sub2.set_octant(ox, oy, oz, base_mat)

            return [(hx, hy, hz, sub1), (cell2_pos[0], cell2_pos[1], cell2_pos[2], sub2)]

    # 2. Full block placed against an exposed face of an already horizontally offset block
    if not held_slab and not held_vslab and target_is_sub:
        sub = world.get_sub_cell(hx, hy, hz)
        if sub is not None:
            octant = hit.get("octant")
            if octant is not None:
                ox, oy, oz = octant
                nox = ox + nx
                noz = oz + nz
                if nz != 0 and 0 <= noz <= 1:
                    oz1 = noz
                    oz2 = 0 if nz > 0 else 1
                    cell2_pos = (hx, hy, hz + nz)
                    blk2 = world.get_block(*cell2_pos)
                    sub2 = world.get_sub_cell(*cell2_pos)
                    if sub2 is not None:
                        sub2 = sub2.copy()
                    elif blk2 in ("", AIR, None):
                        sub2 = SubCell()
                    else:
                        return None
                    octs1 = [(cx, cy, oz1) for cx in (0, 1) for cy in (0, 1)]
                    octs2 = [(cx, cy, oz2) for cx in (0, 1) for cy in (0, 1)]
                    sub1 = sub.copy()
                    if any(sub1.get_octant(cx, cy, cz) != "" for cx, cy, cz in octs1):
                        return None
                    if any(sub2.get_octant(cx, cy, cz) != "" for cx, cy, cz in octs2):
                        return None
                    for cx, cy, cz in octs1:
                        sub1.set_octant(cx, cy, cz, base_mat)
                    for cx, cy, cz in octs2:
                        sub2.set_octant(cx, cy, cz, base_mat)
                    return [(hx, hy, hz, sub1), (cell2_pos[0], cell2_pos[1], cell2_pos[2], sub2)]
                elif nx != 0 and 0 <= nox <= 1:
                    ox1 = nox
                    ox2 = 0 if nx > 0 else 1
                    cell2_pos = (hx + nx, hy, hz)
                    blk2 = world.get_block(*cell2_pos)
                    sub2 = world.get_sub_cell(*cell2_pos)
                    if sub2 is not None:
                        sub2 = sub2.copy()
                    elif blk2 in ("", AIR, None):
                        sub2 = SubCell()
                    else:
                        return None
                    octs1 = [(ox1, cy, cz) for cy in (0, 1) for cz in (0, 1)]
                    octs2 = [(ox2, cy, cz) for cy in (0, 1) for cz in (0, 1)]
                    sub1 = sub.copy()
                    if any(sub1.get_octant(cx, cy, cz) != "" for cx, cy, cz in octs1):
                        return None
                    if any(sub2.get_octant(cx, cy, cz) != "" for cx, cy, cz in octs2):
                        return None
                    for cx, cy, cz in octs1:
                        sub1.set_octant(cx, cy, cz, base_mat)
                    for cx, cy, cz in octs2:
                        sub2.set_octant(cx, cy, cz, base_mat)
                    return [(hx, hy, hz, sub1), (cell2_pos[0], cell2_pos[1], cell2_pos[2], sub2)]

    return None
