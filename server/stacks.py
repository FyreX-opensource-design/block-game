from server.blocks import HOTBAR_SLOTS

MAX_STACK = 64
INVENTORY_SLOTS = 27
CRAFT_SLOTS = 4


def empty_stack():
    return {"id": "", "count": 0}


def make_stack(block_id, count, damage=0):
    from server.items import stack_limit

    if not block_id:
        return empty_stack()
    count = int(count)
    if count <= 0:
        return empty_stack()
    stack = {"id": block_id, "count": max(1, min(stack_limit(block_id), count))}
    if damage:
        stack["damage"] = int(damage)
    return stack


def stack_id(slot):
    if isinstance(slot, dict):
        if int(slot.get("count", 0) or 0) <= 0:
            return ""
        return slot.get("id") or ""
    if isinstance(slot, str):
        return slot
    return ""


def stack_count(slot):
    if isinstance(slot, dict):
        if not slot.get("id"):
            return 0
        return max(0, int(slot.get("count", 0) or 0))
    if isinstance(slot, str) and slot:
        return 1
    return 0


def stack_damage(slot):
    if isinstance(slot, dict):
        return max(0, int(slot.get("damage", 0) or 0))
    return 0


def damage_tool(slot, amount=1):
    from server.items import ITEMS

    item_id = stack_id(slot)
    if not item_id:
        return slot
    item = ITEMS.get(item_id)
    if item is None or not getattr(item, "durability", 0):
        return slot
    cur_damage = stack_damage(slot) + amount
    if cur_damage >= item.durability:
        return empty_stack()
    return make_stack(item_id, stack_count(slot), damage=cur_damage)


def normalize_stack(value):
    from server.items import canonical_item

    if isinstance(value, dict):
        item_id = canonical_item(value.get("id", ""))
        if not item_id:
            return empty_stack()
        damage = int(value.get("damage", 0) or 0)
        return make_stack(item_id, value.get("count", 1), damage=damage)
    item_id = canonical_item(value)
    if not item_id:
        return empty_stack()
    return make_stack(item_id, 1)


def empty_slots(count):
    return [empty_stack() for _ in range(count)]


def normalize_slots(values, count):
    slots = empty_slots(count)
    if values is None:
        return slots
    for index, value in enumerate(list(values)[:count]):
        slots[index] = normalize_stack(value)
    return slots


def normalize_hotbar(values):
    return normalize_slots(values, HOTBAR_SLOTS)


def pack_stack(slot):
    block_id = stack_id(slot)
    if not block_id:
        return ""
    count = stack_count(slot)
    damage = stack_damage(slot)
    if count <= 1 and not damage:
        return block_id
    res = {"id": block_id, "count": count}
    if damage:
        res["damage"] = damage
    return res


def add_to_inventory(slot_groups, item_id, count=1, damage=0):
    from server.items import canonical_item, stack_limit

    item_id = canonical_item(item_id)
    if not item_id or count <= 0:
        return count
    limit = stack_limit(item_id)
    remaining = int(count)
    if not damage:
        for slots in slot_groups:
            for index, raw in enumerate(slots):
                slot = normalize_stack(raw)
                if stack_id(slot) == item_id and stack_count(slot) < limit and not stack_damage(slot):
                    room = limit - stack_count(slot)
                    moved = min(room, remaining)
                    slots[index] = make_stack(item_id, stack_count(slot) + moved)
                    remaining -= moved
                    if remaining <= 0:
                        return 0
    for slots in slot_groups:
        for index, raw in enumerate(slots):
            slot = normalize_stack(raw)
            if not stack_id(slot):
                moved = min(limit, remaining)
                slots[index] = make_stack(item_id, moved, damage=damage)
                remaining -= moved
                if remaining <= 0:
                    return 0
    return remaining

