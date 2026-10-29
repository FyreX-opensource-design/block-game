from server.blocks import atlas_bytes
from server.items import item_atlas_bytes


def build_atlas():
    return atlas_bytes()


def build_item_atlas():
    return item_atlas_bytes()
