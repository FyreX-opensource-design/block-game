return {
  id = "zombie",
  name = "Zombie",
  health = 20,
  radius = 0.35,
  height = 1.95,
  walk_speed = 1.2,
  run_speed = 2.4,
  loot = "entities/zombie",
  model = { type = "builtin", id = "zombie", texture = "entity/zombie.png" },
  goals = {
    { priority = 1, type = "melee", damage = 3, reach = 1.5, range = 16 },
    { priority = 2, type = "wander" },
  },
  spawn = {
    group = "hostile",
    order = 0,
    weight = 1,
    headroom = 2,
    group_size = { 1, 2 },
    exclude_biomes = { "desert", "desert_hills" },
    light = {
      sky = { max = 7 },
      green = { max = 7 },
      blue = { max = 7 },
      uv = { max = 7 },
    },
  },
}
