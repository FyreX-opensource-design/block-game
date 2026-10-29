from server.items import stack_limit
from server.recipes import fuel_burn_time, smelt_match
from server.stacks import empty_stack, make_stack, normalize_stack, stack_count, stack_id


class Furnace:
    def __init__(
        self,
        x=0,
        y=0,
        z=0,
        input_stack=None,
        fuel_stack=None,
        output_stack=None,
        burn_time=0.0,
        burn_duration=0.0,
        cook_time=0.0,
        cook_duration=10.0,
        block_id="furnace",
        recipe_types=None,
        cook_speed=1.0,
    ):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.input = normalize_stack(input_stack) if input_stack else empty_stack()
        self.fuel = normalize_stack(fuel_stack) if fuel_stack else empty_stack()
        self.output = normalize_stack(output_stack) if output_stack else empty_stack()
        self.burn_time = float(burn_time)
        self.burn_duration = float(burn_duration)
        self.cook_time = float(cook_time)
        self.cook_duration = float(cook_duration)
        self.block_id = str(block_id)

        from server.blocks import BLOCKS

        b_obj = BLOCKS.get(self.block_id)
        if recipe_types is not None:
            self.recipe_types = (str(recipe_types),) if isinstance(recipe_types, str) else tuple(str(t) for t in recipe_types)
        elif b_obj is not None and getattr(b_obj, "recipe_types", None) is not None:
            self.recipe_types = b_obj.recipe_types
        else:
            self.recipe_types = ("smelting",)

        if cook_speed != 1.0:
            self.cook_speed = float(cook_speed)
        elif b_obj is not None and getattr(b_obj, "cook_speed", 1.0) != 1.0:
            self.cook_speed = float(b_obj.cook_speed)
        else:
            self.cook_speed = 1.0

    @property
    def is_lit(self):
        return self.burn_time > 0

    def tick(self, dt):
        if self.burn_time > 0:
            self.burn_time = max(0.0, self.burn_time - dt)

        in_id = stack_id(self.input)
        in_cnt = stack_count(self.input)
        smelt = smelt_match(in_id, recipe_types=self.recipe_types) if (in_id and in_cnt > 0) else None

        can_smelt = False
        if smelt is not None:
            res_id, res_cnt, res_time, _res_exp = smelt
            out_id = stack_id(self.output)
            out_cnt = stack_count(self.output)
            if not out_id or (out_id == res_id and out_cnt + res_cnt <= stack_limit(res_id)):
                can_smelt = True
                self.cook_duration = res_time

        # If input cannot be smelted, reset cook_time immediately
        if not can_smelt:
            self.cook_time = 0.0
        else:
            if self.burn_time <= 0:
                fuel_id = stack_id(self.fuel)
                fuel_cnt = stack_count(self.fuel)
                fuel_time = fuel_burn_time(fuel_id)
                if fuel_time > 0 and fuel_cnt > 0:
                    new_cnt = fuel_cnt - 1
                    self.fuel = make_stack(fuel_id, new_cnt) if new_cnt > 0 else empty_stack()
                    self.burn_duration = fuel_time
                    self.burn_time = fuel_time

            if self.burn_time > 0:
                self.cook_time += dt * self.cook_speed
                if self.cook_time >= self.cook_duration:
                    self.cook_time = 0.0
                    res_id, res_cnt, _res_time, _res_exp = smelt
                    new_in = in_cnt - 1
                    self.input = make_stack(in_id, new_in) if new_in > 0 else empty_stack()
                    out_cnt = stack_count(self.output) + res_cnt
                    self.output = make_stack(res_id, out_cnt)
            else:
                self.cook_time = max(0.0, self.cook_time - dt * 2)

        if self.burn_time <= 0:
            self.burn_time = 0.0
            self.burn_duration = 0.0

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "input": dict(self.input),
            "fuel": dict(self.fuel),
            "output": dict(self.output),
            "burn_time": self.burn_time,
            "burn_duration": self.burn_duration,
            "cook_time": self.cook_time,
            "cook_duration": self.cook_duration,
            "block_id": self.block_id,
            "recipe_types": list(self.recipe_types),
            "cook_speed": self.cook_speed,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            z=data.get("z", 0),
            input_stack=data.get("input"),
            fuel_stack=data.get("fuel"),
            output_stack=data.get("output"),
            burn_time=data.get("burn_time", 0.0),
            burn_duration=data.get("burn_duration", 0.0),
            cook_time=data.get("cook_time", 0.0),
            cook_duration=data.get("cook_duration", 10.0),
            block_id=data.get("block_id", "furnace"),
            recipe_types=data.get("recipe_types"),
            cook_speed=data.get("cook_speed", 1.0),
        )

