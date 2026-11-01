import math
import moderngl
import numpy as np
from pyglet import gl

from server.blocks import ASSETS, BLOCKS, CHUNK_SIZE, FACE_INDEX, HOTBAR_SLOTS
from server.images import read_png
from server.items import ITEMS
from server.mesh import CORNERS
from server.stacks import stack_id

from client.geom import ortho, write_matrix
from client.player_render import (
    PlayerMeshes,
    dropped_item_matrix,
    falling_block_mesh,
    first_person_pose,
    held_item_mesh,
    item_hold_matrix,
    selected_item_id,
    third_person_pose,
)

SKY = (0.56, 0.78, 0.96)
FOG_NEAR = 22.0
FOG_FAR = 58.0

# A small pixel-art heart used for the survival health bar.
HEART_PATTERN = (
    "0110110",
    "1111111",
    "1111111",
    "0111110",
    "0011100",
    "0001000",
)
HEART_PATTERN_WIDTH = len(HEART_PATTERN[0])
HEART_PATTERN_HEIGHT = len(HEART_PATTERN)

WORLD_VERT = """
#version 330 core
uniform mat4 vp;
uniform vec3 origin;
uniform vec3 camera;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_shade;
in float in_skylight;
in vec3 in_blocklight_rgb;
in float in_blocklight_uv;
in float in_blocklight_ir;
in vec3 in_color;

out vec2 v_uv;
out vec3 v_normal;
out float v_shade;
out float v_dist;
out float v_skylight;
out vec3 v_blocklight_rgb;
out float v_blocklight_uv;
out float v_blocklight_ir;
out vec3 v_color;

void main() {
    vec3 world = in_pos + origin;
    gl_Position = vp * vec4(world, 1.0);
    v_uv = in_uv;
    v_normal = in_normal;
    v_shade = in_shade;
    v_dist = distance(world, camera);
    v_skylight = in_skylight;
    v_blocklight_rgb = in_blocklight_rgb;
    v_blocklight_uv = in_blocklight_uv;
    v_blocklight_ir = in_blocklight_ir;
    v_color = in_color;
}
"""

WORLD_FRAG = """
#version 330 core
uniform sampler2D atlas;
uniform vec3 fog_color;
uniform float fog_near;
uniform float fog_far;
uniform vec3 sun_dir;
uniform vec3 sun_color;
uniform float sky_intensity;
uniform int vision_mode;

in vec2 v_uv;
in vec3 v_normal;
in float v_shade;
in float v_dist;
in float v_skylight;
in vec3 v_blocklight_rgb;
in float v_blocklight_uv;
in float v_blocklight_ir;
in vec3 v_color;

out vec4 fragColor;

void main() {
    vec4 texel = texture(atlas, v_uv);
    if (texel.a < 0.5) discard;

    vec3 base_rgb = texel.rgb;
    if (abs(v_color.r - 1.0) > 0.001 || abs(v_color.g - 1.0) > 0.001 || abs(v_color.b - 1.0) > 0.001) {
        float diff = texel.g - max(texel.r, texel.b);
        float is_grass = clamp(diff / 0.05, 0.0, 1.0);
        float lum = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
        vec3 tinted_grass = clamp(lum * v_color * 2.1, 0.0, 1.0);
        base_rgb = mix(texel.rgb, tinted_grass, is_grass);
    }

    vec3 norm = normalize(v_normal);
    float lambert = max(dot(norm, normalize(sun_dir)), 0.0);

    // Light and shadow: direct sunlight is shadowed when skylight is low
    float effective_skylight = v_skylight * sky_intensity;
    float sun_factor = smoothstep(0.4, 0.95, v_skylight) * sky_intensity;
    float direct_sun = lambert * 0.55 * sun_factor;
    float ambient_sky = effective_skylight * 0.70;

    vec3 sky_color = sun_color * (ambient_sky + direct_sun) * v_shade;
    vec3 block_rgb = v_blocklight_rgb;
    vec3 uv_fluorescence = vec3(0.40, 0.15, 0.70) * v_blocklight_uv;
    vec3 ir_warmth = vec3(0.65, 0.25, 0.08) * v_blocklight_ir;
    vec3 ambient_base = vec3(0.04);

    vec3 final_color;
    if (vision_mode == 0) {
        // Normal vision
        vec3 total_light = ambient_base + sky_color + block_rgb + uv_fluorescence * 0.35 + ir_warmth * 0.35;
        vec3 lit = base_rgb * total_light;
        float fog = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
        final_color = mix(lit, fog_color, fog);
    } else if (vision_mode == 1) {
        // UV Vision mode
        float uv_val = v_blocklight_uv;
        vec3 uv_color = vec3(0.35, 0.1, 0.85) * uv_val * 1.5 + vec3(0.1, 0.85, 0.75) * pow(uv_val, 2.0);
        vec3 dim_scene = base_rgb * 0.15 * (v_skylight * 0.5 + 0.1);
        vec3 lit = dim_scene + uv_color;
        float fog = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
        final_color = mix(lit, vec3(0.05, 0.02, 0.1), fog);
    } else if (vision_mode == 2) {
        // Infrared / Thermal mode (FLIR false-color)
        float ir_val = v_blocklight_ir;
        vec3 heat_color;
        if (ir_val < 0.25) {
            heat_color = mix(vec3(0.02, 0.02, 0.15), vec3(0.2, 0.0, 0.4), ir_val / 0.25);
        } else if (ir_val < 0.5) {
            heat_color = mix(vec3(0.2, 0.0, 0.4), vec3(0.85, 0.1, 0.1), (ir_val - 0.25) / 0.25);
        } else if (ir_val < 0.75) {
            heat_color = mix(vec3(0.85, 0.1, 0.1), vec3(1.0, 0.7, 0.1), (ir_val - 0.5) / 0.25);
        } else {
            heat_color = mix(vec3(1.0, 0.7, 0.1), vec3(1.0, 1.0, 1.0), (ir_val - 0.75) / 0.25);
        }
        float relief = 0.8 + 0.2 * lambert;
        final_color = heat_color * relief;
    } else {
        // Full Spectrum / Debug mode
        vec3 sky_viz = vec3(v_skylight);
        vec3 block_viz = v_blocklight_rgb;
        vec3 uv_viz = vec3(0.5, 0.0, 0.8) * v_blocklight_uv;
        vec3 ir_viz = vec3(1.0, 0.4, 0.0) * v_blocklight_ir;
        vec3 debug_color = sky_viz * 0.35 + block_viz * 0.35 + uv_viz * 0.35 + ir_viz * 0.35;
        final_color = mix(base_rgb * 0.3, debug_color, 0.75);
    }

    fragColor = vec4(final_color, 1.0);
}
"""

TRANSLUCENT_FRAG = """
#version 330 core
uniform sampler2D atlas;
uniform vec3 fog_color;
uniform float fog_near;
uniform float fog_far;
uniform vec3 sun_dir;
uniform vec3 sun_color;
uniform float sky_intensity;
uniform int vision_mode;

in vec2 v_uv;
in vec3 v_normal;
in float v_shade;
in float v_dist;
in float v_skylight;
in vec3 v_blocklight_rgb;
in float v_blocklight_uv;
in float v_blocklight_ir;
in vec3 v_color;

out vec4 fragColor;

void main() {
    vec4 texel = texture(atlas, v_uv);
    if (texel.a < 0.05) discard;

    vec3 base_rgb = texel.rgb;
    if (abs(v_color.r - 1.0) > 0.001 || abs(v_color.g - 1.0) > 0.001 || abs(v_color.b - 1.0) > 0.001) {
        float diff = texel.g - max(texel.r, texel.b);
        float is_grass = clamp(diff / 0.05, 0.0, 1.0);
        float lum = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
        vec3 tinted_grass = clamp(lum * v_color * 2.1, 0.0, 1.0);
        base_rgb = mix(texel.rgb, tinted_grass, is_grass);
    }

    vec3 norm = normalize(v_normal);
    float lambert = max(dot(norm, normalize(sun_dir)), 0.0);

    float effective_skylight = v_skylight * sky_intensity;
    float sun_factor = smoothstep(0.4, 0.95, v_skylight) * sky_intensity;
    float direct_sun = lambert * 0.55 * sun_factor;
    float ambient_sky = effective_skylight * 0.70;

    vec3 sky_color = sun_color * (ambient_sky + direct_sun) * v_shade;
    vec3 block_rgb = v_blocklight_rgb;
    vec3 uv_fluorescence = vec3(0.40, 0.15, 0.70) * v_blocklight_uv;
    vec3 ir_warmth = vec3(0.65, 0.25, 0.08) * v_blocklight_ir;
    vec3 ambient_base = vec3(0.04);

    vec3 total_light = ambient_base + sky_color + block_rgb + uv_fluorescence * 0.35 + ir_warmth * 0.35;
    vec3 color = base_rgb * total_light;
    float fog = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
    fragColor = vec4(mix(color, fog_color, fog), texel.a);
}
"""

TEXTURE_VERT = """
#version 330 core
uniform mat4 mvp;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_uv = in_uv;
}
"""

TEXTURE_FRAG = """
#version 330 core
uniform sampler2D atlas;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 texel = texture(atlas, v_uv);
    if (texel.a < 0.5) discard;
    fragColor = texel;
}
"""

COLOR_VERT = """
#version 330 core
uniform mat4 mvp;
in vec3 in_pos;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

COLOR_FRAG = """
#version 330 core
uniform vec4 color;
out vec4 fragColor;
void main() {
    fragColor = color;
}
"""

ENTITY_VERT = """
#version 330 core
uniform mat4 vp;
uniform vec3 camera;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_normal;
out float v_dist;
void main() {
    gl_Position = vp * vec4(in_pos, 1.0);
    v_uv = in_uv;
    v_normal = in_normal;
    v_dist = distance(in_pos, camera);
}
"""

ENTITY_FRAG = """
#version 330 core
uniform sampler2D atlas;
uniform vec3 fog_color;
uniform float fog_near;
uniform float fog_far;
uniform float use_fog;
uniform vec3 sun_dir;
uniform vec3 sun_color;
uniform float sky_intensity;
uniform int vision_mode;
uniform float entity_skylight;
uniform vec3 entity_blocklight_rgb;
uniform float entity_blocklight_uv;
uniform float entity_blocklight_ir;
uniform float hurt_flash;

in vec2 v_uv;
in vec3 v_normal;
in float v_dist;
out vec4 fragColor;

void main() {
    vec4 texel = texture(atlas, v_uv);
    if (texel.a < 0.5) discard;

    vec3 norm = normalize(v_normal);
    float lambert = max(dot(norm, normalize(sun_dir)), 0.0);
    float effective_skylight = entity_skylight * sky_intensity;
    float sun_factor = smoothstep(0.4, 0.95, entity_skylight) * sky_intensity;
    float direct_sun = lambert * 0.55 * sun_factor;
    float ambient_sky = effective_skylight * 0.70;

    vec3 sky_color = sun_color * (ambient_sky + direct_sun);
    vec3 total_light = vec3(0.04) + sky_color + entity_blocklight_rgb + vec3(0.40, 0.15, 0.70) * (entity_blocklight_uv * 0.35) + vec3(0.65, 0.25, 0.08) * (entity_blocklight_ir * 0.35);

    vec3 final_color;
    if (vision_mode == 0) {
        vec3 lit = texel.rgb * total_light;
        if (hurt_flash > 0.0) {
            lit = mix(lit, vec3(0.85, 0.1, 0.1), 0.65);
        }
        float fog = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0) * use_fog;
        final_color = mix(lit, fog_color, fog);
    } else if (vision_mode == 1) {
        float uv_val = entity_blocklight_uv;
        vec3 uv_color = vec3(0.35, 0.1, 0.85) * uv_val * 1.5 + vec3(0.1, 0.85, 0.75) * pow(uv_val, 2.0);
        vec3 dim_scene = texel.rgb * 0.15 * (entity_skylight * 0.5 + 0.1);
        final_color = dim_scene + uv_color;
    } else if (vision_mode == 2) {
        float ir_val = entity_blocklight_ir;
        vec3 heat_color;
        if (ir_val < 0.25) {
            heat_color = mix(vec3(0.02, 0.02, 0.15), vec3(0.2, 0.0, 0.4), ir_val / 0.25);
        } else if (ir_val < 0.5) {
            heat_color = mix(vec3(0.2, 0.0, 0.4), vec3(0.85, 0.1, 0.1), (ir_val - 0.25) / 0.25);
        } else if (ir_val < 0.75) {
            heat_color = mix(vec3(0.85, 0.1, 0.1), vec3(1.0, 0.7, 0.1), (ir_val - 0.5) / 0.25);
        } else {
            heat_color = mix(vec3(1.0, 0.7, 0.1), vec3(1.0, 1.0, 1.0), (ir_val - 0.75) / 0.25);
        }
        float relief = 0.8 + 0.2 * lambert;
        final_color = heat_color * relief;
    } else {
        vec3 sky_viz = vec3(entity_skylight);
        vec3 block_viz = entity_blocklight_rgb;
        vec3 uv_viz = vec3(0.5, 0.0, 0.8) * entity_blocklight_uv;
        vec3 ir_viz = vec3(1.0, 0.4, 0.0) * entity_blocklight_ir;
        vec3 debug_color = sky_viz * 0.35 + block_viz * 0.35 + uv_viz * 0.35 + ir_viz * 0.35;
        final_color = mix(texel.rgb * 0.3, debug_color, 0.75);
    }

    fragColor = vec4(final_color, 1.0);
}
"""

CRACK_VERT = """
#version 330 core
uniform mat4 vp;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vp * vec4(in_pos, 1.0);
    v_uv = in_uv;
}
"""

CRACK_FRAG = """
#version 330 core
uniform sampler2D crack_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 texel = texture(crack_tex, v_uv);
    if (texel.a < 0.1) discard;
    fragColor = texel;
}
"""

CELESTIAL_VERT = """
#version 330 core
uniform mat4 vp;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vp * vec4(in_pos, 1.0);
    v_uv = in_uv;
}
"""

CELESTIAL_FRAG = """
#version 330 core
uniform sampler2D tex;
uniform vec4 tint;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, v_uv);
    fragColor = vec4(c.rgb * tint.rgb, c.a * tint.a);
}
"""

STAR_VERT = """
#version 330 core
uniform mat4 vp;
uniform vec3 camera;
uniform mat4 rot;
in vec3 in_pos;
in float in_brightness;
out float v_brightness;
void main() {
    vec4 rotated = rot * vec4(in_pos, 0.0);
    gl_Position = vp * vec4(camera + rotated.xyz, 1.0);
    v_brightness = in_brightness;
}
"""

STAR_FRAG = """
#version 330 core
uniform vec4 star_tint;
in float v_brightness;
out vec4 fragColor;
void main() {
    fragColor = vec4(star_tint.rgb, star_tint.a * v_brightness);
}
"""


def _celestial_quad(center, dir_vec, size):
    d = np.array(dir_vec, dtype=np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-6)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(d[1])) > 0.99:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(up, d)
    u /= max(float(np.linalg.norm(u)), 1e-6)
    v = np.cross(d, u)
    v /= max(float(np.linalg.norm(v)), 1e-6)

    c = np.array(center, dtype=np.float32) + d * 90.0
    s = float(size)
    p0 = c - u * s - v * s
    p1 = c + u * s - v * s
    p2 = c + u * s + v * s
    p3 = c - u * s + v * s

    return np.array([
        p0[0], p0[1], p0[2], 0.0, 0.0,
        p1[0], p1[1], p1[2], 1.0, 0.0,
        p2[0], p2[1], p2[2], 1.0, 1.0,
        p0[0], p0[1], p0[2], 0.0, 0.0,
        p2[0], p2[1], p2[2], 1.0, 1.0,
        p3[0], p3[1], p3[2], 0.0, 1.0,
    ], dtype=np.float32)


class _ChunkDraw:
    def __init__(self, opaque, cutout, translucent, origin):
        self.opaque = opaque
        self.cutout = cutout
        self.translucent = translucent
        self.origin = origin


class Renderer:
    def __init__(self, ctx, atlas_bytes, atlas_size, item_bytes, item_size):
        self.ctx = ctx
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.chunks = {}
        self.world = ctx.program(vertex_shader=WORLD_VERT, fragment_shader=WORLD_FRAG)
        self.translucent = ctx.program(vertex_shader=WORLD_VERT, fragment_shader=TRANSLUCENT_FRAG)
        self.textured = ctx.program(vertex_shader=TEXTURE_VERT, fragment_shader=TEXTURE_FRAG)
        self.colored = ctx.program(vertex_shader=COLOR_VERT, fragment_shader=COLOR_FRAG)
        self.entity = ctx.program(vertex_shader=ENTITY_VERT, fragment_shader=ENTITY_FRAG)
        self.atlas = ctx.texture(atlas_size, 4, atlas_bytes)
        self.atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.atlas.repeat_x = False
        self.atlas.repeat_y = False
        self.items = ctx.texture(item_size, 4, item_bytes)
        self.items.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.items.repeat_x = False
        self.items.repeat_y = False
        self.skin = None
        self.skin_key = None
        self.player_skins = {}
        self.player_meshes = PlayerMeshes()
        from server.pig_model import PigMeshes
        from server.sheep_model import SheepMeshes

        self.pig_meshes = PigMeshes()
        self.pig_texture = None
        self.sheep_meshes = SheepMeshes()
        self.sheep_texture = None
        self._init_mob_textures()
        self.world["atlas"].value = 0
        self.translucent["atlas"].value = 0
        self.textured["atlas"].value = 0
        self.entity["atlas"].value = 0
        self.world["fog_color"].value = SKY
        self.world["fog_near"].value = FOG_NEAR
        self.world["fog_far"].value = FOG_FAR
        self.translucent["fog_color"].value = SKY
        self.translucent["fog_near"].value = FOG_NEAR
        self.translucent["fog_far"].value = FOG_FAR
        self.entity["fog_color"].value = SKY
        self.entity["fog_near"].value = FOG_NEAR
        self.entity["fog_far"].value = FOG_FAR

        self.vision_mode = 0
        if "sun_dir" in self.world:
            self.world["sun_dir"].value = (0.35, 0.88, 0.28)
        if "vision_mode" in self.world:
            self.world["vision_mode"].value = 0
        if "sun_dir" in self.translucent:
            self.translucent["sun_dir"].value = (0.35, 0.88, 0.28)
        if "vision_mode" in self.translucent:
            self.translucent["vision_mode"].value = 0
        if "sun_dir" in self.entity:
            self.entity["sun_dir"].value = (0.35, 0.88, 0.28)
        if "vision_mode" in self.entity:
            self.entity["vision_mode"].value = 0
        if "entity_skylight" in self.entity:
            self.entity["entity_skylight"].value = 1.0
        if "entity_blocklight_rgb" in self.entity:
            self.entity["entity_blocklight_rgb"].value = (0.0, 0.0, 0.0)
        if "entity_blocklight_uv" in self.entity:
            self.entity["entity_blocklight_uv"].value = 0.0
        if "entity_blocklight_ir" in self.entity:
            self.entity["entity_blocklight_ir"].value = 0.0

        for prog in (self.world, self.translucent, self.entity):
            if "sun_color" in prog:
                prog["sun_color"].value = (1.0, 0.98, 0.94)
            if "sky_intensity" in prog:
                prog["sky_intensity"].value = 1.0

        self.celestial = ctx.program(vertex_shader=CELESTIAL_VERT, fragment_shader=CELESTIAL_FRAG)
        self.celestial["tex"].value = 0
        self.celestial_vbo = self.ctx.buffer(reserve=30 * 4)
        self.celestial_vao = self.ctx.vertex_array(self.celestial, [(self.celestial_vbo, "3f 2f", "in_pos", "in_uv")])

        self.star_prog = ctx.program(vertex_shader=STAR_VERT, fragment_shader=STAR_FRAG)
        rng = np.random.RandomState(42)
        star_verts = []
        for _ in range(500):
            # Uniform point distribution over full celestial sphere
            z = rng.uniform(-1.0, 1.0)
            theta = rng.uniform(0.0, 2.0 * math.pi)
            r_xy = math.sqrt(max(0.0, 1.0 - z * z))
            dx = r_xy * math.cos(theta)
            dy = z
            dz = r_xy * math.sin(theta)
            r = 95.0
            d = np.array([dx, dy, dz], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if abs(dy) > 0.99:
                up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            u = np.cross(up, d)
            u /= max(float(np.linalg.norm(u)), 1e-6)
            v = np.cross(d, u)
            v /= max(float(np.linalg.norm(v)), 1e-6)
            sz = rng.uniform(0.30, 0.70)
            bright = rng.uniform(0.5, 1.0)
            c = d * r
            p0 = c - u * sz - v * sz
            p1 = c + u * sz - v * sz
            p2 = c + u * sz + v * sz
            p3 = c - u * sz + v * sz
            for p in (p0, p1, p2, p0, p2, p3):
                star_verts.extend([p[0], p[1], p[2], bright])
        star_data = np.array(star_verts, dtype=np.float32)
        self.star_vbo = self.ctx.buffer(star_data.tobytes())
        self.star_vao = self.ctx.vertex_array(self.star_prog, [(self.star_vbo, "3f 1f", "in_pos", "in_brightness")])

        sun_path = ASSETS / "textures" / "sun.png"
        if sun_path.is_file():
            sun_img = read_png(sun_path)
            self.sun_tex = ctx.texture((sun_img.shape[1], sun_img.shape[0]), 4, np.ascontiguousarray(sun_img).tobytes())
            self.sun_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.sun_tex = None

        moon_path = ASSETS / "textures" / "moon.png"
        if moon_path.is_file():
            moon_img = read_png(moon_path)
            self.moon_tex = ctx.texture((moon_img.shape[1], moon_img.shape[0]), 4, np.ascontiguousarray(moon_img).tobytes())
            self.moon_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.moon_tex = None

        self.crack = ctx.program(vertex_shader=CRACK_VERT, fragment_shader=CRACK_FRAG)
        self.crack_textures = []
        for i in range(10):
            crack_file = ASSETS / "textures" / "crack" / f"crack_{i}.png"
            if crack_file.is_file():
                img = read_png(crack_file)
            else:
                img = np.zeros((16, 16, 4), dtype=np.uint8)
            tex = ctx.texture((img.shape[1], img.shape[0]), 4, np.ascontiguousarray(img).tobytes())
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.crack_textures.append(tex)
        self.crack["crack_tex"].value = 0

    def set_chunk(self, key, mesh, origin):
        self.drop(key)
        if mesh is None:
            return
        opaque = self._buffer(
            self.world,
            mesh.vertices,
            mesh.indices,
            getattr(mesh, "lights", None),
            "3f 3f 2f 1f",
            ("in_pos", "in_normal", "in_uv", "in_shade"),
            colors=getattr(mesh, "colors", None),
        )
        cutout = None
        if mesh.cutout is not None:
            cutout = self._buffer(
                self.world,
                mesh.cutout[0],
                mesh.cutout[1],
                getattr(mesh, "cutout_lights", None),
                "3f 3f 2f 1f",
                ("in_pos", "in_normal", "in_uv", "in_shade"),
                colors=getattr(mesh, "cutout_colors", None),
            )
        translucent = None
        if getattr(mesh, "translucent", None) is not None:
            translucent = self._buffer(
                self.translucent,
                mesh.translucent[0],
                mesh.translucent[1],
                getattr(mesh, "translucent_lights", None),
                "3f 3f 2f 1f",
                ("in_pos", "in_normal", "in_uv", "in_shade"),
                colors=getattr(mesh, "translucent_colors", None),
            )
        if opaque is None and cutout is None and translucent is None:
            return
        self.chunks[key] = _ChunkDraw(opaque, cutout, translucent, origin)

    def _buffer(self, program, vertices, indices, lights, layout, names, colors=None):
        if len(indices) == 0:
            return None
        vbo = self.ctx.buffer(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(indices, dtype=np.uint32).tobytes())
        if lights is None or len(lights) != len(vertices):
            lights = np.zeros((len(vertices), 6), dtype=np.float32)
            lights[:, 0] = 1.0  # default full sunlight
        vbo_light = self.ctx.buffer(np.ascontiguousarray(lights, dtype=np.float32).tobytes())
        if colors is None or len(colors) != len(vertices):
            colors = np.ones((len(vertices), 3), dtype=np.float32)
        vbo_color = self.ctx.buffer(np.ascontiguousarray(colors, dtype=np.float32).tobytes())
        vao = self.ctx.vertex_array(
            program,
            [
                (vbo, layout, *names),
                (vbo_light, "1f 3f 1f 1f", "in_skylight", "in_blocklight_rgb", "in_blocklight_uv", "in_blocklight_ir"),
                (vbo_color, "3f", "in_color"),
            ],
            ibo,
        )
        return vao, vbo, ibo, vbo_light, vbo_color

    def drop(self, key):
        item = self.chunks.pop(key, None)
        if item is None:
            return
        for part in (item.opaque, item.cutout, item.translucent):
            if part is None:
                continue
            for buf in part:
                if buf is not None and hasattr(buf, "release"):
                    buf.release()

    def get_sky_intensity(self, day_time=1000.0):
        t = (day_time / 24000.0) % 1.0
        alpha = 2.0 * math.pi * (t - 0.25)
        sun_height = math.cos(alpha)
        transition = max(0.0, min(1.0, (sun_height + 0.15) / 0.30))
        s = transition * transition * (3.0 - 2.0 * transition)
        return (7.0 + 8.0 * s) / 15.0

    def get_celestial_lighting(self, day_time=1000.0):
        t = (day_time / 24000.0) % 1.0
        alpha = 2.0 * math.pi * (t - 0.25)
        sun_height = math.cos(alpha)
        sun_x = -math.sin(alpha)
        sun_y = sun_height
        sun_z = -0.2
        length = math.hypot(sun_x, sun_y, sun_z)
        sun_dir = (sun_x / length, sun_y / length, sun_z / length)

        transition = max(0.0, min(1.0, (sun_height + 0.15) / 0.30))
        s = transition * transition * (3.0 - 2.0 * transition)

        sunset_factor = max(0.0, 1.0 - abs(sun_height) / 0.22)
        sunset_w = sunset_factor * sunset_factor * (3.0 - 2.0 * sunset_factor)

        # Day: warm sunlight (1.0, 0.98, 0.94)
        # Night: cool moonlight (0.55, 0.65, 0.85)
        # Sunset: golden-orange (1.0, 0.55, 0.30)
        sr = 0.55 + (1.0 - 0.55) * s
        sg = 0.65 + (0.98 - 0.65) * s
        sb = 0.85 + (0.94 - 0.85) * s

        sr = sr * (1.0 - sunset_w * 0.6) + 1.0 * sunset_w * 0.6
        sg = sg * (1.0 - sunset_w * 0.6) + 0.55 * sunset_w * 0.6
        sb = sb * (1.0 - sunset_w * 0.6) + 0.30 * sunset_w * 0.6

        return sun_dir, (sr, sg, sb)

    def get_sky_color(self, day_time=1000.0):
        t = (day_time / 24000.0) % 1.0
        alpha = 2.0 * math.pi * (t - 0.25)
        sun_height = math.cos(alpha)

        transition = max(0.0, min(1.0, (sun_height + 0.15) / 0.30))
        s = transition * transition * (3.0 - 2.0 * transition)

        sunset_factor = max(0.0, 1.0 - abs(sun_height) / 0.22)
        sunset_w = sunset_factor * sunset_factor * (3.0 - 2.0 * sunset_factor)

        # Day: (0.56, 0.78, 0.96)
        # Night: (0.05, 0.06, 0.12)
        # Sunset: (0.82, 0.45, 0.28)
        r = 0.05 + (0.56 - 0.05) * s
        g = 0.06 + (0.78 - 0.06) * s
        b = 0.12 + (0.96 - 0.12) * s

        r = r * (1.0 - sunset_w * 0.7) + 0.82 * sunset_w * 0.7
        g = g * (1.0 - sunset_w * 0.7) + 0.45 * sunset_w * 0.7
        b = b * (1.0 - sunset_w * 0.7) + 0.28 * sunset_w * 0.7

        return (r, g, b)

    def draw_sky(self, vp, camera, day_time=1000.0):
        t = (day_time / 24000.0) % 1.0
        alpha = 2.0 * math.pi * (t - 0.25)
        sun_height = math.cos(alpha)
        transition = max(0.0, min(1.0, (sun_height + 0.15) / 0.30))
        s = transition * transition * (3.0 - 2.0 * transition)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # 1. Stars (visible at night and twilight)
        star_alpha = max(0.0, 1.0 - s * 2.2)
        if star_alpha > 0.01 and getattr(self, "star_vao", None) is not None:
            write_matrix(self.star_prog["vp"], vp)
            self.star_prog["camera"].value = tuple(float(c) for c in camera)
            rot4 = np.eye(4, dtype=np.float32)
            rot4[0, 0] = math.cos(alpha)
            rot4[0, 1] = -math.sin(alpha)
            rot4[1, 0] = math.sin(alpha)
            rot4[1, 1] = math.cos(alpha)
            write_matrix(self.star_prog["rot"], rot4)
            self.star_prog["star_tint"].value = (1.0, 1.0, 1.0, float(star_alpha))
            self.star_vao.render(moderngl.TRIANGLES)

        # 2. Celestial bodies (Sun & Moon)
        sun_dir, _ = self.get_celestial_lighting(day_time)
        moon_dir = (-sun_dir[0], -sun_dir[1], -sun_dir[2])

        # Draw Sun
        sun_alpha = max(0.0, min(1.0, (sun_dir[1] + 0.15) / 0.15))
        if sun_alpha > 0.01 and getattr(self, "sun_tex", None) is not None and getattr(self, "celestial_vbo", None) is not None:
            write_matrix(self.celestial["vp"], vp)
            quad = _celestial_quad(camera, sun_dir, 14.0)
            self.celestial_vbo.write(quad.tobytes())
            self.celestial["tint"].value = (1.0, 1.0, 1.0, float(sun_alpha))
            self.sun_tex.use(0)
            self.celestial_vao.render(moderngl.TRIANGLES)

        # Draw Moon
        moon_alpha = max(0.0, min(1.0, (moon_dir[1] + 0.15) / 0.15))
        if moon_alpha > 0.01 and getattr(self, "moon_tex", None) is not None and getattr(self, "celestial_vbo", None) is not None:
            write_matrix(self.celestial["vp"], vp)
            quad = _celestial_quad(camera, moon_dir, 11.0)
            self.celestial_vbo.write(quad.tobytes())
            m_fade = float(moon_alpha * max(0.35, 1.0 - 0.6 * s))
            self.celestial["tint"].value = (0.95, 0.95, 1.0, m_fade)
            self.moon_tex.use(0)
            self.celestial_vao.render(moderngl.TRIANGLES)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def draw_world(self, vp, camera, cutout=True, day_time=1000.0):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        write_matrix(self.world["vp"], vp)
        self.world["camera"].value = camera
        sky_intensity = self.get_sky_intensity(day_time)
        sun_dir, sun_color = self.get_celestial_lighting(day_time)
        sky_color = self.get_sky_color(day_time)
        self.world["fog_color"].value = sky_color
        if "sun_dir" in self.world:
            self.world["sun_dir"].value = sun_dir
        if "sun_color" in self.world:
            self.world["sun_color"].value = sun_color
        if "sky_intensity" in self.world:
            self.world["sky_intensity"].value = float(sky_intensity)
        if "vision_mode" in self.world:
            self.world["vision_mode"].value = self.vision_mode
        self.atlas.use(0)
        for item in self.chunks.values():
            if item.opaque is None:
                continue
            self.world["origin"].value = item.origin
            item.opaque[0].render()
        if cutout:
            self.draw_cutout(day_time=day_time)

    def draw_cutout(self, day_time=1000.0):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        sky_intensity = self.get_sky_intensity(day_time)
        sun_dir, sun_color = self.get_celestial_lighting(day_time)
        sky_color = self.get_sky_color(day_time)
        self.world["fog_color"].value = sky_color
        if "sun_dir" in self.world:
            self.world["sun_dir"].value = sun_dir
        if "sun_color" in self.world:
            self.world["sun_color"].value = sun_color
        if "sky_intensity" in self.world:
            self.world["sky_intensity"].value = float(sky_intensity)
        if "vision_mode" in self.world:
            self.world["vision_mode"].value = self.vision_mode
        self.atlas.use(0)
        for item in self.chunks.values():
            if item.cutout is None:
                continue
            self.world["origin"].value = item.origin
            item.cutout[0].render()
        self.ctx.enable(moderngl.CULL_FACE)

    def draw_translucent(self, vp, camera, day_time=1000.0):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.screen.depth_mask = True
        write_matrix(self.translucent["vp"], vp)
        self.translucent["camera"].value = camera
        sky_intensity = self.get_sky_intensity(day_time)
        sun_dir, sun_color = self.get_celestial_lighting(day_time)
        sky_color = self.get_sky_color(day_time)
        self.translucent["fog_color"].value = sky_color
        if "sun_dir" in self.translucent:
            self.translucent["sun_dir"].value = sun_dir
        if "sun_color" in self.translucent:
            self.translucent["sun_color"].value = sun_color
        if "sky_intensity" in self.translucent:
            self.translucent["sky_intensity"].value = float(sky_intensity)
        if "vision_mode" in self.translucent:
            self.translucent["vision_mode"].value = self.vision_mode
        self.atlas.use(0)
        for item in self.chunks.values():
            if item.translucent is None:
                continue
            self.translucent["origin"].value = item.origin
            item.translucent[0].render()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)

    def set_skin(self, skin_rgba, player_id=None):
        if skin_rgba is None:
            return
        image = np.flipud(np.ascontiguousarray(skin_rgba, dtype=np.uint8))
        key = hash(image.tobytes())
        if player_id is None:
            if key == self.skin_key and self.skin is not None:
                return
            if self.skin is not None:
                self.skin.release()
            self.skin = self._create_skin_texture(image)
            self.skin_key = key
        else:
            uuid_str = str(player_id)
            prev = self.player_skins.get(uuid_str)
            if prev is not None and prev[0] == key:
                return
            if prev is not None and prev[1] is not None:
                prev[1].release()
            tex = self._create_skin_texture(image)
            self.player_skins[uuid_str] = (key, tex)

    def _create_skin_texture(self, image):
        tex = self.ctx.texture((image.shape[1], image.shape[0]), 4, image.tobytes())
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def get_player_skin(self, player_id=None):
        if player_id is None:
            return self.skin
        entry = self.player_skins.get(str(player_id))
        if entry is not None:
            return entry[1]
        return self.skin

    def draw_player(self, vp, camera, player, hotbar, selected, view_mode, time, day_time=1000.0):
        if player is None or self.skin is None:
            return
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        item_id = selected_item_id(hotbar, selected)
        if view_mode == 0:
            # Depth-only clear so the viewmodel never Z-fights world terrain.
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.CULL_FACE)
            poses = first_person_pose(player, time)
            # Minecraft FP: arm only when empty-handed; held items hide the arm.
            if not item_id:
                vertices, indices = self.player_meshes.first_person_arm(
                    camera, player.yaw, player.pitch, poses
                )
                self._draw_entity(vp, camera, vertices, indices, self.skin, fog=False, day_time=day_time)
            else:
                self._draw_held(vp, camera, player, poses, item_id, first_person=True, eye=camera, fog=False, day_time=day_time)
            return
        poses = third_person_pose(player, time)
        vertices, indices = self.player_meshes.posed_vertices(player.pos, player.yaw, poses)
        self._draw_entity(vp, camera, vertices, indices, self.skin, fog=True, day_time=day_time)
        self._draw_held(vp, camera, player, poses, item_id, first_person=False, eye=None, fog=True, day_time=day_time)

    def draw_fallen_block_overlay(self, width, height, block_id, tiles=4):
        """Fill the viewport with the texture of the block the player is inside.

        When a block that has fallen ends up at the player's head height, its
        texture takes over the first-person view so the player cannot see through
        the world (world chunk meshes do not draw faces buried inside solid
        blocks). Drawn screen-space at the end of the world pass with depth
        testing disabled, so it always covers and can never lose a depth fight —
        unlike a 3D cube at the eye, whose faces get back-face culled when the
        camera is inside it. The HUD and GUI render after this pass and are
        unaffected.
        """
        if block_id not in BLOCKS:
            return
        block = BLOCKS[block_id]
        # Prefer a side face so the overlay reads like being surrounded by the block.
        face_uvs = None
        for face in ("north", "east", "south", "west", "up", "down"):
            idx = FACE_INDEX[face]
            if idx < len(block.face_uvs) and block.face_uvs[idx] is not None:
                face_uvs = block.face_uvs[idx]
                break
        if face_uvs is None:
            return
        us = [float(face_uvs[i][0]) for i in range(len(face_uvs))]
        vs = [float(face_uvs[i][1]) for i in range(len(face_uvs))]
        u0, u1 = min(us), max(us)
        v0, v1 = min(vs), max(vs)
        floats = []
        step_x = width / tiles
        step_y = height / tiles
        for ty in range(tiles):
            for tx in range(tiles):
                floats.extend(
                    _quad_uv(
                        tx * step_x,
                        ty * step_y,
                        (tx + 1) * step_x,
                        (ty + 1) * step_y,
                        u0,
                        v0,
                        u1,
                        v1,
                    )
                )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._textured(ortho(width, height), floats, self.atlas)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_other_players(self, vp, camera, other_players, time, day_time=1000.0):
        if not other_players:
            return
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        for player in other_players.values():
            skin = self.get_player_skin(player.uuid)
            if skin is None:
                continue
            poses = third_person_pose(player, time)
            vertices, indices = self.player_meshes.posed_vertices(player.pos, player.yaw, poses)
            self._draw_entity(vp, camera, vertices, indices, skin, fog=True, day_time=day_time)
            held = getattr(player, "held_item", "")
            if held:
                self._draw_held(vp, camera, player, poses, held, first_person=False, eye=None, fog=True, day_time=day_time)

    def _draw_held(self, vp, camera, player, poses, item_id, first_person, eye, fog, day_time=1000.0):
        matrix = item_hold_matrix(player, poses, first_person=first_person, eye=eye, item_id=item_id)
        packed = held_item_mesh(item_id, matrix)
        if packed is None:
            return
        (vertices, indices), kind = packed
        texture = self.atlas if kind == "block" else self.items
        if first_person:
            # Depth already cleared for this pass; disable test so overlapping
            # extrusion pads don't moiré and near terrain can't poke through.
            self.ctx.disable(moderngl.DEPTH_TEST)
        self._draw_entity(vp, camera, vertices, indices, texture, fog=fog, day_time=day_time)
        if first_person:
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _init_mob_textures(self):
        pig_path = ASSETS / "textures" / "entity" / "pig.png"
        if not pig_path.is_file():
            pig_path = ASSETS / "textures" / "pig.png"
        if pig_path.is_file():
            try:
                rgba = read_png(pig_path)
                image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
                self.pig_texture = self._create_skin_texture(image)
            except Exception:
                self.pig_texture = None

        sheep_path = ASSETS / "textures" / "entity" / "sheep.png"
        if not sheep_path.is_file():
            sheep_path = ASSETS / "textures" / "sheep.png"
        if sheep_path.is_file():
            try:
                rgba = read_png(sheep_path)
                image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
                self.sheep_texture = self._create_skin_texture(image)
            except Exception:
                self.sheep_texture = None

    def draw_item_entities(self, vp, camera, entities, time_val, day_time=1000.0):
        if not entities:
            return
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        for entity in entities:
            if getattr(entity, "count", 1) <= 0 or getattr(entity, "despawn_timer", 1) <= 0:
                continue
            if getattr(entity, "is_mob", False):
                if not getattr(entity, "alive", True):
                    continue
                mob_type = getattr(entity, "mob_type", "")
                if mob_type == "pig" and self.pig_texture is not None:
                    from server.pig_model import pig_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    poses = pig_pose(entity.walk_time, hspeed, entity.yaw, pitch=entity.pitch)
                    vertices, indices = self.pig_meshes.posed_vertices(entity.pos, entity.yaw, poses)
                    hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
                    self._draw_entity(
                        vp,
                        camera,
                        vertices,
                        indices,
                        self.pig_texture,
                        fog=True,
                        day_time=day_time,
                        hurt_flash=hurt,
                    )
                elif mob_type == "sheep" and self.sheep_texture is not None:
                    from server.sheep_model import sheep_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    poses = sheep_pose(entity.walk_time, hspeed, entity.yaw, pitch=entity.pitch)
                    sheared = bool(getattr(entity, "sheared", False))
                    vertices, indices = self.sheep_meshes.posed_vertices(
                        entity.pos, entity.yaw, poses, sheared=sheared
                    )
                    hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
                    self._draw_entity(
                        vp,
                        camera,
                        vertices,
                        indices,
                        self.sheep_texture,
                        fog=True,
                        day_time=day_time,
                        hurt_flash=hurt,
                    )
                continue
            if getattr(entity, "is_falling_block", False) or hasattr(entity, "block_id"):
                mesh = falling_block_mesh(entity.block_id, entity.pos)
                if mesh is not None:
                    vertices, indices = mesh
                    self._draw_entity(vp, camera, vertices, indices, self.atlas, fog=True, day_time=day_time)
                continue
            matrix = dropped_item_matrix(entity.pos, getattr(entity, "age", 0.0))
            packed = held_item_mesh(entity.item, matrix)
            if packed is None:
                continue
            (vertices, indices), kind = packed
            texture = self.atlas if kind == "block" else self.items
            self._draw_entity(vp, camera, vertices, indices, texture, fog=True, day_time=day_time)

    def _draw_entity(self, vp, camera, vertices, indices, texture, fog, skylight=1.0, blocklight=(0, 0, 0, 0, 0), day_time=1000.0, hurt_flash=0.0):
        if len(indices) == 0:
            return
        write_matrix(self.entity["vp"], vp)
        self.entity["camera"].value = tuple(float(v) for v in camera)
        self.entity["use_fog"].value = 1.0 if fog else 0.0
        if "hurt_flash" in self.entity:
            self.entity["hurt_flash"].value = float(hurt_flash)
        sky_intensity = self.get_sky_intensity(day_time)
        sun_dir, sun_color = self.get_celestial_lighting(day_time)
        sky_color = self.get_sky_color(day_time)
        self.entity["fog_color"].value = sky_color
        if "sun_dir" in self.entity:
            self.entity["sun_dir"].value = sun_dir
        if "sun_color" in self.entity:
            self.entity["sun_color"].value = sun_color
        if "sky_intensity" in self.entity:
            self.entity["sky_intensity"].value = float(sky_intensity)
        if "vision_mode" in self.entity:
            self.entity["vision_mode"].value = self.vision_mode
        if "entity_skylight" in self.entity:
            self.entity["entity_skylight"].value = float(skylight)
        if "entity_blocklight_rgb" in self.entity:
            self.entity["entity_blocklight_rgb"].value = (float(blocklight[0]) / 15.0, float(blocklight[1]) / 15.0, float(blocklight[2]) / 15.0)
        if "entity_blocklight_uv" in self.entity:
            self.entity["entity_blocklight_uv"].value = float(blocklight[3]) / 15.0
        if "entity_blocklight_ir" in self.entity:
            self.entity["entity_blocklight_ir"].value = float(blocklight[4]) / 15.0
        texture.use(0)
        vbo = self.ctx.buffer(np.ascontiguousarray(vertices).tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(indices).tobytes())
        vao = self.ctx.vertex_array(self.entity, [(vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")], ibo)
        vao.render()
        vao.release()
        vbo.release()
        ibo.release()

    def draw_outline(self, vp, block, box=None):
        if block is None and box is None:
            return
        pad = 0.004
        if box is not None:
            bx0, by0, bz0, bx1, by1, bz1 = box
            x0, y0, z0 = bx0 - pad, by0 - pad, bz0 - pad
            x1, y1, z1 = bx1 + pad, by1 + pad, bz1 + pad
        else:
            x, y, z = block
            x0, y0, z0 = x - pad, y - pad, z - pad
            x1, y1, z1 = x + 1 + pad, y + 1 + pad, z + 1 + pad
        corners = (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        data = []
        for a, b in edges:
            data.extend(corners[a])
            data.extend(corners[b])
        self._lines(vp, data, (0.05, 0.05, 0.05, 1.0))

    def draw_crack(self, vp, block, stage, box=None):
        if (block is None and box is None) or stage < 0 or stage >= len(self.crack_textures):
            return
        pad = 0.004
        if box is not None:
            bx0, by0, bz0, bx1, by1, bz1 = box
            ox = bx0 - pad
            oy = by0 - pad
            oz = bz0 - pad
            dx_total = (bx1 - bx0) + 2.0 * pad
            dy_total = (by1 - by0) + 2.0 * pad
            dz_total = (bz1 - bz0) + 2.0 * pad
        else:
            x, y, z = block
            ox = x - pad
            oy = y - pad
            oz = z - pad
            dx_total = 1.0 + 2.0 * pad
            dy_total = 1.0 + 2.0 * pad
            dz_total = 1.0 + 2.0 * pad

        verts = []
        for face_idx in range(6):
            c = CORNERS[face_idx]
            p0 = [ox + c[0][0] * dx_total, oy + c[0][1] * dy_total, oz + c[0][2] * dz_total]
            p1 = [ox + c[1][0] * dx_total, oy + c[1][1] * dy_total, oz + c[1][2] * dz_total]
            p2 = [ox + c[2][0] * dx_total, oy + c[2][1] * dy_total, oz + c[2][2] * dz_total]
            p3 = [ox + c[3][0] * dx_total, oy + c[3][1] * dy_total, oz + c[3][2] * dz_total]
            verts.extend([p0[0], p0[1], p0[2], 0.0, 0.0])
            verts.extend([p1[0], p1[1], p1[2], 1.0, 0.0])
            verts.extend([p2[0], p2[1], p2[2], 1.0, 1.0])
            verts.extend([p0[0], p0[1], p0[2], 0.0, 0.0])
            verts.extend([p2[0], p2[1], p2[2], 1.0, 1.0])
            verts.extend([p3[0], p3[1], p3[2], 0.0, 1.0])

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.crack_textures[stage].use(0)
        write_matrix(self.crack["vp"], vp)
        vbo = self.ctx.buffer(np.array(verts, dtype="f4").tobytes())
        vao = self.ctx.vertex_array(self.crack, [(vbo, "3f 2f", "in_pos", "in_uv")])
        vao.render()
        vao.release()
        vbo.release()
        self.ctx.enable(moderngl.CULL_FACE)

    def draw_hud(self, width, height, hotbar, selected, crosshair, break_progress=0.0, health=None, max_health=20.0):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        projection = ortho(width, height)
        self._hotbar(projection, width, height, hotbar, selected)
        if health is not None:
            self._hearts(projection, width, height, health, max_health)
        if crosshair:
            self._crosshair(projection, width, height)
            if break_progress > 0.0:
                self._break_progress_indicator(projection, width, height, break_progress)

    def draw_hurt_flash(self, width, height, intensity):
        """Draw the translucent red overlay shown briefly after taking damage."""
        try:
            intensity = max(0.0, min(1.0, float(intensity)))
        except (TypeError, ValueError):
            return
        if intensity <= 0.0:
            return
        self.draw_rects(width, height, [(0, 0, width, height, (0.72, 0.0, 0.0, 0.42 * intensity))])

    def _hearts(self, projection, width, height, health, max_health):
        try:
            health = float(health)
            max_health = float(max_health)
        except (TypeError, ValueError):
            return
        if max_health <= 0.0:
            return
        hearts = max(1, int(math.ceil(max_health / 2.0)))
        slot = 50
        gap = 8
        total = HOTBAR_SLOTS * slot + (HOTBAR_SLOTS - 1) * gap
        x0 = (width - total) / 2
        y0 = 28 + slot + 10
        scale = 3
        spacing = 2
        heart_width = HEART_PATTERN_WIDTH * scale
        for index in range(hearts):
            hx = x0 + index * (heart_width + spacing)
            self._heart(projection, hx, y0, scale, (0.16, 0.04, 0.05, 0.9), 1.0)
            value = max(0.0, min(2.0, health - index * 2.0))
            if value <= 0.0:
                continue
            color = (0.88, 0.10, 0.12, 1.0) if value >= 2.0 else (0.92, 0.30, 0.18, 1.0)
            self._heart(projection, hx, y0, scale, color, value / 2.0)

    def _heart(self, projection, x, y, scale, color, fraction):
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction <= 0.0:
            return
        floats = []
        for row, line in enumerate(HEART_PATTERN):
            for col, pixel in enumerate(line):
                if pixel != "1":
                    continue
                if (col + 0.5) / HEART_PATTERN_WIDTH > fraction:
                    continue
                px = x + col * scale
                py = y + (HEART_PATTERN_HEIGHT - 1 - row) * scale
                floats.extend(_quad(px, py, px + scale, py + scale))
        if floats:
            self._tris(projection, floats, color)

    def _break_progress_indicator(self, projection, width, height, progress):
        cx = width / 2
        cy = height / 2 - 22
        bw = 28
        bh = 4
        # Dark outline box
        bg = _quad(cx - bw / 2 - 1, cy - 1, cx + bw / 2 + 1, cy + bh + 1)
        self._tris(projection, bg, (0.0, 0.0, 0.0, 0.75))
        # Inner bar
        fill_w = max(0.0, min(1.0, float(progress))) * bw
        if fill_w > 0:
            # Color shifts from reddish orange (start) to yellow to lime green (complete)
            p = max(0.0, min(1.0, float(progress)))
            r = min(1.0, 1.8 * (1.0 - p))
            g = min(1.0, 1.8 * p)
            fill = _quad(cx - bw / 2, cy, cx - bw / 2 + fill_w, cy + bh)
            self._tris(projection, fill, (r, g, 0.2, 0.95))

    def draw_icons(self, width, height, icons):
        if not icons:
            return
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        floats = []
        for x, y, w, h, u0, v0, u1, v1 in icons:
            floats.extend(_quad_uv(x, y, x + w, y + h, u0, v0, u1, v1))
        self._textured(ortho(width, height), floats, self.items)

    def draw_rects(self, width, height, rects):
        if not rects:
            return
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        projection = ortho(width, height)
        grouped = {}
        for x, y, w, h, color in rects:
            grouped.setdefault(color, []).extend(_quad(x, y, x + w, y + h))
        for color, floats in grouped.items():
            self._tris(projection, floats, color)

    def draw_panel(self, width, height, panel_width, panel_height):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        projection = ortho(width, height)
        x0 = (width - panel_width) / 2
        y0 = (height - panel_height) / 2
        self._tris(projection, _quad(x0, y0, x0 + panel_width, y0 + panel_height), (0.16, 0.18, 0.13, 0.9))

    def _hotbar(self, projection, width, height, hotbar, selected):
        slot = 50
        gap = 8
        total = HOTBAR_SLOTS * slot + (HOTBAR_SLOTS - 1) * gap
        x = (width - total) / 2
        y = 28
        self._draw_hotbar_passes(projection, hotbar, selected, slot, gap, x, y)

    def _draw_hotbar_passes(self, projection, hotbar, selected, slot, gap, x, y):
        highlight = []
        slots = []
        icons = []
        for index in range(HOTBAR_SLOTS):
            block_id = stack_id(hotbar[index]) if index < len(hotbar) else ""
            sx = x + index * (slot + gap)
            if index == selected:
                highlight.extend(_quad(sx - 4, y - 4, sx + slot + 4, y + slot + 4))
            slots.extend(_quad(sx, y, sx + slot, y + slot))
            item = ITEMS.get(block_id)
            if item is None:
                continue
            u0, v0, u1, v1 = item.icon
            icons.extend(_quad_uv(sx + 7, y + 7, sx + slot - 7, y + slot - 7, u0, v0, u1, v1))
        if highlight:
            self._tris(projection, highlight, (0.97, 0.97, 0.94, 1.0))
        self._tris(projection, slots, (0.12, 0.1, 0.08, 0.88))
        if icons:
            self._textured(projection, icons, self.items)

    def _crosshair(self, projection, width, height):
        cx = width / 2
        cy = height / 2
        arm = 12
        gap = 4
        white = [
            cx - arm, cy, 0, cx - gap, cy, 0,
            cx + gap, cy, 0, cx + arm, cy, 0,
            cx, cy - arm, 0, cx, cy - gap, 0,
            cx, cy + gap, 0, cx, cy + arm, 0,
        ]
        black = [
            cx - arm, cy - 1, 0, cx - gap, cy - 1, 0,
            cx + gap, cy - 1, 0, cx + arm, cy - 1, 0,
            cx - 1, cy - arm, 0, cx - 1, cy - gap, 0,
            cx - 1, cy + gap, 0, cx - 1, cy + arm, 0,
        ]
        self._lines(projection, black, (0.0, 0.0, 0.0, 0.85))
        self._lines(projection, white, (1.0, 1.0, 1.0, 1.0))

    def _tris(self, mvp, floats, color):
        self._primitive(self.colored, "3f", ("in_pos",), floats, moderngl.TRIANGLES, mvp, color)

    def _lines(self, mvp, floats, color):
        self._primitive(self.colored, "3f", ("in_pos",), floats, moderngl.LINES, mvp, color)

    def _textured(self, mvp, floats, texture=None):
        data = np.asarray(floats, dtype=np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.textured, [(vbo, "3f 2f", "in_pos", "in_uv")])
        write_matrix(self.textured["mvp"], mvp)
        (self.atlas if texture is None else texture).use(0)
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()

    def _primitive(self, program, fmt, names, floats, mode, mvp, color):
        data = np.asarray(floats, dtype=np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(program, [(vbo, fmt, *names)])
        write_matrix(program["mvp"], mvp)
        program["color"].value = color
        vao.render(mode)
        vao.release()
        vbo.release()


def _quad(x0, y0, x1, y1):
    return [
        x0, y0, 0, x1, y0, 0, x1, y1, 0,
        x0, y0, 0, x1, y1, 0, x0, y1, 0,
    ]


def _quad_uv(x0, y0, x1, y1, u0, v0, u1, v1):
    return [
        x0, y0, 0, u0, v0,
        x1, y0, 0, u1, v0,
        x1, y1, 0, u1, v1,
        x0, y0, 0, u0, v0,
        x1, y1, 0, u1, v1,
        x0, y1, 0, u0, v1,
    ]


def chunk_origin(key):
    return (key[0] * CHUNK_SIZE, key[1] * CHUNK_SIZE, key[2] * CHUNK_SIZE)
