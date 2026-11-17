import math
import moderngl
import numpy as np
from pyglet import gl

from server.blocks import ASSETS, BLOCKS, CHUNK_SIZE, FACE_INDEX, HOTBAR_SLOTS
from server.images import read_png
from server.items import ITEMS
from server.mesh import CORNERS
from server.stacks import stack_damage, stack_id

from client.geom import ortho, write_matrix
from client.player_render import (
    PlayerMeshes,
    arrow_flight_matrix,
    dropped_item_matrix,
    falling_block_mesh,
    first_person_pose,
    held_item_matrix,
    held_item_mesh,
    item_hold_matrix,
    nocked_arrow_matrix,
    third_person_nocked_arrow_matrix,
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

# A small pixel-art drumstick used for the survival hunger bar.
DRUMSTICK_PATTERN = (
    "0111000",
    "1111100",
    "1111100",
    "1111000",
    "0110000",
    "0001010",
    "0000111",
)
DRUMSTICK_PATTERN_WIDTH = len(DRUMSTICK_PATTERN[0])
DRUMSTICK_PATTERN_HEIGHT = len(DRUMSTICK_PATTERN)

WORLD_VERT = """
#version 330 core
uniform mat4 vp;
uniform vec3 origin;
uniform vec3 camera;
uniform float anim_time;
uniform vec2 atlas_size;
uniform vec4 fire_frames[4];
uniform vec4 blue_fire_frames[4];
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_shade;
in float in_skylight;
in vec3 in_blocklight_rgb;
in float in_blocklight_uv;
in float in_blocklight_ir;
in vec3 in_color;
in vec4 in_fluorescence;
in vec2 in_fluor_uv;

out vec2 v_uv;
out vec3 v_normal;
out float v_shade;
out float v_dist;
out float v_skylight;
out vec3 v_blocklight_rgb;
out float v_blocklight_uv;
out float v_blocklight_ir;
out vec3 v_color;
out vec4 v_fluorescence;
out vec2 v_fluor_uv;
out float v_emissive;

vec4 frame_rect(int kind, int frame) {
    if (kind == 2) {
        if (frame == 1) return blue_fire_frames[1];
        if (frame == 2) return blue_fire_frames[2];
        if (frame == 3) return blue_fire_frames[3];
        return blue_fire_frames[0];
    }
    if (frame == 1) return fire_frames[1];
    if (frame == 2) return fire_frames[2];
    if (frame == 3) return fire_frames[3];
    return fire_frames[0];
}

vec2 animated_uv() {
    int kind = int(in_color.g + 0.5);
    int frame = int(mod(floor(anim_time * 0.4), 4.0));
    vec4 base = kind == 2 ? blue_fire_frames[0] : fire_frames[0];
    vec4 rect = frame_rect(kind, frame);
    vec2 span = max(base.zw - base.xy, vec2(0.00001));
    vec2 local = clamp((in_uv - base.xy) / span, 0.0, 1.0);
    return mix(rect.xy, rect.zw, local);
}

void main() {
    vec3 world = in_pos + origin;
    gl_Position = vp * vec4(world, 1.0);
    v_emissive = 0.0;
    v_uv = in_uv;
    v_color = in_color;
    v_fluorescence = in_fluorescence;
    v_fluor_uv = in_fluor_uv;
    if (in_color.r > 1.5) {
        int kind = int(in_color.g + 0.5);
        v_color = vec3(1.0);
        if (kind == 1 || kind == 2) {
            v_uv = animated_uv();
            v_emissive = 1.0;
        } else {
            // Lava keeps its tile UVs. The fragment shader boils inside the
            // tile; wrapping here makes the four corners disagree and the
            // whole face stretches into a single texel.
            v_emissive = kind == 4 ? 3.0 : 2.0;
        }
    }
    v_normal = in_normal;
    v_shade = in_shade;
    v_dist = distance(world, camera);
    v_skylight = in_skylight;
    v_blocklight_rgb = in_blocklight_rgb;
    v_blocklight_uv = in_blocklight_uv;
    v_blocklight_ir = in_blocklight_ir;
}
"""

WORLD_FRAG = """
#version 330 core
uniform sampler2D atlas;
uniform vec2 atlas_size;
uniform float anim_time;
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
in vec4 v_fluorescence;
in vec2 v_fluor_uv;
in float v_emissive;

out vec4 fragColor;

// Matches server.lighting.fluorescence_response. The alpha channel packs
// start (0..15) + full * 16 + strength_sixteenths * 256.
float fluorescence_response(vec4 fluor, float uv_level) {
    float strength = floor(fluor.a / 256.0) / 16.0;
    if (strength <= 0.0) {
        return 0.0;
    }
    float start = mod(fluor.a, 16.0);
    float full_level = mod(floor(fluor.a / 16.0), 16.0);
    float end_level = max(full_level, start + 0.05);
    float t = clamp((uv_level - start) / (end_level - start), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t) * strength;
}

vec2 lava_uv(float dir) {
    vec2 px = v_uv * atlas_size - 0.5;
    vec2 origin = floor(px / 16.0) * 16.0;
    vec2 local = clamp(px - origin, 0.0, 15.0);
    float t = anim_time * 0.15 * dir;
    vec2 drift = vec2(sin(t), cos(t * 0.73)) * 1.5;
    vec2 boil = vec2(sin(local.y * 1.2 + t * 2.4), cos(local.x * 1.05 - t * 1.9)) * 0.8;
    local = clamp(local + drift + boil, 0.0, 15.0);
    return (origin + local + 0.5) / atlas_size;
}

void main() {
    vec2 uv = v_uv;
    if (v_emissive > 1.5) {
        uv = lava_uv(v_emissive > 2.5 ? -1.0 : 1.0);
    }
    vec4 texel = texture(atlas, uv);
    if (texel.a < 0.5) discard;

    vec3 base_rgb = texel.rgb;
    if (v_emissive > 0.5) {
        float fog = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
        fragColor = vec4(mix(base_rgb, fog_color, fog), 1.0);
        return;
    }
    if (abs(v_color.r - 1.0) > 0.001 || abs(v_color.g - 1.0) > 0.001 || abs(v_color.b - 1.0) > 0.001) {
        float diff = texel.g - max(texel.r, texel.b);
        float is_grass = clamp(diff / 0.05, 0.0, 1.0);
        float lum = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
        vec3 tinted_grass = clamp(lum * v_color * 2.1, 0.0, 1.0);
        base_rgb = mix(texel.rgb, tinted_grass, is_grass);
    }

    vec3 norm = normalize(gl_FrontFacing ? v_normal : -v_normal);
    float lambert = max(dot(norm, normalize(sun_dir)), 0.0);

    // Light and shadow: direct sunlight is shadowed when skylight is low
    float effective_skylight = v_skylight * sky_intensity;
    float sun_factor = smoothstep(0.4, 0.95, v_skylight) * sky_intensity;
    float direct_sun = lambert * 0.55 * sun_factor;
    float ambient_sky = effective_skylight * 0.70;

    vec3 sky_color = sun_color * (ambient_sky + direct_sun) * v_shade;
    vec3 block_rgb = v_blocklight_rgb;
    vec3 ir_warmth = vec3(0.65, 0.25, 0.08) * v_blocklight_ir;
    vec3 ambient_base = vec3(0.04);
    float fluor_response = fluorescence_response(v_fluorescence, v_blocklight_uv * 15.0);
    vec3 fluor_glow = v_fluorescence.rgb;
    float fluor_mask = 1.0;
    if (v_fluor_uv.x >= 0.0) {
        vec4 fluor_tex = texture(atlas, v_fluor_uv);
        fluor_mask = fluor_tex.a;
        fluor_glow *= fluor_tex.rgb;
    }
    float fluor_amount = fluor_response * fluor_mask;

    vec3 final_color;
    if (vision_mode == 0) {
        // Normal vision. The overlay alpha picks which pixels react, and its color keeps the variation.
        vec3 total_light = ambient_base + sky_color + block_rgb + ir_warmth * 0.35;
        vec3 shifted = mix(base_rgb, fluor_glow, clamp(fluor_amount, 0.0, 1.0));
        float room = clamp(max(total_light.r, max(total_light.g, total_light.b)), 0.0, 1.0);
        vec3 lit = shifted * total_light + fluor_glow * fluor_amount * (1.0 - room);
        float fog = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
        final_color = mix(lit, fog_color, fog);
    } else if (vision_mode == 1) {
        // UV vision: the field shows where the light is, and a defined block uses its own color.
        float shown = clamp(fluor_amount, 0.0, 1.0);
        vec3 shifted = mix(base_rgb, fluor_glow, shown);
        vec3 field = vec3(0.22, 0.05, 0.50) * v_blocklight_uv;
        vec3 dim_scene = shifted * 0.12 * (v_skylight * 0.5 + 0.1);
        vec3 lit = dim_scene + field + fluor_glow * shown;
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
in vec4 v_fluorescence;
in vec2 v_fluor_uv;

out vec4 fragColor;

float fluorescence_response(vec4 fluor, float uv_level) {
    float strength = floor(fluor.a / 256.0) / 16.0;
    if (strength <= 0.0) {
        return 0.0;
    }
    float start = mod(fluor.a, 16.0);
    float full_level = mod(floor(fluor.a / 16.0), 16.0);
    float end_level = max(full_level, start + 0.05);
    float t = clamp((uv_level - start) / (end_level - start), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t) * strength;
}

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

    vec3 norm = normalize(gl_FrontFacing ? v_normal : -v_normal);
    float lambert = max(dot(norm, normalize(sun_dir)), 0.0);

    float effective_skylight = v_skylight * sky_intensity;
    float sun_factor = smoothstep(0.4, 0.95, v_skylight) * sky_intensity;
    float direct_sun = lambert * 0.55 * sun_factor;
    float ambient_sky = effective_skylight * 0.70;

    vec3 sky_color = sun_color * (ambient_sky + direct_sun) * v_shade;
    vec3 block_rgb = v_blocklight_rgb;
    vec3 ir_warmth = vec3(0.65, 0.25, 0.08) * v_blocklight_ir;
    vec3 ambient_base = vec3(0.04);
    float fluor_response = fluorescence_response(v_fluorescence, v_blocklight_uv * 15.0);
    vec3 fluor_glow = v_fluorescence.rgb;
    float fluor_mask = 1.0;
    if (v_fluor_uv.x >= 0.0) {
        vec4 fluor_tex = texture(atlas, v_fluor_uv);
        fluor_mask = fluor_tex.a;
        fluor_glow *= fluor_tex.rgb;
    }
    float fluor_amount = fluor_response * fluor_mask;

    vec3 total_light = ambient_base + sky_color + block_rgb + ir_warmth * 0.35;
    vec3 shifted = mix(base_rgb, fluor_glow, clamp(fluor_amount, 0.0, 1.0));
    float room = clamp(max(total_light.r, max(total_light.g, total_light.b)), 0.0, 1.0);
    vec3 color = shifted * total_light + fluor_glow * fluor_amount * (1.0 - room);
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

WEATHER_VERT = """
#version 330 core
uniform mat4 vp;
uniform vec2 cloud_shift;
in vec3 in_pos;
in vec4 in_color;
in vec2 in_uv;
out vec4 v_color;
out vec2 v_uv;
void main() {
    gl_Position = vp * vec4(in_pos.x + cloud_shift.x, in_pos.y, in_pos.z + cloud_shift.y, 1.0);
    v_color = in_color;
    v_uv = in_uv;
}
"""

WEATHER_FRAG = """
#version 330 core
uniform int soft_round;
uniform int cloud_part;
in vec4 v_color;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    float fade = 1.0;
    if (soft_round == 1) {
        float r2 = dot(v_uv, v_uv);
        if (r2 > 1.0) discard;
        fade = pow(1.0 - r2, 0.65);
    }
    float a = v_color.a * fade;
    // 1 is the solid body, 2 is the soft rim. The body is drawn opaque so
    // the chunks underneath cannot tint it.
    if (cloud_part == 1 && a < 0.5) discard;
    if (cloud_part == 2 && a >= 0.5) discard;
    if (cloud_part == 1) a = 1.0;
    fragColor = vec4(v_color.rgb, a);
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

    vec3 norm = normalize(gl_FrontFacing ? v_normal : -v_normal);
    float lambert = max(dot(norm, normalize(sun_dir)), 0.0);
    float effective_skylight = entity_skylight * sky_intensity;
    float sun_factor = smoothstep(0.4, 0.95, entity_skylight) * sky_intensity;
    float direct_sun = lambert * 0.55 * sun_factor;
    float ambient_sky = effective_skylight * 0.70;

    vec3 sky_color = sun_color * (ambient_sky + direct_sun);
    vec3 total_light = vec3(0.04) + sky_color + entity_blocklight_rgb + vec3(0.65, 0.25, 0.08) * (entity_blocklight_ir * 0.35);

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
    // The art is a disc on a clear square. Drop the faint fringe so the
    // quad does not paint a box over the clouds.
    if (c.a < 0.45) discard;
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


# Far enough that the disc sits behind the cloud deck (Y=200) instead of
# in front of it. The size passed in is the apparent size at 90 blocks.
CELESTIAL_DISTANCE = 320.0


def _animation_frames(block_id):
    block = BLOCKS.get(block_id)
    rects = list(getattr(block, "anim_rects", ()) or ())
    if not rects:
        rects = [(0.0, 0.0, 1.0, 1.0)]
    while len(rects) < 4:
        rects.append(rects[-1])
    return tuple(tuple(float(value) for value in rect) for rect in rects[:4])


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

    c = np.array(center, dtype=np.float32) + d * CELESTIAL_DISTANCE
    s = float(size) * (CELESTIAL_DISTANCE / 90.0)
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
        from server.cow_model import CowMeshes
        from server.pig_model import PigMeshes
        from server.sheep_model import SheepMeshes
        from server.skeleton_model import SkeletonMeshes
        from server.zombie_model import ZombieMeshes
        from server.spider_model import SpiderMeshes

        self.pig_meshes = PigMeshes()
        self.pig_texture = None
        self.cow_meshes = CowMeshes()
        self.cow_texture = None
        self.sheep_meshes = SheepMeshes()
        self.sheep_texture = None
        self.zombie_meshes = ZombieMeshes()
        self.skeleton_meshes = SkeletonMeshes()
        self.zombie_texture = None
        self.desert_zombie_texture = None
        self.skeleton_texture = None
        self.desert_skeleton_texture = None
        self.spider_meshes = SpiderMeshes()
        self.spider_texture = None
        self.desert_spider_texture = None
        self._init_mob_textures()
        self.world["atlas"].value = 0
        self.translucent["atlas"].value = 0
        self.textured["atlas"].value = 0
        self.entity["atlas"].value = 0
        self._fog_color = SKY
        self._fog_near = FOG_NEAR
        self._fog_far = FOG_FAR
        if "atlas_size" in self.world:
            self.world["atlas_size"].value = (float(atlas_size[0]), float(atlas_size[1]))
            self.world["anim_time"].value = 0.0
            self.world["fire_frames"].value = _animation_frames("fire")
            self.world["blue_fire_frames"].value = _animation_frames("blue_fire")
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
        self.weather_prog = ctx.program(vertex_shader=WEATHER_VERT, fragment_shader=WEATHER_FRAG)
        self.cloud_vbo = self.ctx.buffer(reserve=64)
        self.weather_prog["soft_round"].value = 0
        self.weather_prog["cloud_part"].value = 0
        self.weather_prog["cloud_shift"].value = (0.0, 0.0)
        self.cloud_vao = self.ctx.vertex_array(
            self.weather_prog, [(self.cloud_vbo, "3f 4f 2f", "in_pos", "in_color", "in_uv")]
        )
        self.rain_vbo = self.ctx.buffer(reserve=64)
        self.rain_vao = self.ctx.vertex_array(
            self.weather_prog, [(self.rain_vbo, "3f 4f 2f", "in_pos", "in_color", "in_uv")]
        )

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
            fluorescence=getattr(mesh, "fluorescence", None),
            fluor_uvs=getattr(mesh, "fluor_uvs", None),
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
                fluorescence=getattr(mesh, "cutout_fluorescence", None),
                fluor_uvs=getattr(mesh, "cutout_fluor_uvs", None),
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
                fluorescence=getattr(mesh, "translucent_fluorescence", None),
                fluor_uvs=getattr(mesh, "translucent_fluor_uvs", None),
            )
        if opaque is None and cutout is None and translucent is None:
            return
        self.chunks[key] = _ChunkDraw(opaque, cutout, translucent, origin)

    def _buffer(self, program, vertices, indices, lights, layout, names, colors=None, fluorescence=None, fluor_uvs=None):
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
        if fluorescence is None or len(fluorescence) != len(vertices):
            fluorescence = np.zeros((len(vertices), 4), dtype=np.float32)
        vbo_fluor = self.ctx.buffer(np.ascontiguousarray(fluorescence, dtype=np.float32).tobytes())
        if fluor_uvs is None or len(fluor_uvs) != len(vertices):
            fluor_uvs = np.full((len(vertices), 2), -1.0, dtype=np.float32)
        vbo_fluor_uv = self.ctx.buffer(np.ascontiguousarray(fluor_uvs, dtype=np.float32).tobytes())
        vao = self.ctx.vertex_array(
            program,
            [
                (vbo, layout, *names),
                (vbo_light, "1f 3f 1f 1f", "in_skylight", "in_blocklight_rgb", "in_blocklight_uv", "in_blocklight_ir"),
                (vbo_color, "3f", "in_color"),
                (vbo_fluor, "4f", "in_fluorescence"),
                (vbo_fluor_uv, "2f", "in_fluor_uv"),
            ],
            ibo,
        )
        return vao, vbo, ibo, vbo_light, vbo_color, vbo_fluor, vbo_fluor_uv

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

    def set_view_fog(self, world, camera, sky_color):
        """Use a short tinted fog while the camera is inside a fluid."""
        self._fog_color = tuple(sky_color)
        self._fog_near = FOG_NEAR
        self._fog_far = FOG_FAR
        if world is None or camera is None:
            return None
        from server.fluid import submerged_fog

        block = world.get_block(math.floor(camera[0]), math.floor(camera[1]), math.floor(camera[2]))
        fog = submerged_fog(block)
        if fog is None:
            return None
        color, near, far = fog
        self._fog_color = tuple(color)
        self._fog_near = float(near)
        self._fog_far = float(far)
        return self._fog_color

    def _write_fog(self, program):
        program["fog_color"].value = self._fog_color
        program["fog_near"].value = self._fog_near
        program["fog_far"].value = self._fog_far

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

    def draw_clouds(self, vp, world, camera, now):
        weather = getattr(world, "weather", None)
        if not weather:
            return
        received = float(getattr(world, "weather_at", now))
        verts = _cloud_vertices(weather, now, received)
        if verts is None or len(verts) == 0:
            return
        shift = _cloud_shift(weather, now, received)
        # Rim first, against the sky. The body is opaque and writes depth so
        # the world drawn afterwards cannot show through it.
        self._draw_weather_verts(
            self.cloud_vbo, self.cloud_vao, vp, verts, moderngl.TRIANGLES,
            soft=True, shift=shift, part=2,
        )
        self._draw_weather_verts(
            self.cloud_vbo, self.cloud_vao, vp, verts, moderngl.TRIANGLES,
            soft=True, shift=shift, part=1, write_depth=True, upload=False,
        )

    def draw_rain(self, vp, world, camera, now):
        weather = getattr(world, "weather", None)
        if not weather:
            return
        received = float(getattr(world, "weather_at", now))
        verts = _rain_vertices(world, weather, camera, now)
        if verts is None or len(verts) == 0:
            return
        self._draw_weather_verts(
            self.rain_vbo, self.rain_vao, vp, verts, moderngl.TRIANGLES,
            shift=_cloud_shift(weather, now, received),
        )

    def _draw_weather_verts(self, vbo, vao, vp, verts, mode, soft=False, shift=(0.0, 0.0), part=0, write_depth=False, upload=True):
        if upload:
            data = np.asarray(verts, dtype=np.float32).tobytes()
            if vbo.size < len(data):
                vbo.orphan(len(data))
            vbo.write(data)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = bool(write_depth)
        if write_depth:
            self.ctx.disable(moderngl.BLEND)
        else:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        write_matrix(self.weather_prog["vp"], vp)
        self.weather_prog["soft_round"].value = 1 if soft else 0
        self.weather_prog["cloud_part"].value = int(part)
        self.weather_prog["cloud_shift"].value = (float(shift[0]), float(shift[1]))
        vao.render(mode, vertices=len(verts) // 9)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)


    def draw_world(self, vp, camera, cutout=True, day_time=1000.0):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        write_matrix(self.world["vp"], vp)
        self.world["camera"].value = camera
        sky_intensity = self.get_sky_intensity(day_time)
        sun_dir, sun_color = self.get_celestial_lighting(day_time)
        self._write_fog(self.world)
        if "anim_time" in self.world:
            self.world["anim_time"].value = float(day_time)
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
        self._write_fog(self.world)
        if "anim_time" in self.world:
            self.world["anim_time"].value = float(day_time)
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
        self.ctx.screen.depth_mask = False
        write_matrix(self.translucent["vp"], vp)
        self.translucent["camera"].value = camera
        sky_intensity = self.get_sky_intensity(day_time)
        sun_dir, sun_color = self.get_celestial_lighting(day_time)
        self._write_fog(self.translucent)
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
        self.ctx.screen.depth_mask = True
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
        self.draw_entity_flames(vp, player, day_time)

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

    def draw_fire_overlay(self, width, height, kind, day_time=1000.0):
        """First-person flames around the edges of the view."""
        rect = _flame_frame(kind, day_time)
        if rect is None:
            return
        u0, v0, u1, v1 = rect
        band = min(width, height) * 0.22
        floats = []
        floats.extend(_quad_uv(0, 0, band, height, u0, v1, u1, v0))
        floats.extend(_quad_uv(width - band, 0, width, height, u1, v1, u0, v0))
        floats.extend(_quad_uv(band * 0.5, 0, width - band * 0.5, band * 1.35, u0, v1, u1, v0))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._textured(ortho(width, height), floats, self.atlas)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_entity_flames(self, vp, entity, day_time=1000.0):
        """Two flame sheets crossed through a mob or a third-person player."""
        kind = getattr(entity, "fire_kind", "") or ""
        if kind not in ("fire", "blue_fire"):
            return
        rect = _flame_frame(kind, day_time)
        if rect is None or entity is None:
            return
        u0, v0, u1, v1 = rect
        x = float(entity.pos[0])
        y = float(entity.pos[1])
        z = float(entity.pos[2])
        height = max(0.4, float(getattr(entity, "height", 1.0) or 1.0))
        reach = max(0.35, float(getattr(entity, "radius", 0.3) or 0.3) * 1.6)
        y1 = y + height
        floats = []
        floats.extend(
            _world_quad((x - reach, y, z), (x + reach, y, z), (x + reach, y1, z), (x - reach, y1, z), u0, v0, u1, v1)
        )
        floats.extend(
            _world_quad((x, y, z - reach), (x, y, z + reach), (x, y1, z + reach), (x, y1, z - reach), u0, v0, u1, v1)
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.depth_mask = False
        self._textured(vp, floats, self.atlas)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

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
            self.draw_entity_flames(vp, player, day_time)

    def _draw_held(self, vp, camera, player, poses, item_id, first_person, eye, fog, day_time=1000.0):
        matrix = item_hold_matrix(player, poses, first_person=first_person, eye=eye, item_id=item_id)
        packed = held_item_mesh(item_id, matrix)
        if packed is None:
            return
        (vertices, indices), kind = packed
        texture = self.atlas if kind == "block" else self.items
        if first_person and kind != "block":
            # Depth already cleared for this pass; disable test so overlapping
            # extrusion pads don't moiré and near terrain can't poke through.
            self.ctx.disable(moderngl.DEPTH_TEST)
        self._draw_entity(vp, camera, vertices, indices, texture, fog=fog, day_time=day_time)
        self._draw_nocked_arrow(vp, camera, player, item_id, first_person, eye, fog, day_time)
        if first_person and kind != "block":
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _draw_skeleton_bow(self, vp, camera, entity, poses, pull, day_time):
        arm_pitch = poses.get("arm_r", (0.0, 0.0, 0.0))[0]
        matrix = held_item_matrix(entity.pos, entity.yaw, arm_pitch, flat=True, pull=0.0)
        if pull > 0.0:
            from server.player import look_direction

            lx, _ly, lz = look_direction(entity.yaw, 0.0)
            matrix = matrix.copy()
            matrix[0, 3] -= lx * 0.18 * pull
            matrix[2, 3] -= lz * 0.18 * pull
        packed = held_item_mesh("wood_bow", matrix)
        if packed is not None:
            (vertices, indices), kind = packed
            texture = self.atlas if kind == "block" else self.items
            self._draw_entity(vp, camera, vertices, indices, texture, fog=True, day_time=day_time)
        if pull <= 0.05:
            return
        arrow_matrix = third_person_nocked_arrow_matrix(entity.pos, entity.yaw, entity.pitch, pull)
        packed = held_item_mesh("arrow", arrow_matrix)
        if packed is None:
            return
        (vertices, indices), kind = packed
        texture = self.atlas if kind == "block" else self.items
        self._draw_entity(vp, camera, vertices, indices, texture, fog=True, day_time=day_time)

    def _draw_nocked_arrow(self, vp, camera, player, item_id, first_person, eye, fog, day_time):
        from client.player_render import _bow_pull

        pull = _bow_pull(player, item_id)
        arrow_id = str(getattr(player, "nocked_arrow", "") or "")
        if pull <= 0.0 or not arrow_id:
            return
        if first_person:
            matrix = nocked_arrow_matrix(eye, player.yaw, player.pitch, pull)
        else:
            matrix = third_person_nocked_arrow_matrix(player.pos, player.yaw, player.pitch, pull)
        packed = held_item_mesh(arrow_id, matrix)
        if packed is None:
            return
        (vertices, indices), kind = packed
        texture = self.atlas if kind == "block" else self.items
        self._draw_entity(vp, camera, vertices, indices, texture, fog=fog, day_time=day_time)

    def _init_mob_textures(self):
        cow_path = ASSETS / "textures" / "entity" / "cow.png"
        if not cow_path.is_file():
            cow_path = ASSETS / "textures" / "cow.png"
        if cow_path.is_file():
            try:
                rgba = read_png(cow_path)
                image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
                self.cow_texture = self._create_skin_texture(image)
            except Exception:
                self.cow_texture = None

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

        zombie_path = ASSETS / "textures" / "entity" / "zombie.png"
        if not zombie_path.is_file():
            zombie_path = ASSETS / "textures" / "zombie.png"
        if zombie_path.is_file():
            try:
                rgba = read_png(zombie_path)
                image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
                self.zombie_texture = self._create_skin_texture(image)
            except Exception:
                self.zombie_texture = None

        spider_path = ASSETS / "textures" / "entity" / "spider.png"
        if not spider_path.is_file():
            spider_path = ASSETS / "textures" / "spider.png"
        if spider_path.is_file():
            try:
                rgba = read_png(spider_path)
                image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
                self.spider_texture = self._create_skin_texture(image)
            except Exception:
                self.spider_texture = None

        for attr, filename in (
            ("desert_zombie_texture", "desert_zombie.png"),
            ("skeleton_texture", "skeleton.png"),
            ("desert_skeleton_texture", "desert_skeleton.png"),
            ("desert_spider_texture", "desert_spider.png"),
        ):
            path = ASSETS / "textures" / "entity" / filename
            if not path.is_file():
                setattr(self, attr, None)
                continue
            try:
                rgba = read_png(path)
                image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
                setattr(self, attr, self._create_skin_texture(image))
            except Exception:
                setattr(self, attr, None)

    def _asset_mob_texture(self, relative):
        if not relative:
            return None
        cache = getattr(self, "_asset_mob_textures", None)
        if cache is None:
            cache = {}
            self._asset_mob_textures = cache
        if relative in cache:
            return cache[relative]
        path = ASSETS / "textures" / relative
        if not path.is_file():
            cache[relative] = None
            return None
        rgba = read_png(path)
        image = np.flipud(np.ascontiguousarray(rgba, dtype=np.uint8))
        cache[relative] = self._create_skin_texture(image)
        return cache[relative]

    def _draw_asset_mob(self, vp, camera, entity, time_val, day_time):
        from server.mobs import MOBS

        definition = MOBS.get(getattr(entity, "mob_type", ""))
        if definition is None or definition.model.kind == "builtin":
            return False
        model = definition.model.compile()
        if model is None:
            return True
        texture = self._asset_mob_texture(definition.model.texture)
        if texture is None:
            return True
        speed = math.hypot(entity.vel[0], entity.vel[2])
        eye_pitch = getattr(entity, "pitch", 0.0)
        body_pitch = 0.0
        if getattr(entity, "can_fly", False):
            # Keep the wings beating while it hovers, climbs, or dives, and nose the body along.
            speed = max(0.55, math.hypot(entity.vel[0], entity.vel[1], entity.vel[2]))
            body_pitch = eye_pitch
            eye_pitch = 0.0
        vertices, indices = model.posed_vertices(
            entity.pos,
            entity.yaw,
            time_val,
            speed,
            pitch=eye_pitch,
            body_pitch=body_pitch,
        )
        if len(vertices) == 0:
            return True
        if getattr(entity, "is_baby", False):
            vertices = _scale_about_feet(vertices, entity.pos, 0.5)
        hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
        self._draw_entity(
            vp,
            camera,
            vertices,
            indices,
            texture,
            fog=True,
            day_time=day_time,
            hurt_flash=hurt,
        )
        return True

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
                if self._draw_asset_mob(vp, camera, entity, time_val, day_time):
                    continue
                if mob_type == "cow" and self.cow_texture is not None:
                    from server.cow_model import cow_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    poses = cow_pose(entity.walk_time, hspeed, entity.yaw, pitch=entity.pitch)
                    vertices, indices = self.cow_meshes.posed_vertices(
                        entity.pos, entity.yaw, poses, baby=bool(getattr(entity, "is_baby", False))
                    )
                    hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
                    self._draw_entity(
                        vp,
                        camera,
                        vertices,
                        indices,
                        self.cow_texture,
                        fog=True,
                        day_time=day_time,
                        hurt_flash=hurt,
                    )
                elif mob_type == "pig" and self.pig_texture is not None:
                    from server.pig_model import pig_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    poses = pig_pose(entity.walk_time, hspeed, entity.yaw, pitch=entity.pitch)
                    vertices, indices = self.pig_meshes.posed_vertices(
                        entity.pos, entity.yaw, poses, baby=bool(getattr(entity, "is_baby", False))
                    )
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
                        entity.pos,
                        entity.yaw,
                        poses,
                        sheared=sheared,
                        baby=bool(getattr(entity, "is_baby", False)),
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
                elif mob_type in ("zombie", "desert_zombie"):
                    texture = self.zombie_texture if mob_type == "zombie" else self.desert_zombie_texture
                    if texture is None:
                        continue
                    from server.zombie_model import zombie_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    poses = zombie_pose(entity.walk_time, hspeed, entity.yaw, pitch=entity.pitch)
                    vertices, indices = self.zombie_meshes.posed_vertices(entity.pos, entity.yaw, poses)
                    hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
                    self._draw_entity(
                        vp,
                        camera,
                        vertices,
                        indices,
                        texture,
                        fog=True,
                        day_time=day_time,
                        hurt_flash=hurt,
                    )
                elif mob_type in ("skeleton", "desert_skeleton"):
                    texture = self.skeleton_texture if mob_type == "skeleton" else self.desert_skeleton_texture
                    if texture is None:
                        continue
                    from server.zombie_model import skeleton_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    pull = float(getattr(entity, "bow_pull", 0.0) or 0.0)
                    poses = skeleton_pose(entity.walk_time, hspeed, entity.yaw, pitch=entity.pitch, pull=pull)
                    vertices, indices = self.skeleton_meshes.posed_vertices(entity.pos, entity.yaw, poses)
                    hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
                    self._draw_entity(
                        vp,
                        camera,
                        vertices,
                        indices,
                        texture,
                        fog=True,
                        day_time=day_time,
                        hurt_flash=hurt,
                    )
                    self._draw_skeleton_bow(vp, camera, entity, poses, pull, day_time)
                elif mob_type in ("spider", "desert_spider"):
                    texture = self.spider_texture if mob_type == "spider" else self.desert_spider_texture
                    if texture is None:
                        continue
                    from server.spider_model import spider_pose

                    hspeed = math.hypot(entity.vel[0], entity.vel[2])
                    climbing = bool(getattr(entity, "climbing", False))
                    on_ceiling = bool(getattr(entity, "on_ceiling", False))
                    vspeed = abs(entity.vel[1]) if climbing else 0.0
                    movespeed = max(hspeed, vspeed)
                    poses = spider_pose(
                        entity.walk_time,
                        movespeed,
                        entity.yaw,
                        pitch=entity.pitch,
                        climbing=climbing,
                        on_ceiling=on_ceiling,
                    )
                    vertices, indices = self.spider_meshes.posed_vertices(entity.pos, entity.yaw, poses)
                    hurt = 1.0 if getattr(entity, "hurt_time", 0.0) > 0 else 0.0
                    self._draw_entity(
                        vp,
                        camera,
                        vertices,
                        indices,
                        texture,
                        fog=True,
                        day_time=day_time,
                        hurt_flash=hurt,
                    )
                continue
            if getattr(entity, "is_projectile", False):
                matrix = arrow_flight_matrix(entity.pos, entity.vel)
                packed = held_item_mesh(getattr(entity, "arrow_id", ""), matrix)
                if packed is None:
                    continue
                (vertices, indices), kind = packed
                texture = self.atlas if kind == "block" else self.items
                self._draw_entity(vp, camera, vertices, indices, texture, fog=True, day_time=day_time)
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
        for entity in entities:
            if getattr(entity, "fire_kind", ""):
                self.draw_entity_flames(vp, entity, day_time)

    def draw_particles(self, vp, camera, system):
        particles = getattr(system, "particles", None)
        if not particles:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.CULL_FACE)
        grouped = {}
        for particle in particles:
            if particle.life <= 0.0:
                continue
            alpha = max(0.0, 1.0 - particle.age / particle.life)
            if alpha <= 0.02:
                continue
            key = round(alpha * 8) / 8.0
            right, up = _billboard_axes(camera, (particle.x, particle.y, particle.z))
            grouped.setdefault(key, []).extend(
                _heart_billboard((particle.x, particle.y, particle.z), right, up, particle.size)
            )
        for alpha, floats in grouped.items():
            if floats:
                self._tris(vp, floats, (0.93, 0.16, 0.28, alpha))
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)

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
        self._write_fog(self.entity)
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

    def draw_hud(self, width, height, hotbar, selected, crosshair, break_progress=0.0, health=None, max_health=20.0, hunger=None, max_hunger=20.0):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        projection = ortho(width, height)
        self._hotbar(projection, width, height, hotbar, selected)
        if health is not None:
            self._hearts(projection, width, height, health, max_health)
        if hunger is not None:
            self._hunger(projection, width, height, hunger, max_hunger)
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

    def _hunger(self, projection, width, height, hunger, max_hunger):
        try:
            hunger = float(hunger)
            max_hunger = float(max_hunger)
        except (TypeError, ValueError):
            return
        if max_hunger <= 0.0:
            return
        shanks = max(1, int(math.ceil(max_hunger / 2.0)))
        slot = 50
        gap = 8
        total = HOTBAR_SLOTS * slot + (HOTBAR_SLOTS - 1) * gap
        x0 = (width - total) / 2
        x_right = x0 + total
        y0 = 28 + slot + 10
        scale = 3
        spacing = 2
        drumstick_width = DRUMSTICK_PATTERN_WIDTH * scale
        for slot_idx in range(shanks):
            dx = x_right - (shanks - slot_idx) * drumstick_width - (shanks - 1 - slot_idx) * spacing
            self._drumstick(projection, dx, y0, scale, (0.18, 0.12, 0.08, 0.9), 1.0)
            value = max(0.0, min(2.0, hunger - slot_idx * 2.0))
            if value <= 0.0:
                continue
            color = (0.78, 0.46, 0.16, 1.0) if value >= 2.0 else (0.85, 0.58, 0.22, 1.0)
            self._drumstick(projection, dx, y0, scale, color, value / 2.0)

    def _drumstick(self, projection, x, y, scale, color, fraction):
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction <= 0.0:
            return
        floats = []
        for row, line in enumerate(DRUMSTICK_PATTERN):
            for col, pixel in enumerate(line):
                if pixel != "1":
                    continue
                if (col + 0.5) / DRUMSTICK_PATTERN_WIDTH > fraction:
                    continue
                px = x + col * scale
                py = y + (DRUMSTICK_PATTERN_HEIGHT - 1 - row) * scale
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
        durability_bg = []
        durability_fg = []
        for index in range(HOTBAR_SLOTS):
            stack = hotbar[index] if index < len(hotbar) else ""
            block_id = stack_id(stack)
            sx = x + index * (slot + gap)
            if index == selected:
                highlight.extend(_quad(sx - 4, y - 4, sx + slot + 4, y + slot + 4))
            slots.extend(_quad(sx, y, sx + slot, y + slot))
            item = ITEMS.get(block_id)
            if item is None:
                continue
            u0, v0, u1, v1 = item.icon
            icons.extend(_quad_uv(sx + 7, y + 7, sx + slot - 7, y + slot - 7, u0, v0, u1, v1))
            damage = stack_damage(stack)
            if getattr(item, "durability", 0) and damage > 0:
                ratio = max(0.0, min(1.0, 1.0 - damage / item.durability))
                bar_max_w = slot - 14
                bar_w = max(1, int(bar_max_w * ratio))
                durability_bg.extend(_quad(sx + 7, y + 4, sx + slot - 7, y + 7))
                r = min(1.0, 2.0 * (1.0 - ratio))
                g = min(1.0, 2.0 * ratio)
                durability_fg.append((_quad(sx + 7, y + 4, sx + 7 + bar_w, y + 6), (r, g, 0.1, 1.0)))
        if highlight:
            self._tris(projection, highlight, (0.97, 0.97, 0.94, 1.0))
        self._tris(projection, slots, (0.12, 0.1, 0.08, 0.88))
        if icons:
            self._textured(projection, icons, self.items)
        if durability_bg:
            self._tris(projection, durability_bg, (0.0, 0.0, 0.0, 0.9))
        for fg_quads, color in durability_fg:
            self._tris(projection, fg_quads, color)

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


def _weather_grids(weather):
    from server.weather import unpack_grid

    size = int(weather["size"])
    density = weather.get("_density")
    rain = weather.get("_rain")
    if density is None:
        density = unpack_grid(weather["density"], size)
        weather["_density"] = density
    if rain is None:
        rain = unpack_grid(weather["rain"], size)
        weather["_rain"] = rain
    return density, rain


def cloud_cover(x, z):
    """How much of this spot is a cloud body. Low values are the gaps between them."""
    from server.noise import fbm

    broad = fbm(float(x) * 0.0075, float(z) * 0.0075, 4, 5100)
    detail = fbm(float(x) * 0.038 + 14.0, float(z) * 0.038 - 5.0, 3, 7300)
    return broad * 0.58 + detail * 0.42


def _cloud_shift(weather, now, received_at):
    wind = weather.get("wind") or (0.0, 0.0)
    age = min(0.3, max(0.0, float(now) - float(received_at)))
    return float(wind[0]) * age, float(wind[1]) * age


_DISC_X = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
_DISC_Z = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
# A rainy patch only needs a handful of streaks, and only under a puff.
_RAIN_REACH = 18.0
_RAIN_CAP = 24


def _disc_mesh(cx, y, cz, radius, red, green, blue, alpha):
    x = cx[:, None] + _DISC_X * radius[:, None]
    z = cz[:, None] + _DISC_Z * radius[:, None]
    yy = np.broadcast_to(y[:, None], x.shape)
    cols = np.stack(
        [
            np.broadcast_to(red[:, None], x.shape),
            np.broadcast_to(green[:, None], x.shape),
            np.broadcast_to(blue[:, None], x.shape),
            np.broadcast_to(alpha[:, None], x.shape),
        ],
        axis=2,
    )
    uvs = np.stack([np.broadcast_to(_DISC_X, x.shape), np.broadcast_to(_DISC_Z, x.shape)], axis=2)
    return np.concatenate([np.stack([x, yy, z], axis=2), cols, uvs], axis=2).reshape(-1)


def _cloud_vertices(weather, now, received_at):
    # The deck is rebuilt only when a new weather packet arrives. Wind between
    # packets is a shader offset, so a frame does not rebuild the mesh.
    if weather.get("_cloud_ready"):
        return weather.get("_cloud_mesh")
    try:
        density, rain = _weather_grids(weather)
        ox = float(weather["origin"][0])
        oz = float(weather["origin"][1])
        cell = float(weather["cell"])
        cloud_y = float(weather.get("cloud_y", 200.0))
    except Exception:
        return None
    from server.noise import fbm_grid, hash2_grid

    height, width = density.shape
    ii, jj = np.indices((height, width))
    amount = density.astype(np.float64) / 255.0
    wet = rain.astype(np.float64) / 255.0
    xs = ox + (jj + 0.5) * cell
    zs = oz + (ii + 0.5) * cell
    cover = fbm_grid(xs * 0.0075, zs * 0.0075, 4, 5100) * 0.58 + fbm_grid(
        xs * 0.038 + 14.0, zs * 0.038 - 5.0, 3, 7300
    ) * 0.42
    mask = (amount >= 0.12) & (cover >= 0.56 - np.minimum(amount, 1.0) * 0.08)
    sel_i, sel_j = np.nonzero(mask)
    if sel_i.size == 0:
        weather["_cloud_mesh"] = None
        weather["_puffs"] = np.zeros((0, 4), dtype=np.float32)
        weather["_cloud_ready"] = True
        return None
    count = 3
    ii = np.repeat(sel_i, count)
    jj = np.repeat(sel_j, count)
    slot = np.tile(np.arange(count), sel_i.size)
    amt = np.repeat(amount[sel_i, sel_j], count)
    wetv = np.repeat(wet[sel_i, sel_j], count)
    jx = (hash2_grid(jj * 5 + slot, ii, 17) - 0.5) * cell * 1.2
    jz = (hash2_grid(jj * 5 + slot, ii, 29) - 0.5) * cell * 1.2
    radius = (2.4 + amt * 3.2 + wetv * 1.8) * (0.72 + hash2_grid(jj, ii * 3 + slot, 41) * 0.6)
    y = cloud_y + (hash2_grid(jj + slot, ii, 53) - 0.25) * 4.0
    cx = ox + (jj + 0.5) * cell + jx
    cz = oz + (ii + 0.5) * cell + jz
    # A puff turns grey as soon as it is wet enough to rain, and stays white
    # when the cell under it is dry.
    storm = np.clip(wetv / 0.22, 0.0, 1.0)
    red = 0.97 * (1.0 - storm) + 0.50 * storm
    green = 0.98 * (1.0 - storm) + 0.54 * storm
    blue = 0.99 * (1.0 - storm) + 0.60 * storm
    alpha = np.minimum(0.94, 0.58 + amt * 0.34)
    mesh = np.concatenate(
        [
            _disc_mesh(cx, y, cz, radius, red, green, blue, alpha),
            _disc_mesh(
                cx - jx * 0.4,
                y + 1.8,
                cz - jz * 0.4,
                radius * 0.62,
                red * 0.92,
                green * 0.94,
                blue * 0.97,
                alpha * 0.55,
            ),
        ]
    ).astype(np.float32, copy=False)
    weather["_cloud_mesh"] = mesh
    weather["_puffs"] = np.stack([cx, cz, radius, wetv], axis=1).astype(np.float32, copy=False)
    weather["_cloud_ready"] = True
    return mesh


def _precip_streaks(sx, sz, now, top, bottom, span, length, speed, half_w, color, alpha, shortest, cam_x, cam_z):
    """Camera-facing vertical streaks or short flakes."""
    if sx is None or len(sx) == 0:
        return None
    sx = np.asarray(sx, dtype=np.float32)
    sz = np.asarray(sz, dtype=np.float32)
    length = np.asarray(length, dtype=np.float32)
    if length.ndim == 0:
        length = np.full(sx.shape, float(length), dtype=np.float32)
    half_w = np.asarray(half_w, dtype=np.float32)
    if half_w.ndim == 0:
        half_w = np.full(sx.shape, float(half_w), dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim == 0:
        alpha = np.full(sx.shape, float(alpha), dtype=np.float32)
    fall = (float(now) * float(speed) + np.arange(sx.shape[0], dtype=np.float32) * 3.7) % (span + length)
    head = np.minimum(top, top - fall + length)
    tail = np.maximum(bottom, head - length)
    keep = head - tail >= shortest
    if not np.any(keep):
        return None
    sx, sz = sx[keep], sz[keep]
    head, tail, half_w, alpha = head[keep], tail[keep], half_w[keep], alpha[keep]
    rx = float(cam_z) - sz
    rz = sx - float(cam_x)
    scale = np.maximum(np.hypot(rx, rz), 1e-4)
    rx = rx / scale * half_w
    rz = rz / scale * half_w
    side = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    y_at = np.stack([head, head, tail, head, tail, tail], axis=1)
    x = sx[:, None] + rx[:, None] * side
    z = sz[:, None] + rz[:, None] * side
    red = np.full(sx.shape, color[0], dtype=np.float32)
    green = np.full(sx.shape, color[1], dtype=np.float32)
    blue = np.full(sx.shape, color[2], dtype=np.float32)
    zeros = np.zeros(x.shape, dtype=np.float32)
    cols = np.stack(
        [
            np.broadcast_to(red[:, None], x.shape),
            np.broadcast_to(green[:, None], x.shape),
            np.broadcast_to(blue[:, None], x.shape),
            np.broadcast_to(alpha[:, None], x.shape),
        ],
        axis=2,
    )
    uvs = np.stack([zeros, zeros], axis=2)
    return np.concatenate([np.stack([x, y_at, z], axis=2), cols, uvs], axis=2).reshape(-1).astype(np.float32, copy=False)


def _rain_vertices(world, weather, camera, now):
    if not weather.get("_cloud_ready"):
        _cloud_vertices(weather, now, now)
    puffs = weather.get("_puffs")
    if puffs is None or len(puffs) == 0:
        return None
    try:
        cloud_y = float(weather.get("cloud_y", 200.0))
    except Exception:
        return None
    cam_x = float(camera[0])
    cam_y = float(camera[1])
    cam_z = float(camera[2])
    dx = puffs[:, 0] - cam_x
    dz = puffs[:, 1] - cam_z
    dist2 = dx * dx + dz * dz
    wet = puffs[:, 3]
    pick = np.flatnonzero((wet >= 0.08) & (dist2 <= _RAIN_REACH * _RAIN_REACH))
    if pick.size == 0:
        return None
    if pick.size > _RAIN_CAP:
        pick = pick[np.argsort(dist2[pick])[:_RAIN_CAP]]
    from server.noise import hash2_grid

    cx = puffs[pick, 0]
    cz = puffs[pick, 1]
    radius = puffs[pick, 2]
    amount = puffs[pick, 3]
    # One roof sample for the player. Streaks are local, so this is the cover
    # that matters, and it stays one lookup instead of one per streak.
    roof = 0
    tops_of = getattr(world, "column_top", None)
    if tops_of is not None:
        try:
            roof = int(tops_of(int(cam_x), int(cam_z)))
        except Exception:
            roof = 0
    if cam_y <= roof + 1.0:
        return None
    # Two streaks inside the puff, so a gap between clouds stays dry.
    spin = hash2_grid(pick, pick * 3 + 1, 11)
    spin2 = hash2_grid(pick, pick * 3 + 1, 23)
    ang = np.concatenate([spin, spin2]) * math.tau
    reach = np.concatenate([radius, radius])
    rad = np.concatenate([spin, spin2]) * reach * 0.65
    sx = np.concatenate([cx, cx]) + np.cos(ang) * rad
    sz = np.concatenate([cz, cz]) + np.sin(ang) * rad
    amount = np.concatenate([amount, amount])
    top = min(cloud_y - 6.0, cam_y + 16.0)
    bottom = max(roof + 1.2, cam_y - 12.0)
    span = top - bottom
    if span < 4.0:
        return None
    snowing = np.zeros(sx.shape[0], dtype=bool)
    biomes = getattr(world, "biomes", None)
    if biomes:
        from server.biomes import biome_at

        seed = getattr(world, "seed", 0)
        for index, (px, pz) in enumerate(zip(sx, sz)):
            biome = biome_at(float(px), float(pz), seed, biomes)
            snowing[index] = getattr(biome, "precipitation", "rain") == "snow"
    if np.any(snowing):
        flakes = _precip_streaks(
            sx[snowing], sz[snowing], now, top, bottom, span,
            length=0.42, speed=4.2, half_w=0.11, color=(0.97, 0.98, 1.0), alpha=0.9, shortest=0.2,
            cam_x=cam_x, cam_z=cam_z,
        )
        rain = None
        if np.any(~snowing):
            rain_amount = amount[~snowing]
            rain = _precip_streaks(
                sx[~snowing], sz[~snowing], now, top, bottom, span,
                length=5.0 + rain_amount * 3.0, speed=22.0, half_w=0.05 + rain_amount * 0.03,
                color=(0.78, 0.86, 0.95), alpha=np.minimum(0.8, 0.5 + rain_amount * 0.35), shortest=1.5,
                cam_x=cam_x, cam_z=cam_z,
            )
        parts = [part for part in (flakes, rain) if part is not None and len(part)]
        if not parts:
            return None
        return np.concatenate(parts) if len(parts) > 1 else parts[0]
    length = 5.0 + amount * 3.0
    # phase grows with time and the window moves down from the cloud deck.
    fall = (float(now) * 22.0 + np.arange(sx.shape[0], dtype=np.float32) * 3.7) % (span + length)
    head = np.minimum(top, top - fall + length)
    tail = np.maximum(bottom, head - length)
    long_enough = head - tail >= 1.5
    if not np.any(long_enough):
        return None
    sx, sz, amount = sx[long_enough], sz[long_enough], amount[long_enough]
    head, tail = head[long_enough], tail[long_enough]
    alpha = np.minimum(0.8, 0.5 + amount * 0.35)
    half_w = 0.05 + amount * 0.03
    rx = cam_z - sz
    rz = sx - cam_x
    scale = np.maximum(np.hypot(rx, rz), 1e-4)
    rx = rx / scale * half_w
    rz = rz / scale * half_w
    side = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    y_at = np.stack([head, head, tail, head, tail, tail], axis=1)
    x = sx[:, None] + rx[:, None] * side
    z = sz[:, None] + rz[:, None] * side
    red = np.full(sx.shape, 0.78, dtype=np.float32)
    green = np.full(sx.shape, 0.86, dtype=np.float32)
    blue = np.full(sx.shape, 0.95, dtype=np.float32)
    zeros = np.zeros(x.shape, dtype=np.float32)
    cols = np.stack(
        [
            np.broadcast_to(red[:, None], x.shape),
            np.broadcast_to(green[:, None], x.shape),
            np.broadcast_to(blue[:, None], x.shape),
            np.broadcast_to(alpha[:, None], x.shape),
        ],
        axis=2,
    )
    uvs = np.stack([zeros, zeros], axis=2)
    return np.concatenate([np.stack([x, y_at, z], axis=2), cols, uvs], axis=2).reshape(-1).astype(np.float32, copy=False)


def _scale_about_feet(vertices, feet, scale):
    if len(vertices) == 0 or scale == 1.0:
        return vertices
    shaped = vertices.reshape(-1, 8).copy()
    ox, oy, oz = float(feet[0]), float(feet[1]), float(feet[2])
    shaped[:, 0] = ox + (shaped[:, 0] - ox) * scale
    shaped[:, 1] = oy + (shaped[:, 1] - oy) * scale
    shaped[:, 2] = oz + (shaped[:, 2] - oz) * scale
    return shaped.reshape(-1)


def _billboard_axes(camera, origin):
    fx = float(camera[0]) - float(origin[0])
    fy = float(camera[1]) - float(origin[1])
    fz = float(camera[2]) - float(origin[2])
    length = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / length, fy / length, fz / length
    rx, rz = fz, -fx
    rlen = math.hypot(rx, rz)
    if rlen < 1e-4:
        right = (1.0, 0.0, 0.0)
    else:
        right = (rx / rlen, 0.0, rz / rlen)
    ux = fy * right[2] - fz * right[1]
    uy = fz * right[0] - fx * right[2]
    uz = fx * right[1] - fy * right[0]
    return right, (ux, uy, uz)


def _heart_billboard(origin, right, up, size):
    scale = float(size) / HEART_PATTERN_WIDTH
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
    rx, ry, rz = right
    ux, uy, uz = up
    half_w = HEART_PATTERN_WIDTH * 0.5
    half_h = HEART_PATTERN_HEIGHT * 0.5
    floats = []
    for row, line in enumerate(HEART_PATTERN):
        for col, pixel in enumerate(line):
            if pixel != "1":
                continue
            x0 = (col - half_w) * scale
            y0 = ((HEART_PATTERN_HEIGHT - 1 - row) - half_h) * scale
            x1 = x0 + scale
            y1 = y0 + scale
            for lx, ly in ((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)):
                floats.append(ox + rx * lx + ux * ly)
                floats.append(oy + ry * lx + uy * ly)
                floats.append(oz + rz * lx + uz * ly)
    return floats


def _quad(x0, y0, x1, y1):
    return [
        x0, y0, 0, x1, y0, 0, x1, y1, 0,
        x0, y0, 0, x1, y1, 0, x0, y1, 0,
    ]


def _flame_frame(kind, day_time):
    block_id = "blue_fire" if kind == "blue_fire" else "fire"
    frames = _animation_frames(block_id)
    if not frames:
        return None
    frame = int(math.floor(float(day_time) * 0.4)) % len(frames)
    rect = frames[frame]
    if len(rect) < 4:
        return None
    return rect


def _world_quad(p0, p1, p2, p3, u0, v0, u1, v1):
    def vert(point, u, v):
        return (float(point[0]), float(point[1]), float(point[2]), float(u), float(v))

    a = vert(p0, u0, v1)
    b = vert(p1, u1, v1)
    c = vert(p2, u1, v0)
    d = vert(p3, u0, v0)
    return [value for corner in (a, b, c, a, c, d) for value in corner]


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
