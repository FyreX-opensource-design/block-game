mkdir -p ./assets/textures/{smooth_sandstone,smooth_granite,smooth_diorite,smooth_andesite,smooth_marble,smooth_limestone,stone,granite,andesite,diorite,limestone,marble,sandstone}
mkdir -p ./assets/textures/templates/{patterns,borders,ores,smooth_stone}
mkdir -p ./assets/textures/templates/generated/{colors,specks,borders,smooth_bases}

./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#adacac" --tint-strength 1.0 -B 0.95 -o ./assets/textures/templates/generated/colors/stone_color.png
./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#db8888" --tint-strength 0.9 -B 0.95 -o ./assets/textures/templates/generated/colors/granite_color.png
./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#86868c" --tint-strength 0.9 -B 0.6 -o ./assets/textures/templates/generated/colors/andesite_color.png
./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#fafafa" --tint-strength 0.3 -B 0.9 -o ./assets/textures/templates/generated/colors/diorite_color.png
./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#f9fad4" --tint-strength 0.5 -B 0.6 -o ./assets/textures/templates/generated/colors/limestone_color.png
./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#e3dede" --tint-strength 0.4 -B 0.9 -o ./assets/textures/templates/generated/colors/marble_color.png
./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/patterns/blank.png --tint "#c7b76f" --tint-strength 0.9 -B 0.6 -o ./assets/textures/templates/generated/colors/sandstone_color.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/patterns/specks_1.png -n 5 -s 25 --preserve-alpha --seed 15249 -o ./assets/textures/templates/generated/specks/specks_1_noise_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/patterns/specks_2.png -n 5 -s 15 --preserve-alpha --seed 16518 -o ./assets/textures/templates/generated/specks/specks_2_noise_{i}.png

for i in {0..12} {14..16}
do
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/smooth_stone/smooth_stone_$i.png --tint "#80868c" --tint-strength 0.8 -B 0.6 -o ./assets/textures/templates/generated/smooth_bases/smooth_andesite_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/smooth_bases/smooth_andesite_$i.png -n 5 -s 7 --seed 25 -o ./assets/textures/smooth_andesite/smooth_andesite_var_$i.png

    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/smooth_stone/smooth_stone_$i.png --tint "#ffffff" --tint-strength 0.8 -B 0.95 -o ./assets/textures/templates/generated/smooth_bases/smooth_marble_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/smooth_bases/smooth_marble_$i.png -n 5 -s 10 -c "#383838" --seed 1 -o ./assets/textures/smooth_marble/smooth_marble_var_$i.png

    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/smooth_stone/smooth_stone_$i.png --tint "#f9fad4" --tint-strength 0.5 -B 0.6 -o ./assets/textures/templates/generated/smooth_bases/smooth_limestone_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/smooth_bases/smooth_limestone_$i.png -n 5 -s 15 -c "#e5e6ca" --seed 205 -o ./assets/textures/smooth_limestone/smooth_limestone_var_$i.png
done

./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/stone_color.png -n 5 -s 15 --seed 1234 -o ./assets/textures/stone/stone_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/granite_color.png -n 5 -s 15 --seed 18947 -o ./assets/textures/granite/granite_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/andesite_color.png -n 5 -s 7 --seed 25 -o ./assets/textures/andesite/andesite_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/diorite_color.png -n 5 -s 15 --seed 85 -o ./assets/textures/diorite/diorite_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/limestone_color.png -n 5 -s 15 -c "#292929" --seed 205 -o ./assets/textures/limestone/limestone_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/marble_color.png -n 5 -s 10 --seed 1 -o ./assets/textures/marble/marble_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/sandstone_color.png -n 5 -s 10 --seed 16518 -o ./assets/textures/sandstone/sandstone_var_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/sandstone_color.png -n 5 -s 5 --seed 16518 -o ./assets/textures/templates/generated/smooth_bases/smooth_sandstone_var_16_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/granite_color.png -n 5 -s 12 --seed 75 -o ./assets/textures/templates/generated/smooth_bases/smooth_granite_var_16_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/diorite_color.png -n 5 -s 8 --seed 238751 -o ./assets/textures/templates/generated/smooth_bases/smooth_diorite_var_16_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/andesite_color.png -n 5 -s 16 --seed 1389742 -o ./assets/textures/templates/generated/smooth_bases/smooth_andesite_var_16_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/limestone_color.png -n 5 -s 16 --seed 3287594 -o ./assets/textures/templates/generated/smooth_bases/smooth_limestone_var_16_{i}.png
./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/colors/marble_color.png -n 5 -s 10 --seed 543142 -o ./assets/textures/templates/generated/smooth_bases/smooth_marble_var_16_{i}.png

for i in {0..15}
do
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/borders/border_$i.png --tint "#8b8c5a" --tint-strength 0.8 -B 0.7 -o ./assets/textures/templates/generated/borders/smooth_sandstone_border_$i.png
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/borders/border_$i.png --tint "#ba6c78" --tint-strength 0.9 -B 0.5 -o ./assets/textures/templates/generated/borders/smooth_granite_border_$i.png
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/borders/border_$i.png --tint "#c2c0b6" --tint-strength 0.5 -B 0.1 -o ./assets/textures/templates/generated/borders/smooth_diorite_border_$i.png
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/borders/border_$i.png --tint "#505c5e" --tint-strength 0.9 -B 0.3 -o ./assets/textures/templates/generated/borders/smooth_andesite_border_$i.png
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/borders/border_$i.png --tint "#505c5e" --tint-strength 0.9 -B 0.3 -o ./assets/textures/templates/generated/borders/smooth_limestone_border_$i.png
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/templates/borders/border_$i.png --tint "#f5f0ea" --tint-strength 0.5 -B 0.2 -o ./assets/textures/templates/generated/borders/smooth_marble_border_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/borders/smooth_sandstone_border_$i.png -n 5 -s 15 -c "#8b8c5a" --seed 16518 -o ./assets/textures/templates/generated/borders/smooth_sandstone_border_var_${i}_{i}.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/borders/smooth_granite_border_$i.png -n 5 -s 25 -c "#ba6c78" --seed 315987 -o ./assets/textures/templates/generated/borders/smooth_granite_border_var_${i}_{i}.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/borders/smooth_diorite_border_$i.png -n 5 -s 7 -c "#a864a1" --seed 325872 -o ./assets/textures/templates/generated/borders/smooth_diorite_border_var_${i}_{i}.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/borders/smooth_andesite_border_$i.png -n 5 -s 20 -c "#505c5e" --seed 3942385 -o ./assets/textures/templates/generated/borders/smooth_andesite_border_var_${i}_{i}.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/borders/smooth_limestone_border_$i.png -n 5 -s 20 -c "#5e5943" --seed 3942385 -o ./assets/textures/templates/generated/borders/smooth_limestone_border_var_${i}_{i}.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/templates/generated/borders/smooth_marble_border_$i.png -n 5 -s 20 -c "#f5f0ea" --seed 3942385 -o ./assets/textures/templates/generated/borders/smooth_marble_border_var_${i}_{i}.png
    for j in {1..5}
    do
        ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/templates/generated/smooth_bases/smooth_sandstone_var_16_$j.png ./assets/textures/templates/generated/borders/smooth_sandstone_border_var_${i}_$j.png -o ./assets/textures/smooth_sandstone/smooth_sandstone_var_${i}_$j.png
        ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/templates/generated/smooth_bases/smooth_granite_var_16_$j.png ./assets/textures/templates/generated/borders/smooth_granite_border_var_${i}_$j.png -o ./assets/textures/smooth_granite/smooth_granite_var_${i}_$j.png
        ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/templates/generated/smooth_bases/smooth_diorite_var_16_$j.png ./assets/textures/templates/generated/specks/specks_2_noise_$j.png ./assets/textures/templates/generated/borders/smooth_diorite_border_var_${i}_$j.png -o ./assets/textures/smooth_diorite/smooth_diorite_var_${i}_$j.png
        ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/templates/generated/smooth_bases/smooth_andesite_var_16_$j.png ./assets/textures/templates/generated/borders/smooth_andesite_border_var_${i}_$j.png -o ./assets/textures/smooth_andesite/smooth_andesite_var_${i}_$j.png
        ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/templates/generated/smooth_bases/smooth_limestone_var_16_$j.png ./assets/textures/templates/generated/borders/smooth_limestone_border_var_${i}_$j.png -o ./assets/textures/smooth_limestone/smooth_limestone_var_${i}_$j.png
        ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/templates/generated/smooth_bases/smooth_marble_var_16_$j.png ./assets/textures/templates/generated/borders/smooth_marble_border_var_${i}_$j.png -o ./assets/textures/smooth_marble/smooth_marble_var_${i}_$j.png
    done
done

for i in {1..5}
do
    ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/diorite/diorite_var_$i.png ./assets/textures/templates/generated/specks/specks_2_noise_$i.png -m darken -o ./assets/textures/diorite/diorite_var_$i.png
done

for ore in coal copper iron tin
do
    for stone in stone granite andesite diorite limestone marble
    do
        prefix="${stone}_${ore}_ore"
        if [ "$stone" = "stone" ]; then
            prefix="${ore}_ore"
        fi
        mkdir -p ./assets/textures/${prefix}

        for v in {1..5}
        do
            ./.venv/bin/python3 ./tooling/texture_layer.py ./assets/textures/${stone}/${stone}_var_$v.png ./assets/textures/templates/ores/${ore}_overlay.png -o ./assets/textures/${prefix}/${prefix}_var_$v.png
        done
    done
done
