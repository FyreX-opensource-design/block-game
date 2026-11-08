for i in {0..16}
do
    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/smooth_stone/smooth_stone_$i.png --set-hue 350 -B 1.3 -o ./assets/textures/smooth_granite/smooth_granite_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/smooth_granite/smooth_granite_$i.png -n 5 -s 8 --seed 75 -o ./assets/textures/smooth_granite/smooth_granite_var_$i.png

    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/smooth_stone/smooth_stone_$i.png --tint "#fafafa" --tint-strength 0.5 -B 0.9 -o ./assets/textures/smooth_diorite/smooth_diorite_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/smooth_diorite/smooth_diorite_$i.png -n 5 -s 6 --seed 80 -o ./assets/textures/smooth_diorite/smooth_diorite_var_$i.png

    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/smooth_stone/smooth_stone_$i.png --tint "#80868c" --tint-strength 0.8 -B 0.6 -o ./assets/textures/smooth_andesite/smooth_andesite_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/smooth_andesite/smooth_andesite_$i.png -n 5 -s 7 --seed 25 -o ./assets/textures/smooth_andesite/smooth_andesite_var_$i.png

    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/smooth_stone/smooth_stone_$i.png --tint "#ffffff" --tint-strength 0.8 -B 0.95 -o ./assets/textures/smooth_marble/smooth_marble_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/smooth_marble/smooth_marble_$i.png -n 5 -s 10 -c "#383838" --seed 1 -o ./assets/textures/smooth_marble/smooth_marble_var_$i.png

    ./.venv/bin/python3 ./tooling/texture_adjust.py ./assets/textures/smooth_stone/smooth_stone_$i.png --tint "#f9fad4" --tint-strength 0.5 -B 0.6 -o ./assets/textures/smooth_limestone/smooth_limestone_$i.png
    ./.venv/bin/python3 ./tooling/texture_noise.py ./assets/textures/smooth_limestone/smooth_limestone_$i.png -n 5 -s 15 -c "#e5e6ca" --seed 205 -o ./assets/textures/smooth_limestone/smooth_limestone_var_$i.png
done