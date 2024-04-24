#!/bin/bash

echo "Generating results for $1"

python main.py --config configs/image_sai.yaml input=data/"$1"_rgba.png save_path="$1"
python main3.py --config configs/image_sai.yaml input=data/"$1"_rgba.png save_path="$1" mesh=logs/"$1".obj
kire logs/"$1".obj --save_video "$1"_s2.mp4 --wogui