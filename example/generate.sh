#!/bin/bash

python3 ../detect.py example-easy.mkv example-easy-output.mkv example-easy-output.txt --codec="X264"
python3 ../detect.py example-hard.mp4 example-hard-output.mkv example-hard-output.txt --codec="X264"
