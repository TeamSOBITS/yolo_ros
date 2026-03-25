#!/bin/bash
echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

sudo apt update
sudo apt install ros-noetic-vision-msgs

python3 -m pip install ultralytics

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"
