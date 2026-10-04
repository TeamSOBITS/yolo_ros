echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"


# Keep track of the current directory
DIR=`pwd`
cd ..

# Download required packages
ros_packages=(
    "sobits_interfaces"
    "image_to_position"
)

#Clone all packages
for ((i = 0; i < ${#ros_packages[@]}; i++)) {
    if [ -d ${ros_packages[i]} ]; then
        echo "${ros_packages[i]} already exists. Skip clone."
    else
        echo "Clonning: ${ros_packages[i]}"
        git clone --recurse-submodules -b $ROS_DISTRO-devel https://github.com/TeamSOBITS/${ros_packages[i]}.git
    fi

    # Check if install.sh exists in each package
    if [ -f ${ros_packages[i]}/install.sh ]; then
        echo "Running install.sh in ${ros_packages[i]}."
        cd ${ros_packages[i]}
        bash install.sh
        cd ..
    fi
}

# Go back to previous directory
cd ${DIR}

python3 -m pip install torch --break-system-packages
python3 -m pip install typing-extensions --break-system-packages
python3 -m pip install ultralytics --break-system-packages
python3 -m pip install 'numpy<2' --break-system-packages
python3 -m pip install lap --break-system-packages
# GUI (yolo_gui)
python3 -m pip install PySide6 --break-system-packages

sudo apt install -y \
    ros-$ROS_DISTRO-vision-msgs \
    v4l-utils

# yolo.launch.py の既定モデル（姿勢推定 + トラッカー ReID）と物体検出モデル．無いと yolo_node が configure できない
echo "Download default weights into weights/ (skipped when already present)..."
(cd weights && python3 -c "from ultralytics.utils.downloads import attempt_download_asset as d; [d(n, release='v8.4.0') for n in ('yolo26m-pose.pt', 'yolo26m.pt', 'yolo26m-reid.onnx')]")


echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"
