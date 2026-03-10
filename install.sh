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
    echo "Clonning: ${ros_packages[i]}"
    git clone --recurse-submodules -b $ROS_DISTRO-devel https://github.com/TeamSOBITS/${ros_packages[i]}.git

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

sudo apt install -y \
    ros-$ROS_DISTRO-vision-msgs


echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"
