echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

cd ..
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/sobits_interfaces.git
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/image_to_position.git

cd sobits_interfaces/ && bash install.sh && cd ..
cd image_to_position/ && bash install.sh && cd ..

pip3 install torch # TODO: CPU || GPU
pip3 install typing-extensions ultralytics "numpy<2" lap

sudo apt install ros-$ROS_DISTRO-vision-msgs

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"