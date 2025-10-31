echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

cd ..
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/sobits_interfaces.git
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/bbox_to_tf.git
cd bbox_to_tf/
bash install.sh
cd ..

pip3 install torch --break-system-packages
pip3 install typing-extensions --break-system-packages
pip3 install ultralytics --break-system-packages
pip3 install numpy==1.24.2 --break-system-packages
pip3 install lap --break-system-packages

sudo apt install ros-$ROS_DISTRO-vision-msgs

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"
