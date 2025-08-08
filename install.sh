echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

cd ..
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/sobits_interfaces.git
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/bbox_to_tf.git
cd bbox_to_tf/
bash install.sh
cd ..

pip3 install torch
pip3 install typing-extensions
pip3 install ultralytics
pip3 install numpy==1.24.2
pip3 install lap

sudo apt install ros-$ROS_DISTRO-vision-msgs

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"
