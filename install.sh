
echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

cd ..
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/sobits_msgs.git
cd sobits_msgs/
bash install.sh
cd ..
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/bbox_to_tf.git
cd bbox_to_tf/
bash install.sh
cd ..

pip3 install torch
pip3 install typing-extensions
pip3 install ultralytics
pip3 install super-gradients
pip3 install lap

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"