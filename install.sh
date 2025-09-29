echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

cd ..
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/sobits_interfaces.git
git clone -b ${ROS_DISTRO}-devel https://github.com/TeamSOBITS/bbox_to_tf.git

cd sobits_interfaces/ && bash install.sh && cd ..
cd bbox_to_tf/ && bash install.sh && cd ..

pip3 install torch ultralytics typing-extensions lap numpy==1.24.2

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"