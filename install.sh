
echo "╔══╣ Install: YOLO ROS (STARTING) ╠══╗"

pip3 install torch
pip3 install typing-extensions
pip3 install ultralytics
pip3 install super-gradients
pip3 install lap


cd ~/colcon_ws/src/
git clone -b feature/humble-devel https://github.com/TeamSOBITS/bbox_to_tf.git
cd bbox_to_tf/
bash install.sh

echo "╚══╣ Install: YOLO ROS (FINISHED) ╠══╝"