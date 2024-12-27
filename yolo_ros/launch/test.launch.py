# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    image_topic_name = LaunchConfiguration("image_topic_name")
    image_topic_name_cmd = DeclareLaunchArgument(
        "image_topic_name",
        default_value="/rgb/image_raw",
        description="ROS Topic Name of sensor_msgs/msg/Image message",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace", default_value="test_node", description="Namespace for the nodes"
    )

    class_list = os.path.join(
        get_package_share_directory("yolo_ros"),
        "yolo_world_classes",
        "class_list.yaml"
        )

    test_node_cmd = Node(
        package="yolo_ros",
        executable="test",
        name="test_node",
        namespace=namespace,
        parameters=[
            {
                "image_topic_name": image_topic_name,
            },
            class_list,
        ],
        output="screen"
    )

    return LaunchDescription(
        [
            image_topic_name_cmd,
            namespace_cmd,
            test_node_cmd,
        ]
    )
