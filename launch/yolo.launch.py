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
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():

    image_topic_name = LaunchConfiguration("image_topic_name")
    image_topic_name_cmd = DeclareLaunchArgument(
        "image_topic_name",
        description="ROS Topic Name of sensor_msgs/msg/Image message",
        # default_value="/camera/camera/color/image_raw",   ## realsense
        default_value="/rgb/image_raw",                   ## azure_kinect
    )

    point_cloud_topic = LaunchConfiguration("point_cloud_topic")
    point_cloud_topic_cmd = DeclareLaunchArgument(
        "point_cloud_topic",
        description="ROS Topic Name of sensor_msgs/msg/PointCloud2 message",
        # default_value="/camera/camera/depth/color/points",   ## realsense
        default_value="/points2",                            ## azure_kinect
    )

    model_type = LaunchConfiguration("model_type")
    model_type_cmd = DeclareLaunchArgument(
        "model_type",
        default_value="YOLO",
        choices=["YOLO", "NAS", "World"],
        description="Model type form Ultralytics (YOLO, NAS",
    )

    weight_file = LaunchConfiguration("weight_file")
    weight_file_cmd = DeclareLaunchArgument(
        "weight_file", description="weight file path",
        default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),  ## custom weight file
        # default_value="/yolov5mu.pt",         ## YOLOv5
        # default_value="/yolov8m.pt",          ## YOLOv8
        # default_value="/yolov9c.pt",          ## YOLOv9
        # default_value="/yolov10m.pt",         ## YOLOv10
        # default_value="/yolov11m.pt",         ## YOLOv11
        # default_value="/yolo_nas_s.pt",       ## YOLO NAS
        # default_value="/yolov8s-worldv2.pt",  ## YOLO World
    )

    init_prediction = LaunchConfiguration("init_prediction")
    init_prediction_cmd = DeclareLaunchArgument(
        "init_prediction", default_value="True", description="Whether to start YOLO enabled"
    )

    image_show = LaunchConfiguration("image_show")
    image_show_cmd = DeclareLaunchArgument(
        "image_show",
        default_value="False",
        description="image show flag",
    )

    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.5",
        description="Minimum probability of a detection to be published",
    )

    iou = LaunchConfiguration("iou")
    iou_cmd = DeclareLaunchArgument(
        "iou", default_value="0.7", description="IoU threshold"
    )

    imgsz_height = LaunchConfiguration("imgsz_height")
    imgsz_height_cmd = DeclareLaunchArgument(
        "imgsz_height",
        default_value="480",
        description="Image height for inference",
    )

    imgsz_width = LaunchConfiguration("imgsz_width")
    imgsz_width_cmd = DeclareLaunchArgument(
        "imgsz_width", default_value="640", description="Image width for inference"
    )

    half = LaunchConfiguration("half")
    half_cmd = DeclareLaunchArgument(
        "half",
        default_value="False",
        description="Whether to enable half-precision (FP16) inference speeding up model inference with minimal impact on accuracy",
    )

    max_det = LaunchConfiguration("max_det")
    max_det_cmd = DeclareLaunchArgument(
        "max_det",
        default_value="300",
        description="Maximum number of detections allowed per image",
    )

    agnostic_nms = LaunchConfiguration("agnostic_nms")
    agnostic_nms_cmd = DeclareLaunchArgument(
        "agnostic_nms",
        default_value="False",
        description="Whether to enable class-agnostic Non-Maximum Suppression (NMS) merging overlapping boxes of different classes",
    )

    retina_masks = LaunchConfiguration("retina_masks")
    retina_masks_cmd = DeclareLaunchArgument(
        "retina_masks",
        default_value="False",
        description="Whether to use high-resolution segmentation masks if available in the model, enhancing mask quality for segmentation",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace", default_value="yolo_ros", description="Namespace for the nodes"
    )

    class_list = os.path.join(
        get_package_share_directory("yolo_ros"),
        "yolo_world_classes",
        "class_list.yaml"
        )

    yolo_node_cmd = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace=namespace,
        parameters=[
            {
                "model_type": model_type,
                "weight_file": weight_file,
                "init_prediction": init_prediction,
                "image_topic_name": image_topic_name,
                "threshold": threshold,
                "iou": iou,
                "imgsz_height": imgsz_height,
                "imgsz_width": imgsz_width,
                "half": half,
                "max_det": max_det,
                "agnostic_nms": agnostic_nms,
                "retina_masks": retina_masks,
                "image_show": image_show,
            },
            class_list
        ],
        output="screen"
    )

    use_3d = LaunchConfiguration("use_3d")
    use_3d_cmd = DeclareLaunchArgument(
        "use_3d", default_value="True", description="Whether to activate 3D detections"
    )

    bbox_to_3d_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("image_to_position"),
                "launch",
                "bbox_to_3d.launch.py",
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "base_frame_name": "camera_base",
            "bbox_topic_name": "/yolo_ros/object_boxes",
            "cloud_topic_name": point_cloud_topic,
            "img_topic_name": image_topic_name,
            "execute_default": init_prediction,
        }.items(),
        condition=IfCondition(use_3d),  # use_3dがTrueのときのみ実行
    )

    return LaunchDescription(
        [
            use_3d_cmd,
            model_type_cmd,
            weight_file_cmd,
            init_prediction_cmd,
            image_topic_name_cmd,
            point_cloud_topic_cmd,
            threshold_cmd,
            iou_cmd,
            imgsz_height_cmd,
            imgsz_width_cmd,
            half_cmd,
            max_det_cmd,
            agnostic_nms_cmd,
            retina_masks_cmd,
            image_show_cmd,
            namespace_cmd,
            yolo_node_cmd,
            bbox_to_3d_cmd,
        ]
    )
