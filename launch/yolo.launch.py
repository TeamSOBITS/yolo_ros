import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    model_type_cmd = DeclareLaunchArgument(
        "model_type",
        description="Model type from Ultralytics (YOLO, NAS, World)",
        default_value="YOLO",
        choices=["YOLO", "NAS", "World"],
    )

    # 重みファイルのパス
    weight_file_cmd = DeclareLaunchArgument(
        "weight_file",
        description="Weight file path",
        default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),
        # default_value="yolo11n.pt",     # YOLOv11
    )

    # サブスクライブするトピック設定
    image_topic_name_cmd = DeclareLaunchArgument(
        "image_topic_name",
        description="Image topic name",
        default_value="/rgb/image_raw",                   # azure_kinect
        # default_value="/camera/camera/color/image_raw", # realsense
    )
    point_cloud_topic_cmd = DeclareLaunchArgument(
        "point_cloud_topic",
        description="Point cloud topic name",
        default_value="/points2",                            # azure_kinect
        # default_value="/camera/camera/depth/color/points", # realsense
    )

    # 各種パラメータ
    param_cmds = [
        DeclareLaunchArgument("init_prediction", default_value="True", description="Initial prediction enabled"),
        DeclareLaunchArgument("image_show", default_value="False", description="Show image output"),
        DeclareLaunchArgument("threshold", default_value="0.5", description="Detection threshold"),
        DeclareLaunchArgument("iou", default_value="0.7", description="IoU threshold"),
        DeclareLaunchArgument("imgsz_height", default_value="480", description="Image height for inference"),
        DeclareLaunchArgument("imgsz_width", default_value="640", description="Image width for inference"),
        DeclareLaunchArgument("half", default_value="False", description="Enable half-precision inference"),
        DeclareLaunchArgument("max_det", default_value="300", description="Maximum detections per image"),
        DeclareLaunchArgument("agnostic_nms", default_value="False", description="Enable class-agnostic NMS"),
        DeclareLaunchArgument("retina_masks", default_value="False", description="Use high-resolution segmentation masks"),
    ]

    yolo_node_cmd = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {param.name: LaunchConfiguration(param.name) for param in param_cmds},
            {"image_topic_name": LaunchConfiguration("image_topic_name")},
            {"point_cloud_topic": LaunchConfiguration("point_cloud_topic")},
            {"weight_file": LaunchConfiguration("weight_file")},
        ],
        output="screen"
    )

    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="yolo_ros", description="Namespace for the nodes"),
        model_type_cmd, weight_file_cmd, image_topic_name_cmd, point_cloud_topic_cmd, *param_cmds, yolo_node_cmd
    ])
