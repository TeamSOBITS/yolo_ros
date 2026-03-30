import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    image_topic_name = LaunchConfiguration("image_topic_name")
    weight_file = LaunchConfiguration("weight_file")
    weights_path = LaunchConfiguration("weights_path")
    execute_default = LaunchConfiguration("execute_default")
    conf = LaunchConfiguration("conf")
    iou = LaunchConfiguration("iou")
    use_3d = LaunchConfiguration("use_3d")

    launch_args = [
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Namespace for the nodes",
        ),
        DeclareLaunchArgument(
            "image_topic_name",
            description="ROS Topic Name of sensor_msgs/msg/Image message. (sensor_msgs/msg/Image)",
            # default_value="camera/color/image_raw",            ## realsense
            # default_value="rgb/image_raw",                   ## azure_kinect
            # default_value="camera/color/image_raw",          ## orbbec_series ##
            default_value="camera/rgb/image_raw",            ## xtion
        ),
        DeclareLaunchArgument(
            "weight_file",
            # default_value="yolo26n.pt",       # YOLOv26
            default_value="yolo26n-pose.pt",  # KeyPoint model
            # default_value="yolo26n-seg.pt",   # Segmentation
            # default_value="yoloe-26n-seg.pt",   # YOLOE
            # default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),
            description="Weight file name",
        ),
        DeclareLaunchArgument(
            "weights_path",
            default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights"),
            description="Directory path where weight files are stored",
        ),
        DeclareLaunchArgument(
            "execute_default",
            default_value="True",
            description="Whether to start YOLO enabled",
        ),
        DeclareLaunchArgument(
            "conf",
            default_value="0.35",
            description="Minimum probability of a detection to be published",
        ),
        DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="IoU threshold",
        ),
        DeclareLaunchArgument(
            "use_3d",
            default_value="True",
            description="Whether to activate 3D detections",
        ),
    ]

    yoloe_prompts = os.path.join(
        get_package_share_directory("yolo_ros"),
        "config",
        "yoloe_prompts.yaml"
    )

    detection_filters = os.path.join(
        get_package_share_directory("yolo_ros"),
        "config",
        "detection_filters.yaml"
    )

    keypoint_dictionary = os.path.join(
        get_package_share_directory("yolo_ros"),
        "config",
        "key_point_dictionary.yaml"
    )

    yolo_node_cmd = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_ros",
        namespace=namespace,
        parameters=[
            {
                "image_topic_name": image_topic_name,
                "weight_file": weight_file,
                "weights_path": weights_path,
                "execute_default": execute_default,
                "conf": conf,
                "iou": iou,
            },
            yoloe_prompts,
            detection_filters,
            keypoint_dictionary,
        ],
        output="screen"
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
            "params_file": os.path.join(
                get_package_share_directory("image_to_position"), "config",
                "bbox_to_3d.yaml"
            ),
            "execute_default": execute_default,
        }.items(),
        condition=IfCondition(use_3d),
    )

    keypoint_to_3d_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("image_to_position"),
                "launch",
                "keypoint_to_3d.launch.py",
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "params_file": os.path.join(
                get_package_share_directory("image_to_position"), "config",
                "keypoint_to_3d.yaml"
            ),
            "execute_default": execute_default,
        }.items(),
        condition=IfCondition(use_3d),
    )

    return LaunchDescription(
        launch_args + [
            yolo_node_cmd,
            bbox_to_3d_cmd,
            keypoint_to_3d_cmd,
        ]
    )