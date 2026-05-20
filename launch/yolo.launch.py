import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, AndSubstitution, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    image_topic_name = LaunchConfiguration("image_topic_name")
    weight_file = LaunchConfiguration("weight_file")
    weights_path = LaunchConfiguration("weights_path")
    bbox_to_3d_params_file = LaunchConfiguration("bbox_to_3d_params_file")
    keypoint_to_3d_params_file = LaunchConfiguration("keypoint_to_3d_params_file")
    mask_to_3d_params_file = LaunchConfiguration("mask_to_3d_params_file")
    execute_default = LaunchConfiguration("execute_default")
    conf = LaunchConfiguration("conf")
    iou = LaunchConfiguration("iou")
    use_bbox_to_3d = LaunchConfiguration("use_bbox_to_3d")
    use_keypoint_to_3d = LaunchConfiguration("use_keypoint_to_3d")
    use_mask_to_3d = LaunchConfiguration("use_mask_to_3d")

    launch_args = [
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Namespace for the nodes",
        ),
        DeclareLaunchArgument(
            "node_name",
            default_value="yolo_node",
            description="Name of the YOLO node",
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
            # default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),
            description="Weight file name",
        ),
        DeclareLaunchArgument(
            "weights_path",
            default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights"),
            description="Directory path where weight files are stored",
        ),
        DeclareLaunchArgument(
            "bbox_to_3d_params_file",
            default_value=PathJoinSubstitution([FindPackageShare("image_to_position"), "config", "bbox_to_3d.yaml"]),
            description="Parameter file path for bbox_to_3d",
        ),
        DeclareLaunchArgument(
            "keypoint_to_3d_params_file",
            default_value=PathJoinSubstitution([FindPackageShare("image_to_position"), "config", "keypoint_to_3d.yaml"]),
            description="Parameter file path for keypoint_to_3d",
        ),
        DeclareLaunchArgument(
            "mask_to_3d_params_file",
            default_value=PathJoinSubstitution([FindPackageShare("image_to_position"), "config", "mask_to_3d.yaml"]),
            description="Parameter file path for mask_to_3d",
        ),
        DeclareLaunchArgument(
            "execute_default",
            default_value="True",
            description="Whether to auto-configure and auto-activate the YOLO lifecycle node",
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
            "use_bbox_to_3d",
            default_value="True",
            description="Whether to activate bbox_to_3d",
        ),
        DeclareLaunchArgument(
            "use_keypoint_to_3d",
            default_value="True",
            description="Whether to activate keypoint_to_3d",
        ),
        DeclareLaunchArgument(
            "use_mask_to_3d",
            default_value="False",
            description="Whether to activate mask_to_3d",
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
        name=LaunchConfiguration("node_name", default="yolo_node"),
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
            PathJoinSubstitution([FindPackageShare("image_to_position"), "launch", "bbox_to_3d.launch.py"])
        ),
        launch_arguments={
            "namespace": namespace,
            "params_file": bbox_to_3d_params_file,
            "execute_default": execute_default,
            "bbox_topic_name": [LaunchConfiguration("node_name"), "/object_boxes"],
        }.items(),
        condition=IfCondition(use_bbox_to_3d),
    )

    keypoint_to_3d_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("image_to_position"), "launch", "keypoint_to_3d.launch.py"])
        ),
        launch_arguments={
            "namespace": namespace,
            "params_file": keypoint_to_3d_params_file,
            "execute_default": execute_default,
            "keypoint_topic_name": [LaunchConfiguration("node_name"), "/object_keypoints"],
        }.items(),
        condition=IfCondition(use_keypoint_to_3d),
    )

    mask_to_3d_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("image_to_position"), "launch", "mask_to_3d.launch.py"])
        ),
        launch_arguments={
            "namespace": namespace,
            "params_file": mask_to_3d_params_file,
            "execute_default": execute_default,
            "mask_topic_name": [LaunchConfiguration("node_name"), "/object_masks"],
        }.items(),
        condition=IfCondition(use_mask_to_3d),
    )

    return LaunchDescription(
        launch_args + [
            yolo_node_cmd,
            bbox_to_3d_cmd,
            keypoint_to_3d_cmd,
            mask_to_3d_cmd,
        ]
    )
