import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    image_topic_name = LaunchConfiguration("image_topic_name")
    model_type = LaunchConfiguration("model_type")
    weight_file = LaunchConfiguration("weight_file")
    execute_default = LaunchConfiguration("execute_default")
    image_show = LaunchConfiguration("image_show")
    threshold = LaunchConfiguration("threshold")
    iou = LaunchConfiguration("iou")
    namespace = LaunchConfiguration("namespace")
    use_3d = LaunchConfiguration("use_3d")

    launch_args = [
        DeclareLaunchArgument(
            "image_topic_name",
            description="ROS Topic Name of sensor_msgs/msg/Image message. (sensor_msgs/msg/Image)",
            default_value="camera/color/image_raw",            ## realsense
            # default_value="rgb/image_raw",                   ## azure_kinect
            # default_value="camera/color/image_raw",          ## orbbec_series ##
            # default_value="camera/rgb/image_raw",            ## xtion
        ),
        DeclareLaunchArgument(
            "positioning_detection_mode_object",
            description="Select the 3D Pose Detection mode. Choose of ['point_cloud', 'fast_point', 'depth_image']",
            default_value="point_cloud",
        ),
        DeclareLaunchArgument(
            "positioning_detection_mode_keypoint",
            description="Select the 3D Pose Detection mode. Choose of ['point_cloud', 'depth_image']",
            default_value="point_cloud",
        ),
        DeclareLaunchArgument(
            "model_type",
            default_value="YOLO",
            choices=["YOLO", "NAS", "World"],
            description="Model type from Ultralytics (YOLO, NAS, World)",
        ),
        DeclareLaunchArgument(
            "weight_file",
            default_value="yolo26x.pt",       # YOLOv26
            # default_value="yolo11n-pose.pt",  # KeyPoint model
            # default_value="yolo11n-seg.pt",   # Segmentation
            # default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),
            description="Weight file path",
        ),
        DeclareLaunchArgument(
            "execute_default",
            default_value="True",
            description="Whether to start YOLO enabled",
        ),
        DeclareLaunchArgument(
            "image_show",
            default_value="False",
            description="Flag to show image with predictions",
        ),
        DeclareLaunchArgument(
            "threshold",
            default_value="0.35",
            description="Minimum probability of a detection to be published",
        ),
        DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="IoU threshold",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Namespace for the nodes",
        ),
        DeclareLaunchArgument(
            "use_3d",
            default_value="True",
            description="Whether to activate 3D detections",
        ),
    ]

    class_list = os.path.join(
        get_package_share_directory("yolo_ros"),
        "yolo_world_classes",
        "class_list.yaml"
    )

    keypoint_dictionary = os.path.join(
        get_package_share_directory("yolo_ros"),
        "keypoints",
        "key_point_dictionary.yaml"
    )

    yolo_node_cmd = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_ros",
        namespace=namespace,
        parameters=[
            {
                "model_type": model_type,
                "weight_file": weight_file,
                "execute_default": execute_default,
                "image_topic_name": image_topic_name,
                "threshold": threshold,
                "iou": iou,
                "imgsz_height": 480,
                "imgsz_width": 640,
                "half": False,
                "max_det": 300,
                "agnostic_nms": False,
                "retina_masks": False,
                "image_show": image_show,
            },
            class_list,
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
