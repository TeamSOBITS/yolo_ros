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
    point_cloud_topic = LaunchConfiguration("point_cloud_topic")
    model_type = LaunchConfiguration("model_type")
    weight_file = LaunchConfiguration("weight_file")
    init_prediction = LaunchConfiguration("init_prediction")
    image_show = LaunchConfiguration("image_show")
    threshold = LaunchConfiguration("threshold")
    iou = LaunchConfiguration("iou")
    imgsz_height = LaunchConfiguration("imgsz_height")
    imgsz_width = LaunchConfiguration("imgsz_width")
    half = LaunchConfiguration("half")
    max_det = LaunchConfiguration("max_det")
    agnostic_nms = LaunchConfiguration("agnostic_nms")
    retina_masks = LaunchConfiguration("retina_masks")
    namespace = LaunchConfiguration("namespace")
    base_frame_name = LaunchConfiguration("base_frame_name")
    use_3d = LaunchConfiguration("use_3d")

    launch_args = [
        DeclareLaunchArgument(
            "image_topic_name",
            # default_value="/camera/camera/color/image_raw",  # Realsense
            # default_value="/rgb/image_raw",                  # Azure Kinect
            default_value="/camera/rgb/image_raw",             # xtion
            # default_value="/camera/color/image_raw",         # Orbbec
            description="ROS Topic Name of sensor_msgs/msg/Image message",
        ),
        DeclareLaunchArgument(
            "point_cloud_topic",
            # default_value="/camera/camera/depth/color/points",    # Realsense
            # default_value="/points2",                             # Azure Kinect
            default_value="/camera/depth_registered/points",        # xtion
            # default_value="/camera/depth_registered/points",      # Orbbec
            description="ROS Topic Name of sensor_msgs/msg/PointCloud2 message",
        ),
        DeclareLaunchArgument(
            "model_type",
            default_value="YOLO",
            choices=["YOLO", "NAS", "World"],
            description="Model type from Ultralytics (YOLO, NAS, World)",
        ),
        DeclareLaunchArgument(
            "weight_file",
            default_value="yolo11n.pt",             # YOLOv11
            # default_value="yolov8s-worldv2.pt",   # YOLO World
            # default_value="yolo11n-pose.pt",      # KeyPoint model
            # default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),
            description="Weight file path",
        ),
        DeclareLaunchArgument(
            "init_prediction",
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
            default_value="0.5",
            description="Minimum probability of a detection to be published",
        ),
        DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="IoU threshold",
        ),
        DeclareLaunchArgument(
            "imgsz_height",
            default_value="480",
            description="Image height for inference",
        ),
        DeclareLaunchArgument(
            "imgsz_width",
            default_value="640",
            description="Image width for inference",
        ),
        DeclareLaunchArgument(
            "half",
            default_value="False",
            description="Enable FP16 inference",
        ),
        DeclareLaunchArgument(
            "max_det",
            default_value="300",
            description="Maximum number of detections per image",
        ),
        DeclareLaunchArgument(
            "agnostic_nms",
            default_value="False",
            description="Enable class-agnostic NMS",
        ),
        DeclareLaunchArgument(
            "retina_masks",
            default_value="False",
            description="Use high-res segmentation masks if available",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="yolo_ros",
            description="Namespace for the nodes",
        ),
        DeclareLaunchArgument(
            "base_frame_name",
            default_value="camera_rgb_frame",
            description="Base frame name for TF and 3D detection",
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
                "base_frame_name": base_frame_name,
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
            "base_frame_name": base_frame_name,
            "bbox_topic_name": "/yolo_ros/object_boxes",
            "cloud_topic_name": point_cloud_topic,
            "img_topic_name": image_topic_name,
            "execute_default": init_prediction,
            "cluster_tolerance": "0.01",
            "min_clusterSize": "100",
            "max_clusterSize": "20000",
            "noise_point_cloud_range": "0.01",
            "fast_shot": "false",
            "enable_id": "false",
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
            "base_frame_name": base_frame_name,
            "keypoints_topic_name": "/yolo_ros/object_keypoints",
            "cloud_topic_name": point_cloud_topic,
            "img_topic_name": image_topic_name,
            "execute_default": init_prediction,
            "enable_id": "true",
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
