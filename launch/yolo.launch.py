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
    depth_image_topic_name = LaunchConfiguration("depth_image_topic_name")
    info_topic_name = LaunchConfiguration("info_topic_name")
    positioning_detection_mode_object = LaunchConfiguration("positioning_detection_mode_object")
    positioning_detection_mode_keypoint = LaunchConfiguration("positioning_detection_mode_keypoint")
    model_type = LaunchConfiguration("model_type")
    weight_file = LaunchConfiguration("weight_file")
    execute_default = LaunchConfiguration("execute_default")
    image_show = LaunchConfiguration("image_show")
    threshold = LaunchConfiguration("threshold")
    iou = LaunchConfiguration("iou")
    namespace = LaunchConfiguration("namespace")
    base_frame_name = LaunchConfiguration("base_frame_name")
    use_3d = LaunchConfiguration("use_3d")

    launch_args = [
        DeclareLaunchArgument(
            "image_topic_name",
            description="ROS Topic Name of sensor_msgs/msg/Image message. (sensor_msgs/msg/Image)",
            default_value="/camera/color/image_raw",            ## realsense
            # default_value="/rgb/image_raw",                   ## azure_kinect
            # default_value="/camera/color/image_raw",          ## orbbec_series ##
            # default_value="/camera/rgb/image_raw",            ## xtion
        ),
        DeclareLaunchArgument(
            "point_cloud_topic",
            description="Detection 3D Pose from 2D Pose (sensor_msgs/msg/PointCloud2). if you select the 'point_cloud' in 'positioning_detection_mode'.",
            default_value="/camera/depth/color/points",            ## realsense
            # default_value="/points2",                            ## azure_kinect
            # default_value="/camera/depth_registered/points",     ## orbbec_series ##
            # default_value="/camera/depth_registered/points",     ## xtion
        ),
        DeclareLaunchArgument(
            "depth_image_topic_name",
            description="Detection 3D Pose from 2D Pose (sensor_msgs/msg/Image). if you select the 'depth_image' in 'positioning_detection_mode'.",
            default_value="/camera/depth/image_rect_raw", ## realsense
            # default_value="/depth_to_rgb/image_raw", ## azure_kinect
            # default_value="/camera/depth/image_raw", ## orbbec_series ##
            # default_value="/camera/depth/image_raw",    ## xtion
        ),
        DeclareLaunchArgument(
            "info_topic_name",
            description="Setup the camera info topic name. (sensor_msgs/msg/CameraInfo)",
            default_value="/camera/color/camera_info", ## realsense
            # default_value="/rgb/camera_info", ## azure_kinect
            # default_value="/camera/color/camera_info", ## orbbec_series ##
            # default_value="/camera/rgb/camera_info", ## xtion
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
            default_value="yolo11n.pt",       # YOLOv11
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
            default_value="yolo_ros",
            description="Namespace for the nodes",
        ),
        DeclareLaunchArgument(
            "base_frame_name",
            default_value="base_footprint",
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
            "base_frame_name": base_frame_name,
            "bbox_topic_name": "/yolo_ros/object_boxes",
            "cloud_topic_name": point_cloud_topic,
            "depth_image_topic_name": depth_image_topic_name,
            "info_topic_name": info_topic_name,
            "execute_default": execute_default,
            "cluster_tolerance": "0.01",
            "min_clusterSize": "200",
            "max_clusterSize": "20000",
            "noise_point_cloud_range": "0.03",
            "enable_id": "False",
            "positioning_detection_mode": positioning_detection_mode_object,
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
            "depth_image_topic_name": depth_image_topic_name,
            "info_topic_name": info_topic_name,
            "execute_default": execute_default,
            "enable_id": "False",
            "positioning_detection_mode": positioning_detection_mode_keypoint,
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
