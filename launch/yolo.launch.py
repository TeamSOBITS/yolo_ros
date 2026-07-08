import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    auto_configure_2d = LaunchConfiguration("auto_configure_2d")
    auto_activate_2d = LaunchConfiguration("auto_activate_2d")
    auto_configure_3d = LaunchConfiguration("auto_configure_3d")
    auto_activate_3d = LaunchConfiguration("auto_activate_3d")
    conf = LaunchConfiguration("conf")
    iou = LaunchConfiguration("iou")
    mode = LaunchConfiguration("mode")
    tracker = LaunchConfiguration("tracker")
    tracker_with_reid = LaunchConfiguration("tracker_with_reid")
    tracker_reid_model = LaunchConfiguration("tracker_reid_model")
    tracker_reid_weights_path = LaunchConfiguration("tracker_reid_weights_path")
    draw_trails = LaunchConfiguration("draw_trails")
    use_detection_filter = LaunchConfiguration("use_detection_filter")
    image_reliability = LaunchConfiguration("image_reliability")
    device = LaunchConfiguration("device")
    fuse = LaunchConfiguration("fuse")
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
            # default_value="camera/color/image_raw",          ## realsense
            # default_value="rgb/image_raw",                   ## azure_kinect
            # default_value="camera/color/image_raw",          ## orbbec_series ##
            # default_value="camera/rgb/image_raw",            ## xtion
            default_value="/image_raw",                        ## 内部カメラ

        ),
        DeclareLaunchArgument(
            "weight_file",
            # default_value="yolo26m.pt",       # YOLOv26
            default_value="yolo26m-pose.pt",  # KeyPoint model
            # default_value="yolo26m-seg.pt",   # Segmentation
            # default_value="yolo26m-sem.pt",   # Semantic Segmentation
            # default_value="yoloe-26m-seg.pt", # YOLO-E segmentation
            description="Weight file name",
        ),
        DeclareLaunchArgument(
            "tracker_reid_model",
            default_value="yolo26m-reid.onnx",
            description=(
                "ReID model filename or path. Supports explicit files such as .onnx, .engine, "
                ".torchscript, .openvino, or .pt when tracker_with_reid is true"
            ),
        ),
        DeclareLaunchArgument(
            "weights_path",
            default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights"),
            description="Directory path where weight files are stored",
        ),
        DeclareLaunchArgument(
            "tracker_reid_weights_path",
            default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights"),
            description="Directory path where tracker ReID model files are stored",
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
            "auto_configure_2d",
            default_value="true",
            description="Whether to configure the YOLO lifecycle node on startup",
        ),
        DeclareLaunchArgument(
            "auto_activate_2d",
            default_value="true",
            description="Whether to activate the YOLO lifecycle node on startup",
        ),
        DeclareLaunchArgument(
            "auto_configure_3d",
            default_value="false",
            description="Whether to configure the Image to Position lifecycle node on startup",
        ),
        DeclareLaunchArgument(
            "auto_activate_3d",
            default_value="false",
            description="Whether to activate the Image to Position lifecycle node on startup",
        ),
        DeclareLaunchArgument(
            "conf",
            default_value="0.5",
            description="Minimum probability of a detection to be published",
        ),
        DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="IoU threshold",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="track",
            description="Inference mode: 'detect' uses predict(), 'track' uses track()",
        ),
        DeclareLaunchArgument(
            "tracker",
            default_value="tracktrack.yaml",
            description=(
                "Ultralytics tracker config: botsort.yaml, bytetrack.yaml, "
                "ocsort.yaml, deepocsort.yaml, fasttrack.yaml, or tracktrack.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "tracker_with_reid",
            default_value="true",
            description="Whether to enable ReID for BoT-SORT, Deep OC-SORT, or TrackTrack",
        ),
        DeclareLaunchArgument(
            "draw_trails",
            default_value="true",
            description="Whether to draw trajectory trails for tracked objects",
        ),
        DeclareLaunchArgument(
            "use_detection_filter",
            default_value="true",
            description="Use filter_classes from detection_filters.yaml",
        ),
        DeclareLaunchArgument(
            "image_reliability",
            default_value="best_effort",
            description="QoS reliability for the image subscription: 'best_effort', 'reliable', 'system_default', 'best_available', or 'unknown'",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Inference device: 'cuda', 'cpu', or 'cuda:0'",
        ),
        DeclareLaunchArgument(
            "fuse",
            default_value="true",
            description="Fuse Conv+BN layers after load for faster inference",
        ),
        DeclareLaunchArgument(
            "use_bbox_to_3d",
            default_value="false",
            description="Whether to activate bbox_to_3d",
        ),
        DeclareLaunchArgument(
            "use_keypoint_to_3d",
            default_value="false",
            description="Whether to activate keypoint_to_3d",
        ),
        DeclareLaunchArgument(
            "use_mask_to_3d",
            default_value="false",
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
                "auto_configure": auto_configure_2d,
                "auto_activate": auto_activate_2d,
                "conf": conf,
                "iou": iou,
                "yolo_mode": mode,
                "use_tracking": ParameterValue(
                    PythonExpression(["'", mode, "' == 'track'"]),
                    value_type=bool,
                ),
                "tracker": tracker,
                "tracker_with_reid": tracker_with_reid,
                "tracker_reid_model": tracker_reid_model,
                "tracker_reid_weights_path": tracker_reid_weights_path,
                "draw_trails": draw_trails,
                "use_detection_filter": use_detection_filter,
                "image_reliability": image_reliability,
                "device": device,
                "fuse": fuse,
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
            "auto_configure": auto_configure_3d,
            "auto_activate": auto_activate_3d,
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
            "auto_configure": auto_configure_3d,
            "auto_activate": auto_activate_3d,
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
            "auto_configure": auto_configure_3d,
            "auto_activate": auto_activate_3d,
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
