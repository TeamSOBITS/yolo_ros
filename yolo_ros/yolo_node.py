import os
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from sobits_interfaces.msg import KeyPointArray, DetectMaskArray

from ultralytics import YOLO

class YoloNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("yolo_ros")

        self.declare_parameter("image_topic_name", "camera/color/image_raw")
        self.declare_parameter("weight_file", "yolo26n.pt")
        self.declare_parameter("weights_path", "")
        self.declare_parameter("execute_default", True)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.7)

        self.declare_parameter("filter_classes", [""])
        self.declare_parameter("keypoint_name_list", [""])
        self.declare_parameter("yoloe_prompts", [""])

        self.cv_bridge = CvBridge()
        self.model = None

        self._pub_img = None
        self._pub_rect = None
        self._pub_keypoint = None
        self._pub_mask = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        self.weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
        self.weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        self.enable = self.get_parameter("execute_default").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value

        self.filter_classes = self.get_parameter("filter_classes").get_parameter_value().string_array_value
        self.keypoint_name_list = self.get_parameter("keypoint_name_list").get_parameter_value().string_array_value
        self.yoloe_prompts = self.get_parameter("yoloe_prompts").get_parameter_value().string_array_value

        self.get_logger().info("(YOLO Parameters)")
        self.get_logger().info(f"Image Topic    : {self.image_topic_name}")
        self.get_logger().info(f"Weight File    : {self.weight_file}")
        self.get_logger().info(f"Execute Default: {self.enable}")
        self.get_logger().info(f"Conf           : {self.conf}")
        self.get_logger().info(f"IoU            : {self.iou}")
        self.get_logger().info(f"Filter Classes : {self.filter_classes}")
        self.get_logger().info(f"YOLOE Prompts  : {self.yoloe_prompts}")

        try:
            model_full_path = os.path.join(self.weights_path, self.weight_file)
            self.model = YOLO(model_full_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return TransitionCallbackReturn.FAILURE

        self._pub_img = self.create_lifecycle_publisher(Image, self.get_name() + "/detected_image", 1)
        self._pub_rect = self.create_lifecycle_publisher(Detection2DArray, self.get_name() + "/object_boxes", 1)
        self._pub_keypoint = self.create_lifecycle_publisher(KeyPointArray, self.get_name() + "/object_keypoints", 1)
        self._pub_mask = self.create_lifecycle_publisher(DetectMaskArray, self.get_name() + "/object_masks", 1)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.destroy_lifecycle_publisher(self._pub_img)
        self.destroy_lifecycle_publisher(self._pub_rect)
        self.destroy_lifecycle_publisher(self._pub_keypoint)
        self.destroy_lifecycle_publisher(self._pub_mask)
        self.model = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    execute_default = node.get_parameter("execute_default").get_parameter_value().bool_value
    if execute_default:
        node.trigger_configure()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()