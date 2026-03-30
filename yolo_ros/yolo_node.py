import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from sobits_interfaces.msg import KeyPointArray, DetectMaskArray

class YoloNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("yolo_ros")

        self.declare_parameter("image_topic_name", "camera/color/image_raw")
        self.declare_parameter("weight_file", "yolo26n.pt")
        self.declare_parameter("weights_path", "")
        self.declare_parameter("execute_default", True)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.7)

        self.cv_bridge = CvBridge()

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

        self.get_logger().info(f"Image topic name: {self.image_topic_name}")
        self.get_logger().info(f"Weight file: {self.weight_file}")
        self.get_logger().info(f"Weights path: {self.weights_path}")
        self.get_logger().info(f"Execute default: {self.enable}")
        self.get_logger().info(f"Confidence threshold: {self.conf}")
        self.get_logger().info(f"IoU threshold: {self.iou}")

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
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()