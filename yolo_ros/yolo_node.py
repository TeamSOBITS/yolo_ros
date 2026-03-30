import os
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from ament_index_python.packages import get_package_share_directory

class YoloNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("yolo_ros")

        self.declare_parameter("image_topic_name", "camera/color/image_raw")
        self.declare_parameter("weight_file", "yolo26n.pt")
        self.declare_parameter("weights_path", os.path.join(get_package_share_directory("yolo_ros"), "weights"))
        self.declare_parameter("execute_default", True)
        self.declare_parameter("threshold", 0.35)
        self.declare_parameter("iou", 0.7)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        self.weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
        self.weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        self.enable = self.get_parameter("execute_default").get_parameter_value().bool_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value

        self.get_logger().info(f"Image topic name: {self.image_topic_name}")
        self.get_logger().info(f"Weight file: {self.weight_file}")
        self.get_logger().info(f"Weights path: {self.weights_path}")
        self.get_logger().info(f"Execute default: {self.enable}")
        self.get_logger().info(f"Threshold: {self.threshold}")
        self.get_logger().info(f"IoU: {self.iou}")

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