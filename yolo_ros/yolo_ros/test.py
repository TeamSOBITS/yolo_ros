import rclpy
from rclpy.node import Node


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")
        self.declare_parameter("classes", [""])
        self.declare_parameter("image_topic_name", "image_raw")

        self.classes = self.get_parameter("classes").get_parameter_value().string_array_value
        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        print("\033[31m================")
        print(type(self.classes))
        print(self.classes)
        print(self.image_topic_name)
        print("================\033[0m\n")


def main():
    rclpy.init()
    node = TestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
