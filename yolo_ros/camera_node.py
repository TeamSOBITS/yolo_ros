"""yolo_camera: publish a V4L2 (internal/USB) camera with OpenCV at a chosen size and rate."""

import sys
import threading

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera")
        self.declare_parameter("video_device", "/dev/video0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera")

        device = self.get_parameter("video_device").value
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        self.fps = max(0.5, float(self.get_parameter("fps").value))
        self.frame_id = self.get_parameter("frame_id").value

        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open {device}")
        # MJPG gives the full frame rate at high resolutions on most USB/internal cameras.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, "image_raw", 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        # Read continuously so that publishing below the camera rate never serves old frames.
        self.reader = threading.Thread(target=self._read_frames, daemon=True)
        self.reader.start()
        self.create_timer(1.0 / self.fps, self._publish)
        actual = (
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.get_logger().info(f"{device}: {actual[0]}x{actual[1]}, publishing at {self.fps:g} Hz")

    def _read_frames(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame

    def _publish(self):
        with self.lock:
            frame, self.frame = self.frame, None
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def close(self):
        self.running = False
        self.reader.join(timeout=1.0)
        self.capture.release()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = CameraNode()
    except RuntimeError as exc:
        print(f"yolo_camera: {exc}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
