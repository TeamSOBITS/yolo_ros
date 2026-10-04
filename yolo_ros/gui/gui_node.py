"""The GUI's ROS node: camera frames, YOLO's drawing, detections, stats, model info and the log."""

from collections import deque
import json
import os
import threading
import time

from cv_bridge import CvBridge
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from yolo_ros.gui.controller import SystemController
from yolo_ros.gui.system import default_data_root

ALIVE_TIMEOUT_SEC = 2.0
LATCHED = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class RateMeter:
    """Messages per second over the last couple of seconds."""

    def __init__(self, window_sec=2.0):
        self.window_sec = window_sec
        self.stamps = deque()

    def tick(self):
        now = time.monotonic()
        self.stamps.append(now)
        while self.stamps and now - self.stamps[0] > self.window_sec:
            self.stamps.popleft()

    def hz(self):
        now = time.monotonic()
        return sum(1 for stamp in self.stamps if now - stamp <= self.window_sec) / self.window_sec


def subscribe_images(node, topic, callback):
    """
    Subscribe with reliable and best-effort QoS so that any camera works; drop duplicates.

    A best-effort reader loses most large frames from a reliable camera, and a reliable reader
    cannot match a best-effort camera at all.
    """
    recent = deque(maxlen=16)
    lock = threading.Lock()

    def deliver(msg):
        key = (msg.header.frame_id, msg.header.stamp.sec, msg.header.stamp.nanosec)
        if key[1] or key[2]:
            with lock:
                if key in recent:
                    return
                recent.append(key)
        callback(msg)

    quiet = SubscriptionEventCallbacks(incompatible_qos=lambda event: None)
    return [
        node.create_subscription(
            Image,
            topic,
            deliver,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=reliability),
            event_callbacks=quiet,
        )
        for reliability in (ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT)
    ]


class YoloGuiNode(Node):
    def __init__(self):
        super().__init__("yolo_gui")
        self.declare_parameter("detector_node", "yolo_node")
        self.declare_parameter("data_root", default_data_root())
        detector = self.get_parameter("detector_node").value
        self.data_root = os.path.expanduser(self.get_parameter("data_root").value)

        self.bridge = CvBridge()
        self.data_lock = threading.Lock()
        self.camera_frame = None
        self.yolo_frame = None
        self.last_image_time = 0.0
        self.last_boxes_time = 0.0
        self.last_stats_time = 0.0
        self.detections = []
        self.stats = {}
        self.yolo_model_info = {}
        self.meters = {"camera": RateMeter(), "yolo": RateMeter()}
        self.messages = deque(maxlen=200)
        self.image_topic = None
        self.image_subscriptions = []
        self.yolo_image_subscription = None
        self.yolo_image_topic = f"{detector}/detected_image"

        self.create_subscription(
            Detection2DArray, f"{detector}/object_boxes", self._boxes_callback, 10
        )
        self.create_subscription(String, f"{detector}/stats", self._stats_callback, 10)
        self.create_subscription(
            String, f"{detector}/model_info", self._model_info_callback, LATCHED
        )
        self.system = SystemController(
            self,
            detector_node=detector,
            settings_path=os.path.join(self.data_root, "gui_settings.json"),
            image_topic_listener=self.set_image_topic,
        )

    # ------------------------------------------------------------------ subscriptions

    def set_image_topic(self, topic):
        """Subscribe to the camera topic, or stop receiving frames when topic is None."""
        for subscription in self.image_subscriptions:
            self.destroy_subscription(subscription)
        with self.data_lock:
            self.image_topic = topic
            self.camera_frame = None
            self.last_image_time = 0.0
        self.image_subscriptions = (
            subscribe_images(self, topic, self._image_callback) if topic else []
        )

    def show_yolo_image(self, enabled):
        """
        Subscribe to YOLO's drawing only while a pane shows it.

        yolo_node skips drawing while nobody subscribes, which saves several ms per frame.
        """
        if enabled and self.yolo_image_subscription is None:
            self.yolo_image_subscription = self.create_subscription(
                Image,
                self.yolo_image_topic,
                self._yolo_image_callback,
                QoSProfile(
                    depth=1,
                    history=HistoryPolicy.KEEP_LAST,
                    reliability=ReliabilityPolicy.RELIABLE,
                ),
            )
        elif not enabled and self.yolo_image_subscription is not None:
            self.destroy_subscription(self.yolo_image_subscription)
            self.yolo_image_subscription = None
            with self.data_lock:
                self.yolo_frame = None

    def _image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.meters["camera"].tick()
        with self.data_lock:
            self.camera_frame = (time.monotonic(), frame)
            self.last_image_time = time.monotonic()

    def _yolo_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.data_lock:
            self.yolo_frame = (time.monotonic(), frame)

    def _boxes_callback(self, msg):
        self.meters["yolo"].tick()
        with self.data_lock:
            self.detections = [detection.id for detection in msg.detections]
            self.last_boxes_time = time.monotonic()

    def _stats_callback(self, msg):
        try:
            stats = json.loads(msg.data)
        except ValueError:
            return
        with self.data_lock:
            self.stats = stats
            self.last_stats_time = time.monotonic()

    def _model_info_callback(self, msg):
        try:
            info = json.loads(msg.data)
        except ValueError:
            return
        with self.data_lock:
            self.yolo_model_info = info

    # ------------------------------------------------------------------ queries (GUI thread)

    def frame(self, source):
        """Latest BGR frame for a pane source ('camera' or 'yolo'), or None when stale."""
        with self.data_lock:
            stamped = self.camera_frame if source == "camera" else self.yolo_frame
        if stamped is None or time.monotonic() - stamped[0] > ALIVE_TIMEOUT_SEC:
            return None
        return stamped[1]

    def rates(self):
        return {name: meter.hz() for name, meter in self.meters.items()}

    def camera_alive(self):
        with self.data_lock:
            return time.monotonic() - self.last_image_time < ALIVE_TIMEOUT_SEC

    def detector_alive(self):
        with self.data_lock:
            return time.monotonic() - self.last_boxes_time < ALIVE_TIMEOUT_SEC

    def latest_stats(self):
        with self.data_lock:
            fresh = time.monotonic() - self.last_stats_time < ALIVE_TIMEOUT_SEC + 1.0
            return dict(self.stats) if fresh else {}

    def latest_detections(self):
        with self.data_lock:
            alive = time.monotonic() - self.last_boxes_time < ALIVE_TIMEOUT_SEC
            return list(self.detections) if alive else []

    def model_info(self):
        with self.data_lock:
            return dict(self.yolo_model_info)

    # ------------------------------------------------------------------ log

    def log(self, level, message):
        """Show a message in the GUI log (any thread) and in the node's log."""
        logger = self.get_logger()
        if level == "error":
            logger.error(message)
        elif level == "warn":
            logger.warning(message)
        else:
            logger.info(message)
        with self.data_lock:
            self.messages.append((level, message))

    def pop_messages(self):
        with self.data_lock:
            messages = list(self.messages)
            self.messages.clear()
        return messages
