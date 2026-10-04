from collections import deque
import gc
import json
import os
import time

from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sobits_interfaces.msg import DetectMaskArray, KeyPointArray
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from vision_msgs.msg import Detection2DArray
from yolo_ros import keypoints, tracking
from yolo_ros.ros_conversion import build_detections, build_keypoints, build_masks, TrailDrawer
from yolo_ros.stats import inference_ms, ProcessingStats


def _precision_argument(half):
    """FP16 or FP32 inference; Ultralytics >= 8.4 replaced the deprecated `half` by `quantize`."""
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16 if half else None}
    return {"half": half}


class YoloNode(LifecycleNode):
    _MODE_CHOICES = ("detect", "track")
    _TRAIL_MODE_CHOICES = ("false", "bbox", "keypoint", "all")

    _RELIABILITY_MAP = {
        "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
        "reliable": QoSReliabilityPolicy.RELIABLE,
        "system_default": QoSReliabilityPolicy.SYSTEM_DEFAULT,
        "best_available": QoSReliabilityPolicy.BEST_AVAILABLE,
        "unknown": QoSReliabilityPolicy.UNKNOWN,
    }
    _RELIABILITY_CHOICES = ("auto", *_RELIABILITY_MAP)

    # Changing these reloads the model or tracker, so they are only accepted while inactive.
    _VALIDATED_TOGETHER = (
        "use_person_keypoint_filter",
        "keypoint_publish_names",
        "person_keypoint_filter_names",
        "keypoint_trail_names",
        "trail_mode",
    )
    _INACTIVE_ONLY_PARAMS = (
        "weight_file",
        "weights_path",
        "fuse",
        "tracker",
        "tracker_with_reid",
        "tracker_reid_model",
        "tracker_reid_weights_path",
        "image_reliability",
    )

    def __init__(self) -> None:
        super().__init__("yolo_ros")

        self.declare_parameter("image_topic_name", "camera/color/image_raw")
        self.declare_parameter("weight_file", "yolo26n.pt")
        self.declare_parameter("weights_path", "")
        self.declare_parameter("auto_configure", True)
        self.declare_parameter("auto_activate", True)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("yolo_mode", "detect")
        self.declare_parameter("use_tracking", False)
        self.declare_parameter("tracker", "tracktrack.yaml")
        self.declare_parameter("tracker_with_reid", True)
        self.declare_parameter("tracker_reid_model", "yolo26m-reid.onnx")
        self.declare_parameter("tracker_reid_weights_path", "")
        self.declare_parameter("trail_mode", "bbox")
        self.declare_parameter("trail_length", 30)
        self.declare_parameter("line_width", 2)
        self.declare_parameter("use_detection_filter", True)
        self.declare_parameter("use_person_keypoint_filter", False)

        self.declare_parameter("filter_classes", [""])
        self.declare_parameter("keypoint_publish_names", [""])
        self.declare_parameter("person_keypoint_filter_names", [""])
        self.declare_parameter("keypoint_trail_names", [""])
        self.declare_parameter("yoloe_prompts", [""])
        self.declare_parameter("image_reliability", "auto")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("fuse", True)
        self.declare_parameter("max_rate_hz", 0.0)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("half", False)

        self._cv_bridge = CvBridge()
        self._predictor = None
        self._image_subs = []
        self._recent_image_keys = deque(maxlen=16)
        self._detect_request_sub = None

        self._pub_img = None
        self._pub_rect = None
        self._pub_keypoint = None
        self._pub_mask = None
        self._pub_detect_response = None
        self._pub_stats = None
        self._stats_timer = None
        self.stats = ProcessingStats()

        self.image_topic_name = ""
        self.weight_file = ""
        self.weights_path = ""
        self.conf = 0.35
        self.iou = 0.7
        self.yolo_mode = "detect"
        self.use_tracking = False
        self.tracker = "tracktrack.yaml"
        self.tracker_with_reid = True
        self.tracker_reid_model = "yolo26m-reid.onnx"
        self.tracker_reid_weights_path = ""
        self._tracker_runtime_config = "tracktrack.yaml"
        self._tracker_runtime_config_path = None
        self.trail_mode = "bbox"
        self.trail_length = 30
        self.line_width = 2
        self.use_detection_filter = True
        self.use_person_keypoint_filter = False
        self.filter_classes = [""]
        self.keypoint_publish_names = [""]
        self.person_keypoint_filter_names = [""]
        self.keypoint_trail_names = [""]
        self.yoloe_prompts = [""]
        self.image_reliability = "auto"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fuse = True
        self.max_rate_hz = 0.0
        self.imgsz = 640
        self.half = False
        self._last_processed = 0.0
        self.trails = TrailDrawer()

        self._param_cb = self.add_on_set_parameters_callback(self.parameters_callback)
        # What the loaded model can detect, for GUIs choosing filter_classes / keypoint settings.
        # Latched (transient local) and published on every model load, even while inactive.
        self._pub_model_info = self.create_publisher(
            String,
            self.get_name() + "/model_info",
            QoSProfile(
                depth=1,
                history=QoSHistoryPolicy.KEEP_LAST,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._param_cb_registered = True

    # ------------------------------------------------------------------ lifecycle

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configure...")

        self.image_topic_name = (
            self.get_parameter("image_topic_name").get_parameter_value().string_value
        )
        self.weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
        self.weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.yolo_mode = self.get_parameter("yolo_mode").get_parameter_value().string_value
        self.use_tracking = self.get_parameter("use_tracking").get_parameter_value().bool_value
        if self.yolo_mode not in self._MODE_CHOICES:
            self.yolo_mode = "track" if self.use_tracking else "detect"
        self.use_tracking = self.yolo_mode == "track"
        self.tracker = self.get_parameter("tracker").get_parameter_value().string_value
        self.tracker_with_reid = (
            self.get_parameter("tracker_with_reid").get_parameter_value().bool_value
        )
        self.tracker_reid_model = (
            self.get_parameter("tracker_reid_model").get_parameter_value().string_value or "auto"
        )
        self.tracker_reid_weights_path = (
            self.get_parameter("tracker_reid_weights_path").get_parameter_value().string_value
        ) or self.weights_path
        self.trail_mode = self.get_parameter("trail_mode").get_parameter_value().string_value
        self.trail_length = self.get_parameter("trail_length").get_parameter_value().integer_value
        self.line_width = self.get_parameter("line_width").get_parameter_value().integer_value
        self.use_detection_filter = (
            self.get_parameter("use_detection_filter").get_parameter_value().bool_value
        )
        self.use_person_keypoint_filter = (
            self.get_parameter("use_person_keypoint_filter").get_parameter_value().bool_value
        )

        self.filter_classes = self._get_string_array_parameter("filter_classes")
        self.keypoint_publish_names = self._get_string_array_parameter("keypoint_publish_names")
        self.person_keypoint_filter_names = self._get_string_array_parameter(
            "person_keypoint_filter_names"
        )
        self.keypoint_trail_names = self._get_string_array_parameter("keypoint_trail_names")
        self.yoloe_prompts = self._get_string_array_parameter("yoloe_prompts")
        self.image_reliability = (
            self.get_parameter("image_reliability").get_parameter_value().string_value
        )
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse = self.get_parameter("fuse").get_parameter_value().bool_value
        self.max_rate_hz = self.get_parameter("max_rate_hz").get_parameter_value().double_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value

        error = self._validate_configuration()
        if error:
            self.get_logger().error(error)
            return TransitionCallbackReturn.FAILURE

        self._log_configuration()

        if not self.load_model():
            return TransitionCallbackReturn.FAILURE
        if not self._prepare_tracker_runtime_config():
            return TransitionCallbackReturn.FAILURE

        self._warn_keypoint_names()
        keypoint_valid, keypoint_reason = self._validate_keypoint_settings()
        if not keypoint_valid:
            self.get_logger().error(keypoint_reason)
            return TransitionCallbackReturn.FAILURE
        self._warn_filter_classes_ignored()
        self._warn_keypoint_settings_ignored()
        self._log_filter_class_ids()

        self._pub_img = self.create_lifecycle_publisher(
            Image, self.get_name() + "/detected_image", 1
        )
        self._pub_rect = self.create_lifecycle_publisher(
            Detection2DArray, self.get_name() + "/object_boxes", 1
        )
        self._pub_keypoint = self.create_lifecycle_publisher(
            KeyPointArray, self.get_name() + "/object_keypoints", 1
        )
        self._pub_mask = self.create_lifecycle_publisher(
            DetectMaskArray, self.get_name() + "/object_masks", 1
        )
        self._pub_detect_response = self.create_lifecycle_publisher(
            Detection2DArray, self.get_name() + "/detect_response", 10
        )
        self._pub_stats = self.create_lifecycle_publisher(
            String, self.get_name() + "/stats", 10
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._image_subs = self._subscribe_images()
        self._detect_request_sub = self.create_subscription(
            Image, self.get_name() + "/detect_request", self.detect_request_callback, 10
        )
        self.stats.reset()
        self._stats_timer = self.create_timer(1.0, self._publish_stats)
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        self._destroy_subscriptions()
        self.trails.clear()
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._remove_param_cb()
        self._destroy_publishers()
        self._remove_tracker_runtime_config()
        self._release_predictor()
        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._destroy_subscriptions()
        self._remove_param_cb()
        self._destroy_publishers()
        self._remove_tracker_runtime_config()
        self._release_predictor()
        return super().on_shutdown(state)

    def _subscribe_images(self):
        """
        Create the camera subscription(s) for image_reliability.

        'auto' subscribes with both reliable and best-effort QoS: a best-effort reader drops most
        large frames from a reliable camera, and a reliable reader cannot receive from a
        best-effort camera. Frames received by both readers are dropped by header in
        image_callback.
        """
        reliabilities = (
            (QoSReliabilityPolicy.RELIABLE, QoSReliabilityPolicy.BEST_EFFORT)
            if self.image_reliability == "auto"
            else (self._RELIABILITY_MAP[self.image_reliability],)
        )
        quiet = SubscriptionEventCallbacks(incompatible_qos=lambda event: None)
        return [
            self.create_subscription(
                Image,
                self.image_topic_name,
                self.image_callback,
                QoSProfile(
                    reliability=reliability,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=1,
                ),
                event_callbacks=quiet if len(reliabilities) > 1 else None,
            )
            for reliability in reliabilities
        ]

    def _destroy_subscriptions(self) -> None:
        for sub in self._image_subs:
            self.destroy_subscription(sub)
        self._image_subs = []
        if self._stats_timer is not None:
            self.destroy_timer(self._stats_timer)
            self._stats_timer = None
        if self._detect_request_sub is not None:
            self.destroy_subscription(self._detect_request_sub)
            self._detect_request_sub = None

    def _destroy_publishers(self) -> None:
        for attr in (
            "_pub_img",
            "_pub_rect",
            "_pub_keypoint",
            "_pub_mask",
            "_pub_detect_response",
            "_pub_stats",
        ):
            pub = getattr(self, attr, None)
            if pub is not None:
                self.destroy_lifecycle_publisher(pub)
                setattr(self, attr, None)

    def _remove_param_cb(self) -> None:
        if getattr(self, "_param_cb_registered", False):
            try:
                self.remove_on_set_parameters_callback(self._param_cb)
            except Exception:
                pass
            self._param_cb_registered = False

    def _is_active(self) -> bool:
        return self._state_machine.current_state[1] == "active"

    # ------------------------------------------------------------------ validation / logging

    def _validate_configuration(self):
        if not 0.0 < self.conf <= 1.0:
            return f"conf must be in (0.0, 1.0], got {self.conf}"
        if not 0.0 < self.iou <= 1.0:
            return f"iou must be in (0.0, 1.0], got {self.iou}"
        if self.yolo_mode not in self._MODE_CHOICES:
            return f"yolo_mode must be one of {self._MODE_CHOICES}, got '{self.yolo_mode}'"
        tracker_valid, tracker_reason = tracking.validate_tracker_settings(
            self.tracker,
            self.tracker_with_reid,
            self.tracker_reid_model,
            self.tracker_reid_weights_path,
        )
        if not tracker_valid:
            return tracker_reason
        if self.trail_length <= 0:
            return f"trail_length must be > 0, got {self.trail_length}"
        if self.line_width <= 0:
            return f"line_width must be > 0, got {self.line_width}"
        if self.imgsz < 32:
            return f"imgsz must be >= 32, got {self.imgsz}"
        if self.trail_mode not in self._TRAIL_MODE_CHOICES:
            return f"trail_mode must be one of {self._TRAIL_MODE_CHOICES}, got '{self.trail_mode}'"
        if self.image_reliability not in self._RELIABILITY_CHOICES:
            return (
                f"image_reliability must be one of {list(self._RELIABILITY_CHOICES)}, "
                f"got '{self.image_reliability}'"
            )
        return ""

    def _log_configuration(self) -> None:
        logger = self.get_logger()
        logger.info(f"Image topic: {self.image_topic_name}")
        logger.info(f"Weight file: {self.weight_file}")
        logger.info(f"Weights path: {self.weights_path}")
        logger.info(f"Conf: {self.conf}")
        logger.info(f"IoU: {self.iou}")
        logger.info(f"YOLO mode: {self.yolo_mode}")
        logger.info(f"Use tracking: {self.use_tracking}")
        logger.info(f"Tracker: {self.tracker}")
        logger.info(f"Tracker with ReID: {self.tracker_with_reid}")
        self._log_reid_ignored()
        if self._reid_enabled():
            logger.info(f"Tracker ReID model: {self._resolved_reid_model()}")
            logger.info(f"Tracker ReID weights path: {self.tracker_reid_weights_path}")
        logger.info(f"Trail mode: {self.trail_mode}")
        logger.info(f"Trail length: {self.trail_length}")
        logger.info(f"Line width: {self.line_width}")
        logger.info(f"Use detection filter: {self.use_detection_filter}")
        logger.info(f"Use person keypoint filter: {self.use_person_keypoint_filter}")
        logger.info(f"Filter classes: {self.filter_classes}")
        logger.info(f"Keypoint publish names: {self.keypoint_publish_names}")
        logger.info(f"Person keypoint filter names: {self.person_keypoint_filter_names}")
        logger.info(f"Keypoint trail names: {self.keypoint_trail_names}")
        logger.info(f"YOLOE prompts: {self.yoloe_prompts}")
        logger.info(f"Image reliability: {self.image_reliability}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Fuse: {self.fuse}")
        logger.info(f"Image size: {self.imgsz}")
        logger.info(f"Half precision: {self.half}")

    def _get_string_array_parameter(self, name):
        try:
            return list(self.get_parameter(name).get_parameter_value().string_array_value)
        except ParameterUninitializedException:
            return []

    # ------------------------------------------------------------------ class filter

    def _is_yoloe(self) -> bool:
        return self._predictor is not None and hasattr(self._predictor, "set_classes")

    def _model_names(self):
        if self._predictor is None:
            return {}
        names = getattr(self._predictor, "names", {})
        if isinstance(names, dict):
            return names
        return dict(enumerate(names))

    def _active_filter_classes(self):
        if not self.use_detection_filter:
            return []
        return [name for name in self.filter_classes if name]

    def _filter_class_ids(self):
        active = self._active_filter_classes()
        if not active or self._is_yoloe():
            return None

        name_to_id = {name: class_id for class_id, name in self._model_names().items()}
        unknown_names = [name for name in active if name not in name_to_id]
        if unknown_names:
            self.get_logger().warn(
                f"filter_classes contains names not found in this model: {unknown_names}"
            )
        return [int(name_to_id[name]) for name in active if name in name_to_id]

    def _log_filter_class_ids(self) -> None:
        class_ids = self._filter_class_ids()
        if class_ids is not None:
            self.get_logger().info(f"Filter class IDs: {class_ids}")

    def _warn_filter_classes_ignored(self) -> None:
        if not self._is_yoloe() or not self.use_detection_filter:
            return
        if any(self.filter_classes):
            self.get_logger().warn(
                "filter_classes is set but this is a YOLOE model — "
                "use yoloe_prompts to filter classes; filter_classes will be ignored"
            )

    # ------------------------------------------------------------------ keypoints

    def _publish_keypoint_names(self):
        return keypoints.active_names(self.keypoint_publish_names)

    def _person_keypoint_filter_names(self):
        if not self.use_person_keypoint_filter:
            return []
        return keypoints.active_names(self.person_keypoint_filter_names)

    def _draw_bbox_trails_enabled(self):
        return self.trail_mode in ("bbox", "all")

    def _draw_keypoint_trails_enabled(self):
        return self.trail_mode in ("keypoint", "all")

    def _trail_keypoint_names(self):
        if not self._draw_keypoint_trails_enabled():
            return []
        return keypoints.active_names(self.keypoint_trail_names)

    def _validate_keypoint_settings(self):
        return keypoints.validate_keypoint_settings(
            self._predictor,
            self._publish_keypoint_names(),
            self._person_keypoint_filter_names(),
            self._trail_keypoint_names(),
        )

    def _warn_keypoint_names(self) -> None:
        kpt_shape = keypoints.model_keypoint_shape(self._predictor)
        if kpt_shape is None:
            return
        if not self._publish_keypoint_names():
            self.get_logger().warn(
                f"Pose model detected ({kpt_shape[0]} keypoints) "
                "but keypoint_publish_names is empty — no keypoints will be published"
            )
        for warning in keypoints.keypoint_name_warnings(
            self._predictor,
            (
                ("keypoint_publish_names", self.keypoint_publish_names),
                ("person_keypoint_filter_names", self.person_keypoint_filter_names),
                ("keypoint_trail_names", self.keypoint_trail_names),
            ),
        ):
            self.get_logger().warn(warning)

    def _warn_keypoint_settings_ignored(self) -> None:
        if keypoints.model_keypoint_shape(self._predictor) is not None:
            return
        if self._person_keypoint_filter_names():
            self.get_logger().warn(
                "use_person_keypoint_filter is true but this model has no keypoints — "
                "person keypoint filtering will be ignored"
            )
        if self._trail_keypoint_names():
            self.get_logger().warn(
                "trail_mode requests keypoint trails but this model has no keypoints — "
                "keypoint trails will be ignored"
            )

    # ------------------------------------------------------------------ tracker

    def _reid_enabled(self):
        return tracking.reid_enabled(self.tracker, self.tracker_with_reid)

    def _log_reid_ignored(self):
        if self.tracker_with_reid and not self._reid_enabled():
            self.get_logger().info(
                f"{self.tracker} has no ReID; tracker_with_reid is ignored for this tracker"
            )

    def _resolved_reid_model(self):
        return tracking.resolve_reid_model(
            self.tracker_reid_model, self.tracker_reid_weights_path or self.weights_path
        )

    def _remove_tracker_runtime_config(self) -> None:
        tracking.remove_file_quietly(self._tracker_runtime_config_path)
        self._tracker_runtime_config_path = None
        self._tracker_runtime_config = self.tracker

    def _prepare_tracker_runtime_config(self) -> bool:
        self._remove_tracker_runtime_config()
        if not self._reid_enabled():
            return True
        try:
            self._tracker_runtime_config_path = tracking.write_reid_tracker_config(
                self.tracker, self._resolved_reid_model()
            )
            self._tracker_runtime_config = self._tracker_runtime_config_path
            return True
        except Exception as error:
            self.get_logger().error(f"Failed to prepare tracker config: {error}")
            return False

    def _reset_tracking_state(self) -> None:
        self.trails.clear()
        tracking.reset_tracker_state(self._predictor)

    # ------------------------------------------------------------------ model

    def _release_predictor(self) -> None:
        model = self._predictor
        model_device = str(getattr(model, "device", ""))
        self._predictor = None
        if model is not None:
            if getattr(model, "predictor", None) is not None:
                model.predictor = None
            if getattr(model, "trainer", None) is not None:
                model.trainer = None
            del model
        gc.collect()
        if "cuda" in model_device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def load_model(
        self, weight_file=None, weights_path=None, yoloe_prompts=None, device=None, fuse=None
    ):
        model_weight_file = self.weight_file if weight_file is None else weight_file
        model_weights_path = self.weights_path if weights_path is None else weights_path
        model_yoloe_prompts = self.yoloe_prompts if yoloe_prompts is None else yoloe_prompts
        model_device = self.device if device is None else device
        model_fuse = self.fuse if fuse is None else fuse
        model_full_path = os.path.join(model_weights_path, model_weight_file)

        if not os.path.exists(model_full_path):
            self.get_logger().error(
                f"Model file not found: {model_full_path} "
                "(download it into yolo_ros/weights/, see README 'Model Downloads')"
            )
            return False

        try:
            self._release_predictor()
            self.get_logger().info(f"Loading model: {model_full_path} on {model_device}")
            model = YOLO(model_full_path)
            model.to(model_device)
            if hasattr(model, "set_classes"):
                active_prompts = [p for p in model_yoloe_prompts if p]
                if not active_prompts:
                    self.get_logger().error(
                        "YOLOE model requires at least one non-empty prompt in yoloe_prompts"
                    )
                    return False
                model.set_classes(active_prompts)
            if model_fuse:
                if hasattr(model, "fuse"):
                    model.fuse()
                    self.get_logger().info("Model fused (Conv+BN layers merged)")
                else:
                    self.get_logger().warn(
                        "fuse=True but this model type does not support fuse() — skipping"
                    )
            self._predictor = model
            self.get_logger().info(f"Model loaded: {model_full_path} on {model.device}")
            self._publish_model_info(model_weight_file)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False

    def _publish_model_info(self, weight_file):
        """JSON {"weight_file", "classes": [names by class id], "keypoints": [names]}."""
        names = self._model_names()
        info = {
            "weight_file": weight_file,
            "classes": [names[class_id] for class_id in sorted(names)],
            "keypoints": keypoints.model_keypoint_names(self._predictor) or [],
        }
        self._pub_model_info.publish(String(data=json.dumps(info)))

    # ------------------------------------------------------------------ parameters

    def parameters_callback(self, params):
        # Keypoint settings are applied one by one and validated together, so a rejected set
        # (e.g. a filter on a non-pose model) could leave earlier ones in effect: restore them.
        saved = {name: getattr(self, name) for name in self._VALIDATED_TOGETHER}
        result = self._apply_parameters(params)
        if not result.successful:
            for name, value in saved.items():
                setattr(self, name, value)
        return result

    def _apply_parameters(self, params):
        next_model = {
            "weight_file": self.weight_file,
            "weights_path": self.weights_path,
            "yoloe_prompts": self.yoloe_prompts,
            "fuse": self.fuse,
        }
        next_tracker = {
            "tracker": self.tracker,
            "tracker_with_reid": self.tracker_with_reid,
            "tracker_reid_model": self.tracker_reid_model,
            "tracker_reid_weights_path": self.tracker_reid_weights_path,
        }
        should_reload = False
        tracker_runtime_dirty = False

        for param in params:
            name = param.name
            if name in self._INACTIVE_ONLY_PARAMS and self._is_active():
                return SetParametersResult(
                    successful=False,
                    reason=f"{name} cannot be changed while active; deactivate first",
                )

            if name in ("weight_file", "weights_path"):
                next_model[name] = param.value
                should_reload = True
            elif name == "fuse":
                next_model["fuse"] = bool(param.value)
                should_reload = True
            elif name == "yoloe_prompts":
                if self._is_yoloe():
                    active_prompts = [p for p in param.value if p]
                    if not active_prompts:
                        return SetParametersResult(
                            successful=False,
                            reason=(
                                "YOLOE model requires at least one non-empty prompt "
                                "in yoloe_prompts"
                            ),
                        )
                    self._predictor.set_classes(active_prompts)
                    self.yoloe_prompts = param.value
                    self.get_logger().info(f"Updated yoloe_prompts: {active_prompts}")
                    self._warn_filter_classes_ignored()
                else:
                    next_model["yoloe_prompts"] = param.value
                    should_reload = True
            elif name == "tracker":
                next_tracker["tracker"] = str(param.value)
                tracker_runtime_dirty = True
            elif name == "tracker_with_reid":
                next_tracker["tracker_with_reid"] = bool(param.value)
                tracker_runtime_dirty = True
            elif name == "tracker_reid_model":
                next_tracker["tracker_reid_model"] = str(param.value).strip() or "auto"
                tracker_runtime_dirty = True
            elif name == "tracker_reid_weights_path":
                next_tracker["tracker_reid_weights_path"] = (
                    str(param.value).strip() or self.weights_path
                )
                tracker_runtime_dirty = True
            else:
                result = self._apply_runtime_parameter(param)
                if result is not None:
                    return result

        if tracker_runtime_dirty:
            result = self._apply_tracker_parameters(next_tracker)
            if result is not None:
                return result

        if should_reload and self._predictor is not None:
            if not self.load_model(
                weight_file=next_model["weight_file"],
                weights_path=next_model["weights_path"],
                yoloe_prompts=next_model["yoloe_prompts"],
                fuse=next_model["fuse"],
            ):
                return SetParametersResult(successful=False, reason="Model reload failed")
            self.weight_file = next_model["weight_file"]
            self.weights_path = next_model["weights_path"]
            self.yoloe_prompts = next_model["yoloe_prompts"]
            self.fuse = next_model["fuse"]

        return SetParametersResult(successful=True)

    def _apply_runtime_parameter(self, param):
        """Apply a parameter that can change while active. Returns a failure result or None."""
        name = param.name
        if name == "filter_classes":
            self.filter_classes = list(param.value)
            self._warn_filter_classes_ignored()
            self._log_filter_class_ids()
            self._reset_tracking_state()
        elif name == "use_detection_filter":
            self.use_detection_filter = bool(param.value)
            self._warn_filter_classes_ignored()
            self._log_filter_class_ids()
            self._reset_tracking_state()
            self.get_logger().info(f"Updated use_detection_filter: {self.use_detection_filter}")
        elif name in (
            "use_person_keypoint_filter",
            "keypoint_publish_names",
            "person_keypoint_filter_names",
            "keypoint_trail_names",
        ):
            value = (
                bool(param.value) if name == "use_person_keypoint_filter" else list(param.value)
            )
            setattr(self, name, value)
            keypoint_valid, keypoint_reason = self._validate_keypoint_settings()
            if not keypoint_valid:
                return SetParametersResult(successful=False, reason=keypoint_reason)
            if name == "keypoint_publish_names":
                self._warn_keypoint_names()
            if name == "person_keypoint_filter_names":
                self._reset_tracking_state()
            elif name == "keypoint_trail_names":
                self.trails.keypoint_history.clear()
            self.get_logger().info(f"Updated {name}: {value}")
        elif name == "max_rate_hz":
            self.max_rate_hz = max(0.0, float(param.value))
            self.get_logger().info(f"Updated max_rate_hz: {self.max_rate_hz}")
        elif name in ("conf", "iou"):
            value = float(param.value)
            if not 0.0 < value <= 1.0:
                return SetParametersResult(
                    successful=False, reason=f"{name} must be in (0.0, 1.0]"
                )
            setattr(self, name, value)
            self.get_logger().info(f"Updated {name}: {value}")
        elif name in ("yolo_mode", "use_tracking"):
            if name == "yolo_mode":
                value = str(param.value)
                if value not in self._MODE_CHOICES:
                    return SetParametersResult(
                        successful=False, reason=f"yolo_mode must be one of {self._MODE_CHOICES}"
                    )
            else:
                value = "track" if bool(param.value) else "detect"
            # The tracker is kept on purpose: detect mode does not advance it, so returning to
            # track mode continues the same ids for people who are still in view.
            if value != self.yolo_mode:
                self.yolo_mode = value
                self.use_tracking = value == "track"
                self.trails.clear()
                self.get_logger().info(f"Updated yolo_mode: {self.yolo_mode}")
        elif name == "image_topic_name":
            value = str(param.value)
            if value != self.image_topic_name:
                self.image_topic_name = value
                if self._image_subs:
                    for sub in self._image_subs:
                        self.destroy_subscription(sub)
                    self._image_subs = self._subscribe_images()
                # Tracks from the previous camera are meaningless for the new one.
                self._reset_tracking_state()
                self.get_logger().info(f"Updated image_topic_name: {self.image_topic_name}")
        elif name == "trail_mode":
            value = str(param.value)
            if value not in self._TRAIL_MODE_CHOICES:
                return SetParametersResult(
                    successful=False,
                    reason=f"trail_mode must be one of {self._TRAIL_MODE_CHOICES}",
                )
            self.trail_mode = value
            keypoint_valid, keypoint_reason = self._validate_keypoint_settings()
            if not keypoint_valid:
                return SetParametersResult(successful=False, reason=keypoint_reason)
            if not self._draw_bbox_trails_enabled():
                self.trails.track_history.clear()
            if not self._draw_keypoint_trails_enabled():
                self.trails.keypoint_history.clear()
            self.get_logger().info(f"Updated trail_mode: {self.trail_mode}")
        elif name == "imgsz":
            value = int(param.value)
            if value < 32:
                return SetParametersResult(successful=False, reason="imgsz must be >= 32")
            self.imgsz = value
            self.get_logger().info(f"Updated imgsz: {self.imgsz}")
        elif name == "half":
            value = bool(param.value)
            if value != self.half:
                # Ultralytics rebuilds its predictor for another precision, which also drops the
                # tracker: ids start again.
                self.half = value
                self.trails.clear()
                self.get_logger().info(f"Updated half: {self.half} (track ids restart)")
        elif name in ("trail_length", "line_width"):
            value = int(param.value)
            if value <= 0:
                return SetParametersResult(successful=False, reason=f"{name} must be > 0")
            setattr(self, name, value)
            if name == "trail_length":
                self.trails.trim(value)
            self.get_logger().info(f"Updated {name}: {value}")
        elif name == "image_reliability":
            value = str(param.value)
            if value not in self._RELIABILITY_CHOICES:
                return SetParametersResult(
                    successful=False,
                    reason=f"image_reliability must be one of {list(self._RELIABILITY_CHOICES)}",
                )
            self.image_reliability = value
            self.get_logger().info(f"Updated image_reliability: {self.image_reliability}")
        return None

    def _apply_tracker_parameters(self, next_tracker):
        tracker_valid, tracker_reason = tracking.validate_tracker_settings(
            next_tracker["tracker"],
            next_tracker["tracker_with_reid"],
            next_tracker["tracker_reid_model"],
            next_tracker["tracker_reid_weights_path"] or self.weights_path,
        )
        if not tracker_valid:
            return SetParametersResult(successful=False, reason=tracker_reason)

        for name, value in next_tracker.items():
            setattr(self, name, value)
        if not self._prepare_tracker_runtime_config():
            return SetParametersResult(successful=False, reason="Failed to prepare tracker config")
        self._reset_tracking_state()
        self.get_logger().info(f"Updated tracker: {self.tracker}")
        self.get_logger().info(f"Updated tracker_with_reid: {self.tracker_with_reid}")
        self._log_reid_ignored()
        if self._reid_enabled():
            self.get_logger().info(f"Updated tracker_reid_model: {self._resolved_reid_model()}")
            self.get_logger().info(
                f"Updated tracker_reid_weights_path: {self.tracker_reid_weights_path}"
            )
        return None

    # ------------------------------------------------------------------ inference

    def image_callback(self, msg):
        stamp = msg.header.stamp
        if stamp.sec or stamp.nanosec:
            key = (msg.header.frame_id, stamp.sec, stamp.nanosec)
            if key in self._recent_image_keys:
                return
            self._recent_image_keys.append(key)
        self.stats.frame_received()
        if self.max_rate_hz > 0.0:
            now = time.monotonic()
            if now - self._last_processed < 1.0 / self.max_rate_hz:
                return
            self._last_processed = now
        started = time.monotonic()
        cv_img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.predict(cv_img)
        detections = self.convert_to_ros_msg(results, msg.header)
        self.stats.frame_processed(
            inference_ms(results[0]),
            (time.monotonic() - started) * 1000.0,
            self._latency_ms(msg.header),
            detections,
        )

    def _latency_ms(self, header):
        stamp = header.stamp
        if not (stamp.sec or stamp.nanosec):
            return None
        now = self.get_clock().now().nanoseconds
        latency = (now - (stamp.sec * 1_000_000_000 + stamp.nanosec)) / 1e6
        # A negative or huge value means the camera uses another clock (e.g. sim time).
        return latency if 0.0 <= latency < 60_000.0 else None

    def _publish_stats(self):
        if self._pub_stats is not None:
            self._pub_stats.publish(
                String(
                    data=self.stats.to_json(
                        imgsz=self.imgsz, half=self.half, mode=self.yolo_mode
                    )
                )
            )

    def _inference_kwargs(self):
        """
        Arguments shared by every predict()/track() call.

        They must be identical for all calls (also detect_request): Ultralytics rebuilds its
        predictor, and with it the tracker, when the device or precision differs from last time.
        """
        return {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            **_precision_argument(self.half),
            "verbose": False,
        }

    def detect_request_callback(self, msg):
        """
        Detect objects in a one-off image (e.g. a registration photo), reply on detect_response.

        Runs without the tracker so it never disturbs live track ids, and skips the person keypoint
        filter so that people photographed from behind can still be registered.
        """
        cv_img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with tracking.tracker_disabled(self._predictor):
            results = self._predictor.predict(
                source=cv_img, classes=self._filter_class_ids(), **self._inference_kwargs()
            )
        det_array, _ = build_detections(results[0], msg.header, self._active_filter_classes())
        self._pub_detect_response.publish(det_array)

    def predict(self, cv_image):
        classes = self._filter_class_ids()
        if self.yolo_mode == "track":
            return self._predictor.track(
                source=cv_image,
                classes=classes,
                tracker=self._tracker_runtime_config,
                persist=True,
                **self._inference_kwargs(),
            )

        with tracking.tracker_disabled(self._predictor):
            return self._predictor.predict(
                source=cv_image, classes=classes, **self._inference_kwargs()
            )

    def _publish_annotated_image(self, result, header, keypoint_names):
        # Drawing costs several ms per frame, so it is skipped while nobody watches; the trails
        # are dropped then too, otherwise they would jump from where they were last drawn.
        if self._pub_img.get_subscription_count() == 0:
            self.trails.clear()
            return
        annotated_frame = result.plot(line_width=self.line_width)
        if self.use_tracking and self._draw_bbox_trails_enabled():
            self.trails.draw_bbox_trails(
                annotated_frame, result, self.trail_length, self.line_width
            )
        if self.use_tracking and self._draw_keypoint_trails_enabled():
            self.trails.draw_keypoint_trails(
                annotated_frame,
                result,
                self._trail_keypoint_names(),
                keypoint_names,
                self.conf,
                self.trail_length,
                self.line_width,
            )
        det_img = self._cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        det_img.header = header
        self._pub_img.publish(det_img)

    def convert_to_ros_msg(self, results, header):
        """Publish image, keypoints, boxes and masks of one frame; return the number of boxes."""
        keypoint_names = keypoints.model_keypoint_names(self._predictor)
        result = keypoints.apply_keypoint_filter(
            results[0], self._person_keypoint_filter_names(), keypoint_names, self.conf
        )
        self._publish_annotated_image(result, header, keypoint_names)

        det_array, indices = build_detections(result, header, self._active_filter_classes())
        kp_array = build_keypoints(
            result, header, det_array, indices, self._publish_keypoint_names(), keypoint_names
        )
        mask_array = build_masks(result, header, det_array, indices)

        # Keypoints go out before the boxes of the same frame (same stamp, same order as the
        # boxes), so a consumer handling the boxes usually already has them. Pose models publish
        # them even when empty, so consumers can tell "no keypoints" from "not arrived yet".
        if keypoint_names and self._publish_keypoint_names():
            self._pub_keypoint.publish(kp_array)
        # Published even when empty so consumers can tell "nothing detected" from "no new frame".
        self._pub_rect.publish(det_array)
        # Segmentation models publish masks even when empty, like the boxes.
        if getattr(self._predictor, "task", None) == "segment":
            self._pub_mask.publish(mask_array)
        return len(det_array.detections)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    auto_configure = node.get_parameter("auto_configure").get_parameter_value().bool_value
    auto_activate = node.get_parameter("auto_activate").get_parameter_value().bool_value

    configure_succeeded = True
    if auto_configure or auto_activate:
        configure_result = node.trigger_configure()
        configure_succeeded = configure_result == TransitionCallbackReturn.SUCCESS
    if auto_activate:
        if configure_succeeded:
            node.trigger_activate()
        else:
            node.get_logger().error(
                "Auto-activation requested, but node configuration failed; skipping activation."
            )
    node.get_logger().info("YOLO Node started. Spinning...")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
