import gc
import os
import torch
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from sobits_interfaces.msg import KeyPointArray, KeyPoint, DetectMaskArray, DetectMask

from ultralytics import YOLO

class YoloNode(LifecycleNode):

    _RELIABILITY_MAP = {
        "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
        "reliable": QoSReliabilityPolicy.RELIABLE,
        "system_default": QoSReliabilityPolicy.SYSTEM_DEFAULT,
        "best_available": QoSReliabilityPolicy.BEST_AVAILABLE,
        "unknown": QoSReliabilityPolicy.UNKNOWN,
    }

    def __init__(self) -> None:
        super().__init__("yolo_ros")

        self.declare_parameter("image_topic_name", "camera/color/image_raw")
        self.declare_parameter("weight_file", "yolo26n.pt")
        self.declare_parameter("weights_path", "")
        self.declare_parameter("auto_configure", True)
        self.declare_parameter("auto_activate", True)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.7)

        self.declare_parameter("filter_classes", [""])
        self.declare_parameter("keypoint_name_list", [""])
        self.declare_parameter("yoloe_prompts", [""])
        self.declare_parameter("image_reliability", "best_effort")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("fuse", True)

        self._cv_bridge = CvBridge()
        self._predictor = None
        self._sub = None

        self._pub_img = None
        self._pub_rect = None
        self._pub_keypoint = None
        self._pub_mask = None

        self._param_cb = self.add_on_set_parameters_callback(self.parameters_callback)
        self._param_cb_registered = True

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configure...")

        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        self.weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
        self.weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value

        self.filter_classes = self.get_parameter("filter_classes").get_parameter_value().string_array_value
        self.keypoint_name_list = self.get_parameter("keypoint_name_list").get_parameter_value().string_array_value
        self.yoloe_prompts = self.get_parameter("yoloe_prompts").get_parameter_value().string_array_value
        self.image_reliability = self.get_parameter("image_reliability").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse = self.get_parameter("fuse").get_parameter_value().bool_value

        if not 0.0 < self.conf <= 1.0:
            self.get_logger().error(f"conf must be in (0.0, 1.0], got {self.conf}")
            return TransitionCallbackReturn.FAILURE
        if not 0.0 < self.iou <= 1.0:
            self.get_logger().error(f"iou must be in (0.0, 1.0], got {self.iou}")
            return TransitionCallbackReturn.FAILURE
        if self.image_reliability not in self._RELIABILITY_MAP:
            self.get_logger().error(
                f"image_reliability must be one of {list(self._RELIABILITY_MAP)}, got '{self.image_reliability}'"
            )
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info(f"Image topic: {self.image_topic_name}")
        self.get_logger().info(f"Weight file: {self.weight_file}")
        self.get_logger().info(f"Weights path: {self.weights_path}")
        self.get_logger().info(f"Conf: {self.conf}")
        self.get_logger().info(f"IoU: {self.iou}")
        self.get_logger().info(f"Filter classes: {self.filter_classes}")
        self.get_logger().info(f"Keypoint name list: {self.keypoint_name_list}")
        self.get_logger().info(f"YOLOE prompts: {self.yoloe_prompts}")
        self.get_logger().info(f"Image reliability: {self.image_reliability}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Fuse: {self.fuse}")

        if not self.load_model():
            return TransitionCallbackReturn.FAILURE

        self._validate_keypoint_name_list()
        self._warn_filter_classes_ignored()

        self.image_qos_profile = QoSProfile(
            reliability=self._RELIABILITY_MAP[self.image_reliability],
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub_img = self.create_lifecycle_publisher(Image, self.get_name() + "/detected_image", 1)
        self._pub_rect = self.create_lifecycle_publisher(Detection2DArray, self.get_name() + "/object_boxes", 1)
        self._pub_keypoint = self.create_lifecycle_publisher(KeyPointArray, self.get_name() + "/object_keypoints", 1)
        self._pub_mask = self.create_lifecycle_publisher(DetectMaskArray, self.get_name() + "/object_masks", 1)

        return TransitionCallbackReturn.SUCCESS

    def _is_yoloe(self) -> bool:
        return self._predictor is not None and hasattr(self._predictor, "set_classes")

    def _warn_filter_classes_ignored(self) -> None:
        if not self._is_yoloe():
            return
        active = [n for n in self.filter_classes if n]
        if active:
            self.get_logger().warn(
                "filter_classes is set but this is a YOLOE model — "
                "use yoloe_prompts to filter classes; filter_classes will be ignored"
            )

    def _validate_keypoint_name_list(self) -> None:
        kpt_shape = getattr(self._predictor, "kpt_shape", None)
        if kpt_shape is not None:
            self.keypoint_name_list = [n for n in self.keypoint_name_list if n]
            if not self.keypoint_name_list:
                self.get_logger().warn(
                    f"Pose model detected ({kpt_shape[0]} keypoints) but keypoint_name_list is empty — "
                    "no keypoints will be published"
                )

    def _release_predictor(self) -> None:
        model = getattr(self, "_predictor", None)
        model_device = str(getattr(model, "device", ""))
        self._predictor = None
        if model is not None:
            if hasattr(model, "predictor") and model.predictor is not None:
                model.predictor = None
            if hasattr(model, "trainer") and model.trainer is not None:
                model.trainer = None
            del model
        gc.collect()
        if "cuda" in model_device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def load_model(self, weight_file=None, weights_path=None, yoloe_prompts=None, device=None, fuse=None):
        model_weight_file = self.weight_file if weight_file is None else weight_file
        model_weights_path = self.weights_path if weights_path is None else weights_path
        model_yoloe_prompts = self.yoloe_prompts if yoloe_prompts is None else yoloe_prompts
        model_device = self.device if device is None else device
        model_fuse = self.fuse if fuse is None else fuse
        model_full_path = os.path.join(model_weights_path, model_weight_file)

        if not os.path.exists(model_full_path):
            self.get_logger().error(f"Model file not found: {model_full_path}")
            return False

        try:
            self._release_predictor()
            self.get_logger().info(f"Loading model: {model_full_path} on {model_device}")
            model = YOLO(model_full_path)
            model.to(model_device)
            if hasattr(model, "set_classes"):
                active_prompts = [p for p in model_yoloe_prompts if p]
                if not active_prompts:
                    self.get_logger().error("YOLOE model requires at least one non-empty prompt in yoloe_prompts")
                    return False
                model.set_classes(active_prompts)
            if model_fuse:
                if hasattr(model, "fuse"):
                    model.fuse()
                    self.get_logger().info("Model fused (Conv+BN layers merged)")
                else:
                    self.get_logger().warn("fuse=True but model does not support fuse() — skipping")
            self._predictor = model
            self.get_logger().info(f"Model loaded: {model_full_path} on {model.device}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False

    def parameters_callback(self, params):
        next_weight_file = self.weight_file
        next_weights_path = self.weights_path
        next_yoloe_prompts = self.yoloe_prompts
        next_fuse = self.fuse
        should_reload = False

        for param in params:
            if param.name in ("weight_file", "weights_path"):
                if self._state_machine.current_state[1] == "active":
                    return SetParametersResult(
                        successful=False,
                        reason=f"{param.name} cannot be changed while active; deactivate first",
                    )
                if param.name == "weight_file":
                    next_weight_file = param.value
                elif param.name == "weights_path":
                    next_weights_path = param.value
                should_reload = True
            elif param.name == "yoloe_prompts":
                active_prompts = [p for p in param.value if p]
                if self._predictor is not None and hasattr(self._predictor, "set_classes"):
                    if not active_prompts:
                        return SetParametersResult(
                            successful=False,
                            reason="YOLOE model requires at least one non-empty prompt in yoloe_prompts",
                        )
                    self._predictor.set_classes(active_prompts)
                    self.yoloe_prompts = param.value
                    self.get_logger().info(f"Updated yoloe_prompts: {active_prompts}")
                    self._warn_filter_classes_ignored()
                else:
                    next_yoloe_prompts = param.value
                    should_reload = True
            elif param.name == "fuse":
                if self._state_machine.current_state[1] == "active":
                    return SetParametersResult(
                        successful=False,
                        reason="fuse cannot be changed while active; deactivate first",
                    )
                next_fuse = bool(param.value)
                should_reload = True
            elif param.name == "filter_classes":
                self.filter_classes = param.value
                self._warn_filter_classes_ignored()
            elif param.name == "keypoint_name_list":
                self.keypoint_name_list = list(param.value)
                self._validate_keypoint_name_list()
                self.get_logger().info(f"Updated keypoint_name_list: {self.keypoint_name_list}")
            elif param.name == "conf":
                value = float(param.value)
                if not 0.0 < value <= 1.0:
                    return SetParametersResult(successful=False, reason="conf must be in (0.0, 1.0]")
                self.conf = value
                self.get_logger().info(f"Updated conf: {self.conf}")
            elif param.name == "iou":
                value = float(param.value)
                if not 0.0 < value <= 1.0:
                    return SetParametersResult(successful=False, reason="iou must be in (0.0, 1.0]")
                self.iou = value
                self.get_logger().info(f"Updated iou: {self.iou}")
            elif param.name == "image_reliability":
                if self._state_machine.current_state[1] == "active":
                    return SetParametersResult(
                        successful=False,
                        reason="image_reliability cannot be changed while active; deactivate first",
                    )
                value = str(param.value)
                if value not in self._RELIABILITY_MAP:
                    return SetParametersResult(
                        successful=False,
                        reason=f"image_reliability must be one of {list(self._RELIABILITY_MAP)}",
                    )
                self.image_reliability = value
                self.image_qos_profile.reliability = self._RELIABILITY_MAP[value]
                self.get_logger().info(f"Updated image_reliability: {self.image_reliability}")

        if should_reload:
            if not self.load_model(
                weight_file=next_weight_file,
                weights_path=next_weights_path,
                yoloe_prompts=next_yoloe_prompts,
                fuse=next_fuse,
            ):
                return SetParametersResult(successful=False, reason="Model reload failed")
            self.weight_file = next_weight_file
            self.weights_path = next_weights_path
            self.yoloe_prompts = next_yoloe_prompts
            self.fuse = next_fuse

        return SetParametersResult(successful=True)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._sub = self.create_subscription(Image, self.image_topic_name, self.image_callback, self.image_qos_profile)
        return super().on_activate(state)

    def _destroy_subscription(self) -> None:
        sub = getattr(self, "_sub", None)
        if sub is not None:
            self.destroy_subscription(sub)
            self._sub = None

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        self._destroy_subscription()
        return super().on_deactivate(state)

    def _remove_param_cb(self) -> None:
        if getattr(self, "_param_cb_registered", False):
            try:
                self.remove_on_set_parameters_callback(self._param_cb)
            except Exception:
                pass
            self._param_cb_registered = False

    def _destroy_publishers(self) -> None:
        for attr in ("_pub_img", "_pub_rect", "_pub_keypoint", "_pub_mask"):
            pub = getattr(self, attr, None)
            if pub is not None:
                self.destroy_lifecycle_publisher(pub)
                setattr(self, attr, None)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._remove_param_cb()
        self._destroy_publishers()
        self._release_predictor()
        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._destroy_subscription()
        self._remove_param_cb()
        self._destroy_publishers()
        self._release_predictor()
        return super().on_shutdown(state)

    def image_callback(self, msg):
        cv_img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.predict(cv_img)
        self.convert_to_ros_msg(results, msg.header)

    def predict(self, cv_image):
        results = self._predictor.predict(
            source=cv_image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        return results

    def convert_to_ros_msg(self, results, header):
        result = results[0]

        annotated_frame = result.plot()
        det_img = self._cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        det_img.header = header

        det_array = Detection2DArray(header=header)
        kp_array = KeyPointArray(header=header)
        mask_array = DetectMaskArray(header=header)

        active_filter_classes = [name for name in self.filter_classes if name]
        for i, box in enumerate(result.boxes):
            cls_value = self._extract_scalar_value(box.cls)
            cls_idx = int(cls_value)
            label = result.names[cls_idx]
            score_value = self._extract_scalar_value(box.conf)
            score = float(score_value)

            if active_filter_classes and label not in active_filter_classes:
                continue

            det = Detection2D(header=header)
            det.id = label
            det.bbox.center.position.x = float(box.xywh[0][0])
            det.bbox.center.position.y = float(box.xywh[0][1])
            det.bbox.size_x = float(box.xywh[0][2])
            det.bbox.size_y = float(box.xywh[0][3])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score

            det.results.append(hyp)
            det_array.detections.append(det)

            if result.keypoints is not None:
                kp = KeyPoint(key_names=self.keypoint_name_list, score=score)
                for p in result.keypoints[i].xy[0]:
                    pt = Point(x=float(p[0]), y=float(p[1]), z=0.0)
                    kp.key_points.append(pt)
                kp_array.key_points_array.append(kp)

            if result.masks is not None:
                mask = DetectMask(instance_id=label)
                mask.results.append(hyp)
                mask.pixel_x = [int(x) for x in result.masks[i].xy[0][:, 0]]
                mask.pixel_y = [int(y) for y in result.masks[i].xy[0][:, 1]]
                mask_array.masks.append(mask)

        self._pub_img.publish(det_img)

        if len(det_array.detections) > 0:
            self._pub_rect.publish(det_array)
        if len(kp_array.key_points_array) > 0:
            self._pub_keypoint.publish(kp_array)
        if len(mask_array.masks) > 0:
            self._pub_mask.publish(mask_array)

    @staticmethod
    def _extract_scalar_value(value):
        """Return a scalar from tensor/array/scalar YOLO outputs."""
        if hasattr(value, "item"):
            return value.item()
        try:
            return value[0]
        except (TypeError, IndexError, KeyError):
            return value


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
                "Auto-activation requested, but node configuration failed; "
                "skipping activation."
            )
    node.get_logger().info("YOLO Node started. Spinning...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
