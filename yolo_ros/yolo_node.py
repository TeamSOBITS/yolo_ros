import os
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
        self._sub = None

        self._pub_img = None
        self._pub_rect = None
        self._pub_keypoint = None
        self._pub_mask = None

        self.add_on_set_parameters_callback(self.parameters_callback)

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

        if not self.load_model():
            return TransitionCallbackReturn.FAILURE

        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub_img = self.create_lifecycle_publisher(Image, self.get_name() + "/detected_image", 1)
        self._pub_rect = self.create_lifecycle_publisher(Detection2DArray, self.get_name() + "/object_boxes", 1)
        self._pub_keypoint = self.create_lifecycle_publisher(KeyPointArray, self.get_name() + "/object_keypoints", 1)
        self._pub_mask = self.create_lifecycle_publisher(DetectMaskArray, self.get_name() + "/object_masks", 1)

        return TransitionCallbackReturn.SUCCESS

    def load_model(self, weight_file=None, weights_path=None, yoloe_prompts=None):
        try:
            model_weight_file = self.weight_file if weight_file is None else weight_file
            model_weights_path = self.weights_path if weights_path is None else weights_path
            model_yoloe_prompts = self.yoloe_prompts if yoloe_prompts is None else yoloe_prompts
            model_full_path = os.path.join(model_weights_path, model_weight_file)
            self.get_logger().info(f"Loading model: {model_full_path}")
            model = YOLO(model_full_path)
            if hasattr(model, "set_classes"):
                model.set_classes(model_yoloe_prompts)
            self.model = model
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False

    def parameters_callback(self, params):
        next_weight_file = self.weight_file
        next_weights_path = self.weights_path
        next_yoloe_prompts = self.yoloe_prompts
        next_filter_classes = self.filter_classes
        next_conf = self.conf
        next_iou = self.iou
        should_reload_model = False

        for param in params:
            if param.name == "weight_file":
                next_weight_file = param.value
                should_reload_model = True
            elif param.name == "weights_path":
                next_weights_path = param.value
                should_reload_model = True
            elif param.name == "yoloe_prompts":
                next_yoloe_prompts = param.value
                should_reload_model = True
            elif param.name == "filter_classes":
                next_filter_classes = param.value
            elif param.name == "conf":
                next_conf = param.value
            elif param.name == "iou":
                next_iou = param.value

        if should_reload_model and not self.load_model(
            weight_file=next_weight_file,
            weights_path=next_weights_path,
            yoloe_prompts=next_yoloe_prompts,
        ):
            return SetParametersResult(successful=False)

        self.weight_file = next_weight_file
        self.weights_path = next_weights_path
        self.yoloe_prompts = next_yoloe_prompts
        self.filter_classes = next_filter_classes
        self.conf = next_conf
        self.iou = next_iou

        return SetParametersResult(successful=True)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating...")
        self._sub = self.create_subscription(Image, self.image_topic_name, self.image_callback, self.image_qos_profile)
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        self.destroy_subscription(self._sub)
        self._sub = None
        return super().on_deactivate(state)

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

    def image_callback(self, msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.predict(cv_img)
        self.convert_to_ros_msg(results, msg.header)

    def predict(self, cv_image):
        results = self.model.predict(
            source=cv_image,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        return results

    def convert_to_ros_msg(self, results, header):
        result = results[0]

        annotated_frame = result.plot()
        det_img = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        det_img.header = header

        det_array = Detection2DArray(header=header)
        kp_array = KeyPointArray(header=header)
        mask_array = DetectMaskArray(header=header)

        for i, box in enumerate(result.boxes):
            cls_value = self._extract_scalar_value(box.cls)
            cls_idx = int(cls_value)
            label = result.names[cls_idx]
            score_value = self._extract_scalar_value(box.conf)
            score = float(score_value)

            active_filter_classes = [name for name in self.filter_classes if name]
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

    execute_default = node.get_parameter("execute_default").get_parameter_value().bool_value
    if execute_default:
        node.trigger_configure()
        node.trigger_activate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
