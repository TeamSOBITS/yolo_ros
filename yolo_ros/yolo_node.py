import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState

from ultralytics import YOLO, NAS, YOLOWorld
from ultralytics.engine.results import Results

from geometry_msgs.msg import Point, Quaternion
from std_srvs.srv import SetBool
from sobits_interfaces.msg import KeyPoint, KeyPointArray
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class YoloNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("yolo_ros")

        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("weight_file", "yolov8m.pt")
        self.declare_parameter("image_topic_name", "image_raw")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 480)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        self.declare_parameter("image_show", False)
        self.declare_parameter("execute_default", True)
        self.declare_parameter("classes", [""])
        self.declare_parameter("keypoint_name_list", [""])

        self.type_to_model = {"YOLO": YOLO, "NAS": NAS, "World": YOLOWorld}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.model_type = self.get_parameter("model_type").get_parameter_value().string_value
        self.model = self.get_parameter("weight_file").get_parameter_value().string_value
        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
        self.image_show = self.get_parameter("image_show").get_parameter_value().bool_value
        self.enable = self.get_parameter("execute_default").get_parameter_value().bool_value
        self.classes = self.get_parameter("classes").get_parameter_value().string_array_value
        self.keypoint_name_list = self.get_parameter("keypoint_name_list").get_parameter_value().string_array_value

        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub_rect = self.create_lifecycle_publisher(Detection2DArray, "object_boxes", 1)
        self._pub_keypoint = self.create_lifecycle_publisher(KeyPointArray, "object_keypoints", 1)
        self._pub_img = self.create_lifecycle_publisher(Image, "detect_image", 1)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            return TransitionCallbackReturn.ERROR

        try:
            self.yolo.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fuse: {e}")

        self._enable_srv = self.create_service(SetBool, "run_ctrl", self.enable_cb)

        if isinstance(self.yolo, YOLOWorld):
            self.yolo.set_classes(self.classes)

        self._sub = self.create_subscription(Image, self.image_topic_name, self.image_cb, self.image_qos_profile)

        super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        del self.yolo
        self.destroy_service(self._enable_srv)
        self._enable_srv = None
        self.destroy_subscription(self._sub)
        self._sub = None
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.destroy_publisher(self._pub_rect)
        self.destroy_publisher(self._pub_keypoint)
        self.destroy_publisher(self._pub_img)
        del self.image_qos_profile
        super().on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        super().on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response

    def image_cb(self, msg: Image) -> None:
        if not self.enable:
            return

        encoding = msg.encoding
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        if encoding == 'bgr8':
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif encoding == 'bgra8':
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        elif encoding == 'rgba8':
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
        elif encoding == 'rgb8':
            cv_image_rgb = cv_image
        else:
            self.get_logger().error(f"Unsupported encoding: {encoding}")
            return

        results = self.yolo.predict(
            source=cv_image_rgb,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            retina_masks=self.retina_masks,
            show=self.image_show,
        )
        
        results: Results = results[0].cpu()

        detections_bboxes_msg = Detection2DArray()
        detections_keypoints_msg = KeyPointArray()
        header = msg.header
        detections_bboxes_msg.header = header
        detections_keypoints_msg.header = header

        for i in range(len(results)):
            bbox = Detection2D()
            kp = KeyPoint()
            bbox.header = header
            bbox.results = []
            kp.key_names = []
            kp.key_points = []

            if results.boxes:
                box = results.boxes[i].xywh[0]
                cls_idx = int(results.boxes[i].cls[0])
                conf = float(results.boxes[i].conf[0])
                class_name = str(self.yolo.names[cls_idx])

                ohwp = ObjectHypothesisWithPose()
                ohwp.hypothesis.class_id = class_name
                ohwp.hypothesis.score = conf

                bbox.id = class_name
                bbox.bbox.center.position.x = float(box[0])
                bbox.bbox.center.position.y = float(box[1])
                bbox.bbox.size_x = float(box[2])
                bbox.bbox.size_y = float(box[3])
                ohwp.pose.pose.position = Point(x=float(box[0]), y=float(box[1]), z=-1.0)
                ohwp.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                ohwp.pose.covariance = [0.0] * 36

                bbox.results.append(ohwp)
                detections_bboxes_msg.detections.append(bbox)

            if results.keypoints:
                if results.keypoints[i].conf is None:
                    continue
                for kp_id, (p, conf) in enumerate(zip(results.keypoints[i].xy[0], results.keypoints[i].conf[0])):
                    if conf >= self.threshold:
                        kp.key_names.append(str(self.keypoint_name_list[kp_id]))
                        kp.key_points.append(Point(x=float(p[0]), y=float(p[1]), z=-1.0))
                detections_keypoints_msg.key_points_array.append(kp)

        if hasattr(results, 'plot'):
            annotated_image = results.plot()
            ros_image = self.cv_bridge.cv2_to_imgmsg(annotated_image, "rgb8")
            ros_image.header = header
        else:
            ros_image = self.cv_bridge.cv2_to_imgmsg(cv_image_rgb, "rgb8")
            ros_image.header = header

        self._pub_rect.publish(detections_bboxes_msg)
        self._pub_keypoint.publish(detections_keypoints_msg)
        self._pub_img.publish(ros_image)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown
        
        
if __name__ == '__main__':
    main()