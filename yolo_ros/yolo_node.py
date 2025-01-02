# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO, NAS, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from std_srvs.srv import SetBool

from sobits_interfaces.msg import KeyPoint
from sobits_interfaces.msg import KeyPointArray

from sensor_msgs.msg import Image
# from yolo_msgs.msg import Point2D
# from yolo_msgs.msg import BoundingBox2D
# from yolo_msgs.msg import Mask
# from yolo_msgs.msg import KeyPoint2D
# from yolo_msgs.msg import KeyPoint2DArray
# from yolo_msgs.msg import Detection
# from yolo_msgs.msg import DetectionArray
# from yolo_msgs.srv import SetClasses
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose


class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_ros")

        # params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("weight_file", "yolov8m.pt")

        self.declare_parameter("image_topic_name", "image_raw")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        self.declare_parameter("image_show", False)

        self.declare_parameter("init_prediction", True)
        self.declare_parameter("classes", [""])
        self.type_to_model = {"YOLO": YOLO, "NAS": NAS, "World": YOLOWorld}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("weight_file").get_parameter_value().string_value

        # inference params
        self.image_topic_name = (
            self.get_parameter("image_topic_name").get_parameter_value().string_value
        )
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )
        self.image_show = (
            self.get_parameter("image_show").get_parameter_value().bool_value
        )

        # ros params
        self.enable = self.get_parameter("init_prediction").get_parameter_value().bool_value
        self.classes = self.get_parameter("classes").get_parameter_value().string_array_value
        # detection pub
        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub_rect     = self.create_lifecycle_publisher(Detection2DArray, "object_boxes"    , 1)
        self._pub_keypoint = self.create_lifecycle_publisher(KeyPointArray   , "object_keypoints", 1)
        # self._pub_mask     = self.create_lifecycle_publisher(MaskArray       , "object_masks"    , 1)
        self._pub_img      = self.create_lifecycle_publisher(Image           , "detect_image"    , 1)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR

        try:
            self.get_logger().info("Trying to fuse model...")
            self.yolo.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fuse: {e}")

        self._enable_srv = self.create_service(SetBool, "run_ctrl", self.enable_cb)

        if isinstance(self.yolo, YOLOWorld):
            self.yolo.set_classes(self.classes)

        self._sub = self.create_subscription(
            Image, self.image_topic_name, self.image_cb, self.image_qos_profile
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub_rect)
        self.destroy_publisher(self._pub_keypoint)
        # self.destroy_publisher(self._pub_mask)
        self.destroy_publisher(self._pub_img)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response

    def image_cb(self, msg: Image) -> None:
        if self.enable:
            header = msg.header

            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.yolo.predict(
                source=cv_image,
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

            # create detection msgs
            detections_bboxes_msg = Detection2DArray()
            detections_keypoints_msg = KeyPointArray()
            # detections_masks_msg = MaskArray()

            detections_bboxes_msg.header = header
            detections_keypoints_msg.header = header
            # detections_masks_msg.header = header

            for i in range(len(results)):

                bbox = Detection2D()
                kp = KeyPoint()
                # ms = Mask()

                bbox.header = header
                bbox.results = []
                kp.key_names = []
                kp.key_points = []
                ohwp = ObjectHypothesisWithPose()
                if results.boxes or results.obb:
                    if results.boxes:
                        box = results.boxes[i].xywh[0]

                        ohwp.hypothesis.class_id = str(self.yolo.names[int(results.boxes[i].cls)])
                        ohwp.hypothesis.score = float(results.boxes[i].conf)
                        bbox.bbox.center.theta = 0.0
                        bbox.id = str(self.yolo.names[int(results.boxes[i].cls)])
                    
                    else:
                        box = results.obb.xywhr[i]

                        ohwp.hypothesis.class_id = str(self.yolo.names[int(results.obb.cls[i])])
                        ohwp.hypothesis.score = float(results.obb.conf[i])
                        bbox.bbox.center.theta = float(box[4])
                        bbox.id = str(self.yolo.names[int(results.obb.cls[i])])


                    ohwp.pose.pose.position = Point(x=float(box[0]), y=float(box[1]), z=float(-1))
                    ohwp.pose.pose.orientation = Quaternion(x=float(0), y=float(0), z=float(0), w=float(1))
                    ohwp.pose.covariance = [float(0)]*36
                    bbox.results += [ohwp]
                    bbox.bbox.center.position.x = float(box[0])
                    bbox.bbox.center.position.y = float(box[1])
                    bbox.bbox.size_x = float(box[2])
                    bbox.bbox.size_y = float(box[3])
                    detections_bboxes_msg.detections += [bbox]

                    # generate result image
                    label = f"{ohwp.hypothesis.class_id} {ohwp.hypothesis.score:.2f}"
                    x_min = bbox.bbox.center.position.x - bbox.bbox.size_x/2
                    y_min = bbox.bbox.center.position.y - bbox.bbox.size_y/2
                    x_max = bbox.bbox.center.position.x + bbox.bbox.size_x/2
                    y_max = bbox.bbox.center.position.y + bbox.bbox.size_y/2
                    theta = bbox.bbox.center.theta
                    (w, h), baseline = cv2.getTextSize(label,
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.5,
                                            thickness=1)
                    # 回転矩形の頂点を計算
                    rect = ((bbox.bbox.center.position.x, bbox.bbox.center.position.y), (bbox.bbox.size_x, bbox.bbox.size_y), bbox.bbox.center.theta)
                    rec_box = cv2.boxPoints(rect)  # 頂点座標を取得
                    rec_box = np.array(rec_box, dtype=np.int32) # 整数に変換
                    # 回転矩形を描画（輪郭）
                    cv2.polylines(cv_image,
                                [rec_box],
                                isClosed=True,
                                color=colors(c, True),
                                thickness=2,
                                lineType=cv2.LINE_4)
                    # cv2.rectangle(cv_image,
                    #             pt1=(int(x_min), int(y_min)),
                    #             pt2=(int(x_max), int(y_max)),
                    #             color=colors(c, True),
                    #             thickness=2,
                    #             lineType=cv2.LINE_4)
                    cv2.rectangle(cv_image,
                                pt1=(int(x_min), int(y_min) - h),
                                pt2=(int(x_max) + w, int(y_max)),
                                color=colors(c, True),
                                thickness=-1,
                                lineType=cv2.LINE_4)
                    cv2.putText(cv_image,
                                text=label,
                                org=(int(x_min), int(y_min)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(255,255,255),
                                thickness=1,
                                lineType=cv2.LINE_AA)

                if results.keypoints:
                    if results.keypoints[i].conf is None:
                        continue
                    for kp_id, (p, conf) in enumerate(zip(results.keypoints[i].xy[0], results.keypoints[i].conf[0])):
                        if conf >= self.threshold:
                            kp.key_names += [str(kp_id + 1)]
                            kp.key_points += [Point(x=float(p[0]), y=float(p[1]), z=float(-1))]
                            # kp.key_conf += [float(conf)]
                    detections_keypoints_msg.key_points_array += [kp]


                # if results.masks:
                #     def create_point2d(x: float, y: float) -> Point2D:
                #         p = Point2D()
                #         p.x = x
                #         p.y = y
                #         return p

                #     ms.data = [
                #         create_point2d(float(ele[0]), float(ele[1]))
                #         for ele in results.masks[i].xy[0].tolist()
                #     ]
                #     ms.height = results.orig_img.shape[0]
                #     ms.width = results.orig_img.shape[1]
                #     detections_masks_msg.data = [ms]


            detections_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

            # publish detections
            self._pub_rect.publish(detections_bboxes_msg)
            self._pub_keypoint.publish(detections_keypoints_msg)
            # self._pub_mask.publish(detections_masks_msg)
            self._pub_img.publish(detections_image_msg)

            del results
            del cv_image


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()  # ノードを構造する
    node.trigger_activate()   # ノードをアクティブにする
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
