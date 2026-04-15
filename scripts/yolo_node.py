#!/usr/bin/env python3
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics import YOLO
from ultralytics.engine.results import Results

from std_srvs.srv import SetBool, SetBoolResponse


class YoloNode:
    def __init__(self) -> None:
        rospy.init_node("yolo_node", anonymous=True)

        self.is_active = True
        self.srv = rospy.Service('yolo_trigger', SetBool, self.trigger)

        # Load params
        self.image_topic_name = rospy.get_param("~image_topic_name", "/usb_cam/image_raw")
        self.conf_th = rospy.get_param("~conf_th", 0.25)
        self.model_path = rospy.get_param("~model_path")
        self.half_bool = rospy.get_param("~half_bool", False)

        # YOLO Model
        # self.model = YOLO(str(self.model_path))
        self.model = YOLO(str(self.model_path)).to("cuda")

        # Define publishers
        self.result_image_pub = rospy.Publisher("/yolo/result_image", Image)
        self.result_bboxes_pub = rospy.Publisher("/yolo/result_bboxes", Detection2DArray)

        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic_name, Image, callback=self.image_cb, queue_size=1)

        # Cv2 Bridge
        self.cv_bridge = CvBridge()
    
    def trigger(self, req):
        self.is_active = req.data
        status = "ON" if self.is_active else "OFF"
        return SetBoolResponse(success=True, message=f"topic is now {status}")


    def image_cb(self, msg: Image) -> None:
        if not self.is_active:
            return
        
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        header = msg.header

        # results: Results = self.model(cv_image, conf=self.conf_th)[0]
        results: Results = self.model(cv_image, conf=self.conf_th, verbose=False, half=self.half_bool)[0]

        result_bboxes_msg = Detection2DArray()
        result_bboxes_msg.header = header
        result_bboxes_msg.detections = []

        if results.boxes is not None:
            for box_data in results.boxes:
                hyp = ObjectHypothesisWithPose()
                hyp.id = int(box_data.cls[0])
                hyp.score = float(box_data.conf[0])

                xywh = box_data.xywh[0]

                detection = Detection2D()
                detection.header = header
                detection.results = [hyp]
                detection.bbox.center.x = float(xywh[0])
                detection.bbox.center.y = float(xywh[1])
                detection.bbox.size_x = float(xywh[2])
                detection.bbox.size_y = float(xywh[3])
                detection.source_img = msg

                result_bboxes_msg.detections.append(detection)

            if hasattr(results, 'plot'):
                annotated_image = results.plot()
                ros_image = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            else:
                ros_image = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

            ros_image.header = header

            self.result_bboxes_pub.publish(result_bboxes_msg)
            self.result_image_pub.publish(ros_image)
            #rospy.rateを実装

if __name__ == "__main__":
    node = YoloNode()
    rospy.spin()
