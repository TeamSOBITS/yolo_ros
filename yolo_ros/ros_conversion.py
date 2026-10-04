from collections import defaultdict

import cv2
from geometry_msgs.msg import Point
from sobits_interfaces.msg import DetectMask, DetectMaskArray, KeyPoint, KeyPointArray
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from yolo_ros.keypoints import extract_scalar_value, is_keypoint_visible


def extract_track_id(box):
    """Return the tracker id from an Ultralytics box when tracking is active."""
    track_id = getattr(box, "id", None)
    if track_id is None:
        return None
    try:
        return int(extract_scalar_value(track_id))
    except (TypeError, ValueError):
        return None


def detection_id(label, track_id):
    """Detection2D.id is '<class>' in detect mode and '<class>:<track_id>' for tracked boxes."""
    return label if track_id is None else f"{label}:{track_id}"


def result_boxes(result):
    """Axis-aligned boxes, or the oriented boxes of an OBB model (result.boxes is None then)."""
    return result.boxes if result.boxes is not None else getattr(result, "obb", None)


def box_xywhr(box):
    """(center x, center y, width, height, rotation in radians); rotation is 0 for normal boxes."""
    if hasattr(box, "xywhr"):
        x, y, w, h, r = (float(v) for v in box.xywhr[0])
        return x, y, w, h, r
    x, y, w, h = (float(v) for v in box.xywh[0])
    return x, y, w, h, 0.0


def build_detections(result, header, active_filter_classes):
    """Return (Detection2DArray, source box indices) for the boxes passing the class filter."""
    det_array = Detection2DArray(header=header)
    indices = []
    boxes = result_boxes(result)
    if boxes is None:
        return det_array, indices

    for i, box in enumerate(boxes):
        label = result.names[int(extract_scalar_value(box.cls))]
        if active_filter_classes and label not in active_filter_classes:
            continue

        det = Detection2D(header=header)
        det.id = detection_id(label, extract_track_id(box))
        x, y, width, height, rotation = box_xywhr(box)
        det.bbox.center.position.x = x
        det.bbox.center.position.y = y
        # OBB models: size is along the rotated box, theta is its rotation (radians, clockwise
        # in image coordinates as Ultralytics reports it).
        det.bbox.center.theta = rotation
        det.bbox.size_x = width
        det.bbox.size_y = height

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = label
        hyp.hypothesis.score = float(extract_scalar_value(box.conf))
        det.results.append(hyp)

        det_array.detections.append(det)
        indices.append(i)
    return det_array, indices


# Keypoints below this confidence are hidden (out of the image, occluded). Ultralytics still
# returns a guessed position for them, so they are published as (0, 0) like Ultralytics did
# before 8.1 and as consumers expect; z carries the confidence of the visible ones.
KEYPOINT_VISIBLE_CONF = 0.5


def keypoint_point(result, det_idx, kp_idx):
    """Point(x, y, z=confidence) of one keypoint, or (0, 0, 0) when it is hidden."""
    if not is_keypoint_visible(result, det_idx, kp_idx, KEYPOINT_VISIBLE_CONF):
        return Point(x=0.0, y=0.0, z=0.0)
    point = result.keypoints.xy[det_idx][kp_idx]
    conf = result.keypoints.conf
    score = float(extract_scalar_value(conf[det_idx][kp_idx])) if conf is not None else 1.0
    return Point(x=float(point[0]), y=float(point[1]), z=score)


def build_keypoints(result, header, det_array, indices, publish_names, keypoint_names):
    kp_array = KeyPointArray(header=header)
    if result.keypoints is None:
        return kp_array

    name_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    for det, i in zip(det_array.detections, indices):
        kp = KeyPoint(score=det.results[0].hypothesis.score)
        if publish_names and keypoint_names:
            kp.key_names = [name for name in publish_names if name in name_to_idx]
            for name in kp.key_names:
                kp.key_points.append(keypoint_point(result, i, name_to_idx[name]))
        elif publish_names:
            kp.key_names = publish_names
            for index in range(len(result.keypoints[i].xy[0])):
                kp.key_points.append(keypoint_point(result, i, index))
        if kp.key_points:
            kp_array.key_points_array.append(kp)
    return kp_array


def build_masks(result, header, det_array, indices):
    mask_array = DetectMaskArray(header=header)
    if result.masks is None:
        return mask_array

    for det, i in zip(det_array.detections, indices):
        mask = DetectMask(instance_id=det.id)
        mask.results.append(det.results[0])
        mask.pixel_x = [int(x) for x in result.masks[i].xy[0][:, 0]]
        mask.pixel_y = [int(y) for y in result.masks[i].xy[0][:, 1]]
        mask_array.masks.append(mask)
    return mask_array


class TrailDrawer:
    """Keeps per-track position history and draws movement trails on the annotated frame."""

    def __init__(self):
        self.track_history = defaultdict(list)
        self.keypoint_history = defaultdict(list)

    def clear(self):
        self.track_history.clear()
        self.keypoint_history.clear()

    def trim(self, trail_length):
        for key in list(self.track_history):
            self.track_history[key] = self.track_history[key][-trail_length:]
        for key in list(self.keypoint_history):
            self.keypoint_history[key] = self.keypoint_history[key][-trail_length:]

    @staticmethod
    def _append_and_draw(frame, history, center, trail_length, line_width):
        history.append(center)
        if len(history) > trail_length:
            history.pop(0)
        for idx in range(1, len(history)):
            pt1 = (int(history[idx - 1][0]), int(history[idx - 1][1]))
            pt2 = (int(history[idx][0]), int(history[idx][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), line_width)

    def draw_bbox_trails(self, frame, result, trail_length, line_width):
        boxes = result_boxes(result)
        if boxes is None or boxes.id is None:
            return
        track_ids = boxes.id.cpu().numpy()
        boxes = (boxes.xywhr if hasattr(boxes, "xywhr") else boxes.xywh).cpu().numpy()
        for box, track_id in zip(boxes, track_ids):
            self._append_and_draw(
                frame,
                self.track_history[int(track_id)],
                (float(box[0]), float(box[1])),
                trail_length,
                line_width,
            )

    def draw_keypoint_trails(
        self, frame, result, trail_names, keypoint_names, min_conf, trail_length, line_width
    ):
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return
        name_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
        for det_idx, track_id in enumerate(result.boxes.id.cpu().numpy()):
            for name in trail_names:
                kp_idx = name_to_idx.get(name)
                if kp_idx is None or not is_keypoint_visible(result, det_idx, kp_idx, min_conf):
                    continue
                point = result.keypoints[det_idx].xy[0][kp_idx]
                center = (
                    float(extract_scalar_value(point[0])),
                    float(extract_scalar_value(point[1])),
                )
                self._append_and_draw(
                    frame,
                    self.keypoint_history[(int(track_id), name)],
                    center,
                    trail_length,
                    line_width,
                )
