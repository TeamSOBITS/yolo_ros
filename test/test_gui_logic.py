import json

from yolo_ros import tracking
from yolo_ros.gui.catalog import describe, is_pose_model, is_yoloe_model, sort_models
from yolo_ros.gui.controller import SystemController
from yolo_ros.gui.system import GuiSettings
from yolo_ros.stats import inference_ms, ProcessingStats


def controller(**settings):
    """A SystemController without ROS, enough for the value-building methods."""
    system = object.__new__(SystemController)
    system.settings = GuiSettings(**settings)
    system.yolo_values = {}
    system.camera_on = False
    return system


def test_reid_is_ignored_for_trackers_without_reid():
    assert tracking.reid_enabled("tracktrack.yaml", True)
    assert not tracking.reid_enabled("bytetrack.yaml", True)
    assert not tracking.reid_enabled("tracktrack.yaml", False)
    # ByteTrack with the default tracker_with_reid=true used to be rejected.
    assert tracking.validate_tracker_settings("bytetrack.yaml", True, "missing.onnx", "/x") == (
        True,
        "",
    )
    assert not tracking.validate_tracker_settings("tracktrack.yaml", True, "missing.onnx", "/x")[0]
    assert not tracking.validate_tracker_settings("nope.yaml", False, "auto", "")[0]


def test_stats_window():
    stats = ProcessingStats()
    stats.frame_received()
    stats.frame_received()
    stats.frame_processed(10.0, 12.0, None, 3)
    stats.frame_processed(20.0, 22.0, 30.0, 1)
    data = json.loads(stats.to_json(imgsz=640))
    assert data["inference_ms"] == 15.0
    assert data["callback_ms"] == 17.0
    assert data["latency_ms"] == 30.0
    assert data["detections"] == 1
    assert data["imgsz"] == 640
    assert data["received_fps"] > 0
    # A new window starts empty but keeps the last detection count.
    data = json.loads(stats.to_json())
    assert data["inference_ms"] is None and data["processed_fps"] == 0.0
    assert data["detections"] == 1


def test_inference_ms():
    class Result:
        speed = {"preprocess": 1.0, "inference": 5.5, "postprocess": None}

    assert inference_ms(Result()) == 6.5
    assert inference_ms(object()) == 0


def test_model_catalog():
    assert is_pose_model("yolo26m-pose.pt")
    assert not is_pose_model("yolo26m.pt")
    assert is_yoloe_model("yoloe-26s-seg.pt")
    assert describe("yolo26n-obb.pt").usable
    assert not describe("yolo26n-cls.pt").usable
    assert describe("custom.onnx").usable
    assert sort_models(
        ["yolo26n-cls.pt", "yolo26n-obb.pt", "yolo26m-pose.pt", "yolo26n.pt", "yolo26m.pt"]
    ) == ["yolo26n.pt", "yolo26m.pt", "yolo26m-pose.pt", "yolo26n-obb.pt", "yolo26n-cls.pt"]


def test_keypoint_values_fall_back_without_pose():
    system = controller(trail_mode="all", keypoint_filter=True, trail_keypoints=["nose"])
    values = system._keypoint_values("yolo26m.pt")
    assert values["trail_mode"] == "bbox"
    assert values["use_person_keypoint_filter"] is False
    assert values["keypoint_trail_names"] == [""]
    assert values["person_keypoint_filter_names"] == [""]
    values = system._keypoint_values("yolo26m-pose.pt")
    assert values["trail_mode"] == "all"
    assert values["use_person_keypoint_filter"] is True
    assert values["keypoint_trail_names"] == ["nose"]


def test_start_values_send_prompts_only_to_yoloe():
    system = controller(yolo_model="yolo26m.pt", classes=["person"])
    values = system._start_values()
    assert "yoloe_prompts" not in values  # would reload a non-YOLOE model
    assert "image_topic_name" not in values  # the GUI camera is off
    assert values["filter_classes"] == ["person"] and values["use_detection_filter"]
    system = controller(yolo_model="yoloe-26s-seg.pt", yoloe_prompts=["cup"])
    system.yolo_values = {"weight_file": "yoloe-26s-seg.pt"}
    assert system._start_values()["yoloe_prompts"] == ["cup"]


def test_settings_round_trip(tmp_path):
    path = str(tmp_path / "gui_settings.json")
    GuiSettings(imgsz=960, half=True, classes=["bottle"]).save(path)
    with open(path) as settings_file:
        data = json.load(settings_file)
    data["removed_setting"] = 1
    with open(path, "w") as settings_file:
        json.dump(data, settings_file)
    loaded = GuiSettings.load(path)
    assert (loaded.imgsz, loaded.half, loaded.classes) == (960, True, ["bottle"])
    assert GuiSettings.load(str(tmp_path / "missing.json")) == GuiSettings()


def test_obb_detection_keeps_rotation():
    import torch
    from std_msgs.msg import Header
    from ultralytics.engine.results import Results
    from yolo_ros.ros_conversion import build_detections

    import numpy as np

    obb = torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.5, 7.0, 0.9, 1.0]])  # x y w h r id conf cls
    result = Results(np.zeros((200, 200, 3), np.uint8), path="", names={0: "plane", 1: "ship"},
                     obb=obb)
    det_array, indices = build_detections(result, Header(), [])
    assert indices == [0]
    det = det_array.detections[0]
    assert det.id == "ship:7"
    assert (det.bbox.size_x, det.bbox.size_y) == (40.0, 20.0)
    assert abs(det.bbox.center.theta - 0.5) < 1e-6
