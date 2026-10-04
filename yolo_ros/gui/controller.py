"""
Controls yolo_node, the camera and the image_to_position 3D nodes from the GUI.

Every choice is saved (GuiSettings) and sent to yolo_node when YOLO is started, so the GUI
works the same after a restart. Settings yolo_node accepts while active are applied live; the
model, tracker and image QoS need it inactive, so YOLO is paused for a moment to switch them.
"""

from dataclasses import dataclass
import os
import threading

from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from yolo_ros.gui.catalog import (
    is_pose_model,
    is_yoloe_model,
    MODEL_EXTENSIONS,
)
from yolo_ros.gui.system import (
    camera_topics,
    CameraSource,
    ChildProcess,
    GuiSettings,
    INTERNAL_CAMERA_TOPIC,
    internal_cameras,
    InternalCamera,
    LifecycleClient,
)

YOLO_MODES = ("track", "detect")
YOLO_PARAMETERS = [
    "yolo_mode",
    "weight_file",
    "weights_path",
    "image_topic_name",
    "tracker",
    "tracker_with_reid",
    "tracker_reid_model",
]
# kind -> (node name, yolo_node output it reads, the node's parameter for that topic)
POSITION_NODES = {
    "bbox": ("bbox_to_3d", "object_boxes", "bbox_topic_name"),
    "keypoint": ("keypoint_to_3d", "object_keypoints", "keypoint_topic_name"),
    "mask": ("mask_to_3d", "object_masks", "mask_topic_name"),
}


@dataclass
class SystemStatus:
    camera: CameraSource
    camera_on: bool
    camera_error: str | None
    yolo_state: str | None
    yolo_mode: str | None
    yolo_model: str | None
    busy: str | None
    position_states: dict


class SystemController:
    def __init__(self, node, detector_node, settings_path, image_topic_listener):
        self.node = node
        self.detector_node = detector_node
        self.settings_path = settings_path
        self.settings = GuiSettings.load(settings_path)
        self.image_topic_listener = image_topic_listener
        self.lock = threading.Lock()
        logs = os.path.join(os.path.dirname(settings_path), "logs")

        self.yolo = LifecycleClient(node, detector_node)
        self.yolo_params = AsyncParameterClient(node, detector_node)
        self.internal_camera = InternalCamera(
            node.get_namespace(), os.path.join(logs, "camera.log")
        )
        self.positions = {
            kind: LifecycleClient(node, name) for kind, (name, _, _) in POSITION_NODES.items()
        }
        self.position_params = {
            kind: AsyncParameterClient(node, name)
            for kind, (name, _, _) in POSITION_NODES.items()
        }
        self.position_processes = {
            kind: ChildProcess(os.path.join(logs, f"{name}.log"))
            for kind, (name, _, _) in POSITION_NODES.items()
        }
        self.yolo_values = {}
        self.camera_on = False
        self.camera_error = None
        self.busy = None
        node.create_timer(1.0, self._poll)

    # ------------------------------------------------------------------ status

    def status(self):
        with self.lock:
            yolo_known = self.yolo.state is not None
            return SystemStatus(
                camera=self.saved_camera(),
                camera_on=self.camera_on,
                camera_error=self.camera_error,
                yolo_state=self.yolo.state,
                yolo_mode=self.yolo_values.get("yolo_mode") if yolo_known else None,
                yolo_model=self.yolo_values.get("weight_file") if yolo_known else None,
                busy=self.busy,
                position_states={kind: client.state for kind, client in self.positions.items()},
            )

    def _poll(self):
        self.yolo.poll()
        for client in self.positions.values():
            client.poll()
        if self.yolo_params.services_are_ready():
            self.yolo_params.get_parameters(YOLO_PARAMETERS, callback=self._yolo_values_received)
        if self.internal_camera.exited_with_error() and self.camera_error is None:
            with self.lock:
                self.camera_error = (
                    f"{self.internal_camera.device} を開けませんでした"
                    "（他のプロセスが使っている可能性があります）"
                )
            self.node.log("error", self.camera_error)

    def _yolo_values_received(self, future):
        try:
            values = future.result().values
        except Exception:
            return
        parsed = {}
        for name, value in zip(YOLO_PARAMETERS, values):
            parsed[name] = value.bool_value if name == "tracker_with_reid" else value.string_value
        with self.lock:
            self.yolo_values = parsed

    def _set_many(self, client, values, done=None):
        """
        Set parameters atomically; done(success) is called from the executor thread.

        Atomic matters: yolo_node validates related values (model and keypoint settings,
        tracker and ReID) together, and a rejected set leaves every value unchanged.
        """

        def finished(future):
            try:
                result = future.result().result
                failure = None if result.successful else result.reason
            except Exception as exc:
                failure = str(exc)
            if failure:
                self.node.log("error", f"設定を反映できませんでした: {failure}")
            if done is not None:
                done(failure is None)

        if not client.services_are_ready():
            if done is not None:
                done(False)
            return
        client.set_parameters_atomically(
            [Parameter(name, value=value) for name, value in values.items()], callback=finished
        )

    def _set_live(self, **values):
        """Send values yolo_node accepts at any time (if it is there; else on start)."""
        if self.yolo_params.services_are_ready():
            self._set_many(self.yolo_params, values)

    def save_settings(self, **changes):
        with self.lock:
            for name, value in changes.items():
                setattr(self.settings, name, value)
        try:
            self.settings.save(self.settings_path)
        except OSError as exc:
            self.node.log("warn", f"設定を保存できませんでした: {exc}")

    def _set_busy(self, message):
        with self.lock:
            self.busy = message
        if message:
            self.node.log("info", message)

    # ------------------------------------------------------------------ camera

    def camera_sources(self):
        own_topic = {INTERNAL_CAMERA_TOPIC} if self.internal_camera.running() else set()
        return internal_cameras() + camera_topics(self.node, exclude=own_topic)

    def saved_camera(self):
        if self.settings.camera_kind == "topic":
            value = self.settings.camera_value
            return CameraSource("topic", value, f"トピック: {value}")
        value = self.settings.camera_value or "/dev/video0"
        label = next(
            (source.label for source in internal_cameras() if source.value == value),
            f"内蔵/USBカメラ ({value})",
        )
        return CameraSource("device", value, label)

    def camera_topic(self):
        camera = self.saved_camera()
        return INTERNAL_CAMERA_TOPIC if camera.kind == "device" else camera.value

    def select_camera(self, source):
        self.save_settings(camera_kind=source.kind, camera_value=source.value)
        if self.camera_on:
            self.start_camera()
        else:
            self.node.log("info", f"カメラを「{source.label}」にしました（「起動」で使い始めます）。")

    def set_camera_mode(self, width, height, fps):
        self.save_settings(
            camera_width=int(width), camera_height=int(height), camera_fps=float(fps)
        )
        if self.camera_on and self.saved_camera().kind == "device":
            self.start_camera()

    def set_process_hz(self, hz):
        self.save_settings(process_hz=float(hz))
        self._set_live(max_rate_hz=float(hz))

    def start_camera(self):
        camera = self.saved_camera()
        with self.lock:
            self.camera_error = None
            self.camera_on = True
        if camera.kind == "device":
            settings = self.settings
            self.internal_camera.start(
                camera.value, settings.camera_width, settings.camera_height, settings.camera_fps
            )
        else:
            self.internal_camera.stop()
        topic = self.camera_topic()
        self.image_topic_listener(topic)
        self._set_live(image_topic_name=topic)
        self.node.log("info", f"カメラ「{camera.label}」を起動しました。")

    def stop_camera(self):
        with self.lock:
            self.camera_on = False
            self.camera_error = None
        self.internal_camera.stop()
        self.image_topic_listener(None)
        self.node.log("info", "カメラを止めました。")

    def shutdown(self):
        self.internal_camera.stop()
        for process in self.position_processes.values():
            process.stop()

    # ------------------------------------------------------------------ yolo: values

    def _weights_files(self):
        weights_path = self.yolo_values.get("weights_path", "")
        if not os.path.isdir(weights_path):
            return []
        return sorted(name for name in os.listdir(weights_path) if name.endswith(MODEL_EXTENSIONS))

    def yolo_models(self):
        return [name for name in self._weights_files() if "reid" not in name]

    def reid_models(self):
        return [name for name in self._weights_files() if "reid" in name]

    def current_model(self):
        return self.yolo_values.get("weight_file") or self.settings.yolo_model

    def _keypoint_values(self, model):
        """
        Keypoint and trail settings for `model`.

        yolo_node rejects keypoint settings on a model without keypoints (and with them every
        other value of the same atomic set), so those fall back to bbox-only then.
        """
        settings = self.settings
        pose = is_pose_model(model)
        trail_mode = settings.trail_mode
        if not pose and trail_mode in ("keypoint", "all"):
            trail_mode = "bbox" if trail_mode == "all" else "false"
        return {
            "keypoint_publish_names": (list(settings.keypoint_publish) if pose else []) or [""],
            "trail_mode": trail_mode,
            "trail_length": int(settings.trail_length),
            "line_width": int(settings.line_width),
            "keypoint_trail_names": (list(settings.trail_keypoints) if pose else []) or [""],
            "use_person_keypoint_filter": bool(pose and settings.keypoint_filter),
            "person_keypoint_filter_names": (
                list(settings.keypoint_filter_names) if pose else []
            )
            or [""],
        }

    def _class_values(self):
        classes = list(self.settings.classes)
        return {"use_detection_filter": bool(classes), "filter_classes": classes or [""]}

    def _start_values(self):
        settings = self.settings
        values = {
            "yolo_mode": settings.yolo_mode,
            "conf": float(settings.conf),
            "iou": float(settings.iou),
            "imgsz": int(settings.imgsz),
            "half": bool(settings.half),
            "max_rate_hz": float(settings.process_hz),
            "image_reliability": settings.image_reliability,
            **self._class_values(),
            "tracker": settings.tracker,
            "tracker_with_reid": bool(settings.tracker_reid),
        }
        if self.camera_on:
            values["image_topic_name"] = self.camera_topic()
        if settings.reid_model in self.reid_models():
            values["tracker_reid_model"] = settings.reid_model
        loaded = self.yolo_values.get("weight_file")
        model = settings.yolo_model
        # Sending weight_file always reloads the model, so it is only sent when it changes.
        if model in self.yolo_models() and model != loaded:
            values["weight_file"] = model
            # Keypoint settings are checked against the model loaded before this set, so they
            # are sent off here and for real once the new model is in (start_yolo).
            values.update(self._keypoint_values(None))
        else:
            model = loaded or model
            values.update(self._keypoint_values(model))
        # On another model yoloe_prompts would reload it, so they only go with YOLOE models.
        if is_yoloe_model(model):
            values["yoloe_prompts"] = list(settings.yoloe_prompts) or ["object"]
        return values

    # ------------------------------------------------------------------ yolo: lifecycle

    def start_yolo(self):
        """Apply the saved settings while inactive (some need it), then activate YOLO."""
        if self.busy or self.yolo.state == "active":
            return
        if not self.yolo.ready():
            self.node.log("warn", "YOLO のノードが見つかりません（launch で起動されていません）。")
            return
        self._set_busy("YOLO を起動しています…")

        def activated(success):
            self._set_busy(None)
            if success:
                self._apply_keypoint_settings(self.settings.yolo_model)
            self.node.log(
                "info" if success else "error",
                "YOLO を起動しました。" if success else "YOLO を起動できませんでした。",
            )

        def apply_and_activate():
            self._set_many(
                self.yolo_params, self._start_values(), lambda _: self.yolo.activate(activated)
            )

        if self.yolo.state == "unconfigured":
            self.yolo.configure(lambda ok: apply_and_activate() if ok else self._set_busy(None))
        else:
            apply_and_activate()

    def stop_yolo(self):
        if self.busy or self.yolo.state != "active":
            return

        def done(success):
            self.node.log(
                "info" if success else "error",
                "YOLO を止めました。" if success else "YOLO を止められませんでした。",
            )

        self.yolo.deactivate(done)

    def _set_while_inactive(self, values, description, after=None):
        """Set parameters yolo_node only accepts while inactive, pausing it if it is running."""
        if self.busy or not self.yolo.ready():
            return
        was_active = self.yolo.state == "active"
        self._set_busy(f"YOLO の{description}を切り替え中…")

        def applied(success):
            message = (
                f"YOLO の{description}を切り替えました。"
                if success
                else f"YOLO の{description}を切り替えられませんでした（元の設定のままです）。"
            )

            def finish(_=True):
                self._set_busy(None)
                self.node.log("info" if success else "error", message)
                if success and after is not None:
                    after()

            if was_active:
                self.yolo.activate(finish)
            else:
                finish()

        if was_active:
            self.yolo.deactivate(
                lambda ok: (
                    self._set_many(self.yolo_params, values, applied) if ok else applied(False)
                )
            )
        else:
            self._set_many(self.yolo_params, values, applied)

    # ------------------------------------------------------------------ yolo: settings

    def set_yolo_model(self, model):
        self.save_settings(yolo_model=model)
        if self.yolo_values.get("weight_file") == model:
            return
        # Keypoint settings off for the switch, then the real ones for the new model.
        values = {"weight_file": model, **self._keypoint_values(None)}
        if is_yoloe_model(model):
            values["yoloe_prompts"] = list(self.settings.yoloe_prompts) or ["object"]
        self._set_while_inactive(
            values, "モデル", after=lambda: self._apply_keypoint_settings(model)
        )

    def _apply_keypoint_settings(self, model=None):
        self._set_live(**self._keypoint_values(model or self.current_model()))

    def set_keypoint_settings(self, **changes):
        """Save and apply trail / keypoint settings (live; YOLO need not stop)."""
        self.save_settings(**changes)
        self._apply_keypoint_settings()

    def set_classes(self, classes):
        """Detect only these classes (empty: all). Applied live."""
        self.save_settings(classes=list(classes))
        self._set_live(**self._class_values())

    def set_yoloe_prompts(self, prompts):
        self.save_settings(yoloe_prompts=list(prompts))
        if prompts and is_yoloe_model(self.yolo_values.get("weight_file")):
            self._set_live(yoloe_prompts=list(prompts))

    def set_tracker(self, tracker, with_reid, reid_model):
        self.save_settings(tracker=tracker, tracker_reid=bool(with_reid), reid_model=reid_model)
        values = {"tracker": tracker, "tracker_with_reid": bool(with_reid)}
        if reid_model:
            values["tracker_reid_model"] = reid_model
        self._set_while_inactive(values, "トラッカー")

    def set_image_reliability(self, reliability):
        self.save_settings(image_reliability=reliability)
        self._set_while_inactive({"image_reliability": reliability}, "画像の受信方法")

    def set_inference(self, **changes):
        """conf / iou / imgsz / half: saved and applied live."""
        self.save_settings(**changes)
        self._set_live(**changes)

    def set_yolo_mode(self, mode):
        if mode not in YOLO_MODES:
            raise ValueError(f"mode must be one of {YOLO_MODES}")
        self.save_settings(yolo_mode=mode)
        # Re-sending the current mode is skipped so that track ids are never touched.
        if self.yolo_values.get("yolo_mode") == mode:
            return

        def done(success):
            if success:
                with self.lock:
                    self.yolo_values["yolo_mode"] = mode
                self.node.log("info", f"YOLO を {mode} モードにしました。")

        if self.yolo_params.services_are_ready():
            self._set_many(self.yolo_params, {"yolo_mode": mode}, done)

    # ------------------------------------------------------------------ 3D (image_to_position)

    def position_input_topic(self, kind):
        return f"{self.detector_node}/{POSITION_NODES[kind][1]}"

    def start_position(self, kind):
        """Start bbox/keypoint/mask_to_3d: its lifecycle if running, else launch it."""
        name, _, topic_parameter = POSITION_NODES[kind]
        client = self.positions[kind]
        topic = self.position_input_topic(kind)

        def done(success):
            self.node.log(
                "info" if success else "error",
                f"{name} を起動しました（入力: {topic}）。"
                if success
                else f"{name} を起動できませんでした（ログを確認してください）。",
            )

        if client.ready():
            if client.state == "unconfigured":
                # The node reads its input topic on configure.
                self._set_many(
                    self.position_params[kind],
                    {topic_parameter: topic},
                    lambda _: client.start(done),
                )
            else:
                client.start(done)
            return
        namespace = self.node.get_namespace()
        command = [
            "ros2", "launch", "image_to_position", f"{name}.launch.py",
            "auto_activate:=true", f"{topic_parameter}:={topic}",
        ]
        if namespace not in ("", "/"):
            command.append(f"namespace:={namespace.strip('/')}")
        self.position_processes[kind].start(command)
        self.node.log("info", f"{name} を起動しています（入力: {topic}）…")

    def stop_position(self, kind):
        name = POSITION_NODES[kind][0]
        self.positions[kind].deactivate(
            lambda ok: self.node.log(
                "info" if ok else "error",
                f"{name} を止めました。" if ok else f"{name} を止められませんでした。",
            )
        )
