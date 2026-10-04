"""Helpers the GUI uses to control the system: lifecycles, child processes, cameras, settings."""

from dataclasses import asdict, dataclass, field
import glob
import json
import os
import signal
import subprocess

from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from yolo_ros.keypoints import POSE_KEYPOINT_NAMES_17

IMAGE_TYPE = "sensor_msgs/msg/Image"
INTERNAL_CAMERA_TOPIC = "/image_raw"
# Image topics yolo_ros / osnet_ros publish themselves; they are not camera sources.
OWN_IMAGE_TOPIC_SUFFIXES = ("/detected_image", "/detect_request")


def default_data_root():
    """Where the GUI keeps its settings and screenshots (YOLO_ROS_DATA_ROOT overrides it)."""
    return os.environ.get("YOLO_ROS_DATA_ROOT") or os.path.join(
        os.path.expanduser("~"), ".local", "share", "yolo_ros"
    )


class LifecycleClient:
    """Async get_state / change_state for a remote lifecycle node."""

    def __init__(self, node, target):
        self.target = target
        self.get_client = node.create_client(GetState, f"{target}/get_state")
        self.change_client = node.create_client(ChangeState, f"{target}/change_state")
        self.state = None

    def ready(self):
        return self.get_client.service_is_ready() and self.change_client.service_is_ready()

    def poll(self):
        if not self.get_client.service_is_ready():
            self.state = None
            return
        self.get_client.call_async(GetState.Request()).add_done_callback(self._state_received)

    def _state_received(self, future):
        try:
            self.state = future.result().current_state.label
        except Exception:
            self.state = None

    def change(self, transition_id, done):
        """Run a transition; done(success: bool) is called from the executor thread."""
        request = ChangeState.Request()
        request.transition.id = transition_id

        def finished(future):
            try:
                success = future.result().success
            except Exception:
                success = False
            self.poll()
            done(success)

        self.change_client.call_async(request).add_done_callback(finished)

    def configure(self, done):
        self.change(Transition.TRANSITION_CONFIGURE, done)

    def activate(self, done):
        self.change(Transition.TRANSITION_ACTIVATE, done)

    def deactivate(self, done):
        self.change(Transition.TRANSITION_DEACTIVATE, done)

    def start(self, done):
        """Configure if needed, then activate."""
        if self.state == "unconfigured":
            self.configure(lambda ok: self.activate(done) if ok else done(False))
        else:
            self.activate(done)


@dataclass(frozen=True)
class CameraSource:
    kind: str  # 'device' or 'topic'
    value: str = ""
    label: str = ""

    @property
    def key(self):
        return (self.kind, self.value)


def internal_cameras():
    """List video capture devices (video4linux entries with index 0 are the capture nodes)."""
    cameras = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            with open(os.path.join(path, "index")) as index_file:
                if index_file.read().strip() != "0":
                    continue
            with open(os.path.join(path, "name")) as name_file:
                name = name_file.read().strip()
        except OSError:
            continue
        device = "/dev/" + os.path.basename(path)
        cameras.append(CameraSource("device", device, f"内蔵/USBカメラ: {name} ({device})"))
    return cameras


def camera_topics(node, exclude=()):
    topics = []
    for topic, types in node.get_topic_names_and_types():
        if IMAGE_TYPE not in types or topic in exclude:
            continue
        if topic.endswith(OWN_IMAGE_TOPIC_SUFFIXES):
            continue
        topics.append(CameraSource("topic", topic, f"トピック: {topic}"))
    return topics


def camera_modes(device):
    """[(width, height, max fps)] the device offers (MJPG preferred), largest first."""
    try:
        output = subprocess.run(
            ["v4l2-ctl", "--list-formats-ext", "-d", device],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        output = ""
    modes, size, in_mjpg = {}, None, False
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("["):
            in_mjpg = "'MJPG'" in line
        elif line.startswith("Size:") and in_mjpg:
            size = tuple(int(v) for v in line.split()[-1].split("x"))
        elif line.startswith("Interval:") and in_mjpg and size:
            fps = float(line.split("(")[-1].split()[0])
            modes[size] = max(modes.get(size, 0.0), fps)
    if not modes:
        modes = {(1280, 720): 30.0, (640, 480): 30.0, (320, 240): 30.0}
    return sorted(((w, h, fps) for (w, h), fps in modes.items()), reverse=True)


class ChildProcess:
    """A ROS command run in its own process group so that it can be stopped cleanly."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.process = None

    def running(self):
        return self.process is not None and self.process.poll() is None

    def exited_with_error(self):
        return self.process is not None and self.process.poll() not in (None, 0, -signal.SIGINT)

    def start(self, command):
        self.stop()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as log_file:
            self.process = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )

    def stop(self):
        if not self.running():
            self.process = None
            return
        os.killpg(self.process.pid, signal.SIGINT)
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        self.process = None


def namespace_arguments(namespace):
    return ["-r", f"__ns:={namespace}"] if namespace not in ("", "/") else []


class InternalCamera(ChildProcess):
    """Runs yolo_camera for a local camera device as a child process."""

    def __init__(self, namespace, log_path):
        super().__init__(log_path)
        self.namespace = namespace
        self.device = None

    def start(self, device, width, height, fps):
        command = [
            "ros2", "run", "yolo_ros", "yolo_camera", "--ros-args",
            "-p", f"video_device:={device}",
            "-p", f"width:={int(width)}",
            "-p", f"height:={int(height)}",
            "-p", f"fps:={float(fps)}",
            "-r", f"image_raw:={INTERNAL_CAMERA_TOPIC}",
        ] + namespace_arguments(self.namespace)
        super().start(command)
        self.device = device

    def stop(self):
        super().stop()
        self.device = None


@dataclass
class GuiSettings:
    """Last choices made in the GUI; they are sent to yolo_node whenever YOLO is started."""

    camera_kind: str = "device"
    camera_value: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    process_hz: float = 0.0
    image_reliability: str = "auto"
    yolo_model: str = ""
    yolo_mode: str = "track"
    conf: float = 0.5
    iou: float = 0.7
    imgsz: int = 640
    half: bool = False
    classes: list = field(default_factory=list)  # empty = every class the model knows
    yoloe_prompts: list = field(default_factory=lambda: ["person", "bottle"])
    tracker: str = "tracktrack.yaml"
    tracker_reid: bool = True
    reid_model: str = "yolo26m-reid.onnx"
    trail_mode: str = "bbox"
    trail_length: int = 30
    line_width: int = 2
    trail_keypoints: list = field(default_factory=lambda: ["left_wrist", "right_wrist"])
    keypoint_publish: list = field(default_factory=lambda: list(POSE_KEYPOINT_NAMES_17))
    keypoint_filter: bool = False
    keypoint_filter_names: list = field(default_factory=lambda: ["nose"])
    view_layout: int = 1
    view_sources: list = field(default_factory=lambda: ["yolo", "camera"])

    @classmethod
    def load(cls, path):
        try:
            with open(path) as settings_file:
                data = json.load(settings_file)
        except (OSError, ValueError):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as settings_file:
            json.dump(asdict(self), settings_file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
