import json
import time


class ProcessingStats:
    """
    Per-second processing figures of yolo_node, published as JSON on <node>/stats.

    received_fps: camera frames that arrived; processed_fps: frames that went through YOLO
    (lower when max_rate_hz drops frames or inference is slower than the camera);
    inference_ms: Ultralytics preprocess + inference + postprocess; callback_ms: the whole image
    callback including drawing and publishing; latency_ms: from the image header stamp to the
    moment the results were published (only meaningful when the camera stamps with the same
    clock); detections: boxes published for the last frame.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._started = time.monotonic()
        self._received = 0
        self._processed = 0
        self._inference_ms = []
        self._callback_ms = []
        self._latency_ms = []
        self.detections = 0

    def frame_received(self):
        self._received += 1

    def frame_processed(self, inference_ms, callback_ms, latency_ms, detections):
        self._processed += 1
        self._inference_ms.append(inference_ms)
        self._callback_ms.append(callback_ms)
        if latency_ms is not None:
            self._latency_ms.append(latency_ms)
        self.detections = detections

    def to_json(self, **extra):
        """Return the figures since the last call as JSON and start a new window."""
        elapsed = max(time.monotonic() - self._started, 1e-6)
        data = {
            "received_fps": round(self._received / elapsed, 2),
            "processed_fps": round(self._processed / elapsed, 2),
            "inference_ms": _mean(self._inference_ms),
            "callback_ms": _mean(self._callback_ms),
            "latency_ms": _mean(self._latency_ms),
            "detections": self.detections,
            **extra,
        }
        detections = self.detections
        self.reset()
        self.detections = detections
        return json.dumps(data)


def _mean(values):
    return round(sum(values) / len(values), 1) if values else None


def inference_ms(result):
    """Preprocess + inference + postprocess time Ultralytics measured for one result."""
    speed = getattr(result, "speed", None) or {}
    return round(sum(value for value in speed.values() if value is not None), 1)
