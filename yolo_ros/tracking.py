import contextlib
import os
from pathlib import Path
import tempfile

import yaml

TRACKER_CHOICES = (
    "botsort.yaml",
    "bytetrack.yaml",
    "ocsort.yaml",
    "deepocsort.yaml",
    "fasttrack.yaml",
    "tracktrack.yaml",
)
REID_TRACKER_CHOICES = ("botsort.yaml", "deepocsort.yaml", "tracktrack.yaml")

_TRACKER_CALLBACK_EVENTS = ("on_predict_start", "on_predict_postprocess_end")
_TRACKER_CALLBACK_MODULE = "ultralytics.trackers.track"


def supports_tracker_reid(tracker):
    return tracker in REID_TRACKER_CHOICES


def resolve_reid_model(model_name, weights_path):
    model = model_name or "auto"
    if model == "auto":
        return model
    if (
        os.path.isabs(model)
        or os.path.sep in model
        or (os.path.altsep and os.path.altsep in model)
    ):
        return model
    return os.path.join(weights_path, model)


def reid_enabled(tracker, use_reid):
    """
    tracker_with_reid means "use ReID when the tracker supports it".

    Trackers without ReID (ByteTrack, OC-SORT, FastTrack) simply run without it, so switching the
    tracker never needs tracker_with_reid to be changed in the same request.
    """
    return bool(use_reid) and supports_tracker_reid(tracker)


def validate_tracker_settings(tracker, use_reid, reid_model, reid_weights_path):
    if tracker not in TRACKER_CHOICES:
        return False, f"tracker must be one of {TRACKER_CHOICES}, got '{tracker}'"
    use_reid = reid_enabled(tracker, use_reid)

    resolved_model = resolve_reid_model(reid_model, reid_weights_path)
    if use_reid and resolved_model != "auto" and not os.path.exists(resolved_model):
        return False, f"tracker_reid_model not found: {resolved_model}"
    return True, ""


def write_reid_tracker_config(tracker, reid_model_path):
    """Write a temporary copy of the tracker yaml with ReID enabled and return its path."""
    # Imported here so that the GUI can use the tracker lists without loading Ultralytics.
    from ultralytics.utils.checks import check_yaml

    tracker_cfg = yaml.safe_load(Path(check_yaml(tracker)).read_text())
    tracker_cfg["with_reid"] = True
    tracker_cfg["model"] = reid_model_path

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="yolo_ros_tracker_", suffix=".yaml", delete=False
    ) as tmp:
        yaml.safe_dump(tracker_cfg, tmp, sort_keys=False)
        return tmp.name


def remove_file_quietly(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def reset_tracker_state(model):
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        delattr(predictor, "trackers")


@contextlib.contextmanager
def tracker_disabled(model):
    """
    Run predict() without the tracker callbacks that model.track() leaves registered.

    Ultralytics keeps the tracker callbacks on the model after the first track() call, so a
    plain predict() would otherwise still advance (and corrupt) the live tracker state.
    """
    callbacks = getattr(model, "callbacks", None)
    if callbacks is None:
        yield
        return

    saved = {event: list(callbacks[event]) for event in _TRACKER_CALLBACK_EVENTS}
    try:
        for event in _TRACKER_CALLBACK_EVENTS:
            callbacks[event][:] = [
                callback
                for callback in callbacks[event]
                if getattr(getattr(callback, "func", None), "__module__", "")
                != _TRACKER_CALLBACK_MODULE
            ]
        yield
    finally:
        for event, event_callbacks in saved.items():
            callbacks[event][:] = event_callbacks
