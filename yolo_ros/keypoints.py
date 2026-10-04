POSE_KEYPOINT_NAMES_17 = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def active_names(names):
    return [name for name in names if name]


def model_keypoint_shape(model):
    return getattr(model, "kpt_shape", None)


def model_keypoint_names(model):
    kpt_shape = model_keypoint_shape(model)
    if kpt_shape is None:
        return []
    if kpt_shape[0] == len(POSE_KEYPOINT_NAMES_17):
        return list(POSE_KEYPOINT_NAMES_17)
    return []


def extract_scalar_value(value):
    """Return a scalar from tensor/array/scalar YOLO outputs."""
    if hasattr(value, "item"):
        return value.item()
    try:
        return value[0]
    except (TypeError, IndexError, KeyError):
        return value


def is_keypoint_visible(result, det_idx, kp_idx, min_conf):
    conf = result.keypoints.conf
    if conf is not None:
        return float(extract_scalar_value(conf[det_idx][kp_idx])) >= min_conf

    point = result.keypoints.xy[det_idx][kp_idx]
    x = float(extract_scalar_value(point[0]))
    y = float(extract_scalar_value(point[1]))
    return not (x <= 0.0 and y <= 0.0)


def apply_keypoint_filter(result, filter_names, keypoint_names, min_conf):
    """Keep only detections whose filter keypoints are all visible."""
    if not filter_names or result.keypoints is None or result.boxes is None:
        return result

    name_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    target_indices = [name_to_idx[name] for name in filter_names]
    keep_indices = [
        det_idx
        for det_idx in range(len(result.boxes))
        if all(is_keypoint_visible(result, det_idx, kp_idx, min_conf) for kp_idx in target_indices)
    ]
    if len(keep_indices) == len(result.boxes):
        return result
    return result[keep_indices]


def validate_keypoint_settings(model, publish_names, filter_names, trail_names):
    if not filter_names and not trail_names:
        return True, ""

    kpt_shape = model_keypoint_shape(model)
    if kpt_shape is None:
        if filter_names:
            return False, "person keypoint filtering requires a pose model with keypoints"
        return False, "keypoint trails require a pose model with keypoints"

    keypoint_names = model_keypoint_names(model)
    if not keypoint_names:
        return (
            False,
            "No built-in keypoint name mapping is available for pose model "
            f"with {kpt_shape[0]} keypoints",
        )

    for param_name, names in (
        ("keypoint_publish_names", publish_names),
        ("person_keypoint_filter_names", filter_names),
        ("keypoint_trail_names", trail_names),
    ):
        missing = [name for name in names if name not in keypoint_names]
        if missing:
            return False, f"{param_name} contains unknown names: {missing}"
    return True, ""


def keypoint_name_warnings(model, named_lists):
    """Return warnings for keypoint name parameters given as (param_name, names) pairs."""
    kpt_shape = model_keypoint_shape(model)
    if kpt_shape is None:
        return []

    keypoint_names = model_keypoint_names(model)
    warnings = []
    for param_name, names in named_lists:
        active = active_names(names)
        if not active:
            continue
        if not keypoint_names:
            warnings.append(
                f"Pose model detected ({kpt_shape[0]} keypoints) "
                "but no built-in keypoint name mapping is available"
            )
            continue
        unknown = [name for name in active if name not in keypoint_names]
        if unknown:
            warnings.append(f"{param_name} contains unknown names: {unknown}")
    return warnings
