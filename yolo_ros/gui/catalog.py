"""Choices the GUI offers: model files, trackers, trail modes, keypoints and class names."""

from dataclasses import dataclass
import re

from yolo_ros.keypoints import POSE_KEYPOINT_NAMES_17
from yolo_ros.tracking import REID_TRACKER_CHOICES

MODEL_EXTENSIONS = (".pt", ".engine", ".onnx", ".torchscript")
NAME = re.compile(r"^yolo(?:v)?(?P<version>\d+)(?P<size>[nsmlx])(?:-(?P<task>[a-z]+))?\.")
SIZES = {
    "n": (0, "nano: 最速・最軽量。小さい物や遠くの物を見逃しやすい"),
    "s": (1, "small: 速くてそこそこ正確"),
    "m": (2, "medium: 精度と速さのバランスが良い"),
    "l": (3, "large: 正確だが重い"),
    "x": (4, "xlarge: 最も正確だが非常に重い"),
}
# task -> (order in the list, usable by yolo_node, description)
TASKS = {
    None: (0, True, "物体検出: 枠（object_boxes）を出す"),
    "pose": (1, True, "姿勢推定: 枠と人の 17 点（object_keypoints）を出す。導線・人とみなす条件に使える"),
    "seg": (2, True, "セグメンテーション: 枠と輪郭（object_masks）を出す。mask_to_3d に使える"),
    "obb": (3, True, "回転した枠: 斜めの車や船などを回転した枠で囲む（角度は bbox.center.theta）。"
            "標準のモデルは空撮（DOTA）で学習しているので、上から見下ろす映像向け"),
    "cls": (9, False, "画像分類用。枠を出さないので使えない"),
}

TRACKERS = {
    "tracktrack.yaml": "TrackTrack",
    "botsort.yaml": "BoT-SORT",
    "deepocsort.yaml": "Deep OC-SORT",
    "bytetrack.yaml": "ByteTrack",
    "ocsort.yaml": "OC-SORT",
    "fasttrack.yaml": "FastTrack",
}
REID_TRACKERS = REID_TRACKER_CHOICES
TRAIL_MODES = {
    "false": "なし",
    "bbox": "枠の中心",
    "keypoint": "キーポイント",
    "all": "両方",
}
IMAGE_SIZES = (320, 416, 480, 640, 800, 960, 1280)
RELIABILITIES = {
    "auto": "自動（reliable と best_effort の両方。おすすめ）",
    "reliable": "reliable",
    "best_effort": "best_effort",
}

POSE_KEYPOINTS_JA = (
    "鼻", "左目", "右目", "左耳", "右耳", "左肩", "右肩", "左ひじ", "右ひじ",
    "左手首", "右手首", "左腰", "右腰", "左ひざ", "右ひざ", "左足首", "右足首",
)
POSE_KEYPOINTS = dict(zip(POSE_KEYPOINT_NAMES_17, POSE_KEYPOINTS_JA))


@dataclass(frozen=True)
class ModelInfo:
    filename: str
    usable: bool
    task: str  # 'detect', 'pose', 'seg', 'yoloe', 'obb', 'cls' or 'unknown'
    description: str
    order: tuple

    @property
    def title(self):
        return self.filename if self.usable else f"{self.filename}（使えない）"


def describe(filename):
    if filename.startswith("yoloe"):
        task = "yoloe"
        text = "YOLOE: 文字（プロンプト）で検出する物体を決めるモデル。好きな物の名前で探せる"
        return ModelInfo(filename, True, task, text, (3, 0, filename))
    match = NAME.match(filename)
    if match is None:
        return ModelInfo(filename, True, "unknown", "独自のモデル", (7, 0, filename))
    order, usable, task_text = TASKS.get(
        match["task"], (7, True, f"{match['task']} 用のモデル")
    )
    size_order, size_text = SIZES[match["size"]]
    task = match["task"] or "detect"
    text = f"YOLO{match['version']} / {task_text}\n{size_text}"
    return ModelInfo(filename, usable, task, text, (order, size_order, filename))


def sort_models(filenames):
    """Detection, pose, segmentation, YOLOE, then unusable ones; small to large within a task."""
    return sorted(filenames, key=lambda name: describe(name).order)


def is_pose_model(filename):
    return describe(filename or "").task == "pose"


def is_yoloe_model(filename):
    return describe(filename or "").task == "yoloe"


# Japanese names of the COCO classes (the standard YOLO models); other names are shown as is.
CLASS_NAMES_JA = {
    "person": "人", "bicycle": "自転車", "car": "車", "motorcycle": "バイク", "airplane": "飛行機",
    "bus": "バス", "train": "電車", "truck": "トラック", "boat": "船", "traffic light": "信号機",
    "fire hydrant": "消火栓", "stop sign": "一時停止標識", "parking meter": "パーキングメーター",
    "bench": "ベンチ", "bird": "鳥", "cat": "猫", "dog": "犬", "horse": "馬", "sheep": "羊",
    "cow": "牛", "elephant": "象", "bear": "熊", "zebra": "シマウマ", "giraffe": "キリン",
    "backpack": "リュック", "umbrella": "傘", "handbag": "ハンドバッグ", "tie": "ネクタイ",
    "suitcase": "スーツケース", "frisbee": "フリスビー", "skis": "スキー板",
    "snowboard": "スノーボード", "sports ball": "ボール", "kite": "凧", "baseball bat": "バット",
    "baseball glove": "グローブ", "skateboard": "スケートボード", "surfboard": "サーフボード",
    "tennis racket": "テニスラケット", "bottle": "ボトル", "wine glass": "ワイングラス",
    "cup": "カップ", "fork": "フォーク", "knife": "ナイフ", "spoon": "スプーン", "bowl": "ボウル",
    "banana": "バナナ", "apple": "りんご", "sandwich": "サンドイッチ", "orange": "オレンジ",
    "broccoli": "ブロッコリー", "carrot": "にんじん", "hot dog": "ホットドッグ", "pizza": "ピザ",
    "donut": "ドーナツ", "cake": "ケーキ", "chair": "椅子", "couch": "ソファ",
    "potted plant": "鉢植え", "bed": "ベッド", "dining table": "テーブル", "toilet": "トイレ",
    "tv": "テレビ", "laptop": "ノートPC", "mouse": "マウス", "remote": "リモコン",
    "keyboard": "キーボード", "cell phone": "スマホ", "microwave": "電子レンジ", "oven": "オーブン",
    "toaster": "トースター", "sink": "シンク", "refrigerator": "冷蔵庫", "book": "本",
    "clock": "時計", "vase": "花瓶", "scissors": "はさみ", "teddy bear": "ぬいぐるみ",
    "hair drier": "ドライヤー", "toothbrush": "歯ブラシ",
}


def class_label(name):
    japanese = CLASS_NAMES_JA.get(name)
    return f"{japanese}（{name}）" if japanese else name
