from PySide6.QtWidgets import QCheckBox, QSpinBox, QWidget
from yolo_ros.gui.catalog import (
    is_pose_model,
    POSE_KEYPOINTS,
    REID_TRACKERS,
    TRACKERS,
    TRAIL_MODES,
)
from yolo_ros.gui.qt_utils import card, label
from yolo_ros.gui.widgets import compact_combo, KeypointChecks, page_layout, set_combo_items

TRACKER_NOTES = {
    "tracktrack.yaml": "最新・高精度。ReID 対応",
    "botsort.yaml": "動き＋カメラの揺れ補正。ReID 対応",
    "deepocsort.yaml": "隠れても追いやすい。ReID 対応",
    "bytetrack.yaml": "軽くて速い。ReID なし",
    "ocsort.yaml": "動きの予測が強い。ReID なし",
    "fasttrack.yaml": "最も軽い。ReID なし",
}


def int_spin(value, minimum, maximum, suffix):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSuffix(suffix)
    box.setValue(int(value))
    return box


class TrackingPage(QWidget):
    """Tracker, tracker ReID, and the movement trails drawn on YOLO's image."""

    def __init__(self, node):
        super().__init__()
        self.system = node.system
        self.trackers = list(TRACKERS)
        self.reid_models = []
        settings = self.system.settings
        layout = page_layout(self)

        layout.addWidget(label("トラッカー", "Section"))
        self.tracker_combo = compact_combo()
        set_combo_items(
            self.tracker_combo,
            [f"{TRACKERS[name]} — {TRACKER_NOTES[name]}" for name in self.trackers],
            self.trackers.index(settings.tracker) if settings.tracker in TRACKERS else 0,
        )
        self.tracker_combo.activated.connect(lambda _: self._tracker_changed())
        self.reid_check = QCheckBox("ReID（見た目の特徴）も使う")
        self.reid_check.setChecked(bool(settings.tracker_reid))
        self.reid_check.toggled.connect(lambda _: self._tracker_changed())
        self.reid_combo = compact_combo()
        self.reid_combo.activated.connect(lambda _: self._tracker_changed())
        self.tracker_note = label("", "Subtitle")
        layout.addWidget(
            card(
                self.tracker_combo,
                self.reid_check,
                label("ReID モデル", "Subtitle"),
                self.reid_combo,
                self.tracker_note,
                label(
                    "ReID は「さっきの人と同じか」を見た目でも確かめ、隠れた後も同じ ID を保ち"
                    "やすくします（その分重くなります）。ReID のないトラッカーではこの設定は"
                    "無視されます。トラッカーを変えると YOLO を一瞬止め、ID は振り直されます。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("導線（動いた跡）", "Section"))
        self.trail_modes = list(TRAIL_MODES)
        self.trail_combo = compact_combo()
        set_combo_items(
            self.trail_combo,
            list(TRAIL_MODES.values()),
            self.trail_modes.index(settings.trail_mode)
            if settings.trail_mode in TRAIL_MODES
            else 1,
        )
        self.trail_combo.activated.connect(
            lambda i: self.system.set_keypoint_settings(trail_mode=self.trail_modes[i])
        )
        self.length_spin = int_spin(settings.trail_length, 2, 600, " フレーム")
        self.length_spin.editingFinished.connect(
            lambda: self.system.set_keypoint_settings(trail_length=self.length_spin.value())
        )
        self.width_spin = int_spin(settings.line_width, 1, 10, " px")
        self.width_spin.editingFinished.connect(
            lambda: self.system.set_keypoint_settings(line_width=self.width_spin.value())
        )
        self.trail_points = KeypointChecks(POSE_KEYPOINTS, settings.trail_keypoints)
        self.trail_points.changed.connect(
            lambda names: self.system.set_keypoint_settings(trail_keypoints=names)
        )
        self.trail_note = label("", "Subtitle")
        layout.addWidget(
            card(
                label("導線を描く点", "Subtitle"),
                self.trail_combo,
                label("長さ（何フレーム分の跡を残すか）", "Subtitle"),
                self.length_spin,
                label("線の太さ（枠と導線）", "Subtitle"),
                self.width_spin,
                label("キーポイントの導線を描く点（-pose のモデルのとき）", "Subtitle"),
                self.trail_points,
                self.trail_note,
                label(
                    "導線は YOLO の映像（detected_image）に描かれ、Track モードのときだけ出ます。"
                    "足首を選ぶと床の上の動線になります。",
                    "Subtitle",
                ),
            )
        )
        layout.addStretch()

    def _tracker_changed(self):
        tracker = self.trackers[self.tracker_combo.currentIndex()]
        reid_model = self.reid_models[self.reid_combo.currentIndex()] if self.reid_models else ""
        settings = self.system.settings
        new = (tracker, self.reid_check.isChecked(), reid_model)
        old = (settings.tracker, settings.tracker_reid, settings.reid_model)
        if new != old:
            self.system.set_tracker(*new)

    def update_status(self, status):
        settings = self.system.settings
        self.reid_models = self.system.reid_models()
        if not self.reid_combo.view().isVisible():
            set_combo_items(
                self.reid_combo,
                self.reid_models,
                self.reid_models.index(settings.reid_model)
                if settings.reid_model in self.reid_models
                else 0,
            )
        ready = status.yolo_state in ("active", "inactive") and not status.busy
        tracker = self.trackers[self.tracker_combo.currentIndex()]
        supports_reid = tracker in REID_TRACKERS
        self.tracker_combo.setEnabled(ready)
        self.reid_check.setEnabled(ready and supports_reid)
        self.reid_combo.setEnabled(ready and supports_reid and self.reid_check.isChecked())
        if not supports_reid:
            self.tracker_note.setText(f"{TRACKERS[tracker]} は ReID を使いません。")
        elif self.reid_check.isChecked() and not self.reid_models:
            self.tracker_note.setText(
                "weights に ReID モデル（*-reid.onnx など）がないので、Ultralytics の既定を使います。"
            )
        else:
            self.tracker_note.setText("")

        model = status.yolo_model or settings.yolo_model
        pose = is_pose_model(model)
        for index, mode in enumerate(self.trail_modes):
            self.trail_combo.model().item(index).setEnabled(pose or mode in ("false", "bbox"))
        trail_mode = self.trail_modes[self.trail_combo.currentIndex()]
        self.trail_points.setEnabled(pose and trail_mode in ("keypoint", "all"))
        if pose:
            self.trail_note.setText("")
        elif trail_mode in ("keypoint", "all"):
            effect = "枠の中心だけを描きます" if trail_mode == "all" else "導線は描きません"
            self.trail_note.setText(
                f"今のモデル（{model or '?'}）にはキーポイントがないので、{effect}。"
            )
        else:
            self.trail_note.setText("キーポイントの導線は -pose のモデルのときだけ選べます。")
