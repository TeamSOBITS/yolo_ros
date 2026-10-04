from PySide6.QtWidgets import QCheckBox, QWidget
from yolo_ros.gui.catalog import is_pose_model, POSE_KEYPOINTS
from yolo_ros.gui.qt_utils import card, label
from yolo_ros.gui.widgets import all_none_buttons, KeypointChecks, page_layout


class KeypointPage(QWidget):
    """Which keypoints are published, and the 'is a person' keypoint filter (pose models)."""

    def __init__(self, node):
        super().__init__()
        self.system = node.system
        settings = self.system.settings
        layout = page_layout(self)

        layout.addWidget(label("配信するキーポイント（object_keypoints）", "Section"))
        self.publish_points = KeypointChecks(POSE_KEYPOINTS, settings.keypoint_publish)
        self.publish_points.changed.connect(
            lambda names: self.system.set_keypoint_settings(keypoint_publish=names)
        )
        self.publish_note = label("", "Subtitle")
        layout.addWidget(
            card(
                all_none_buttons(
                    lambda: self._set_publish(list(POSE_KEYPOINTS)), lambda: self._set_publish([])
                ),
                self.publish_points,
                self.publish_note,
                label(
                    "選んだ点だけを、枠と同じ順番で配信します（keypoint_publish_names）。"
                    "keypoint_to_3d や osnet_ros の距離推定に使う点は残してください。"
                    "見えない点は (0, 0) になります。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("人とみなす条件", "Section"))
        self.filter_check = QCheckBox("選んだ点がすべて見えている人だけを残す")
        self.filter_check.setChecked(bool(settings.keypoint_filter))
        self.filter_check.toggled.connect(
            lambda on: self.system.set_keypoint_settings(keypoint_filter=on)
        )
        self.filter_points = KeypointChecks(POSE_KEYPOINTS, settings.keypoint_filter_names)
        self.filter_points.changed.connect(
            lambda names: self.system.set_keypoint_settings(keypoint_filter_names=names)
        )
        self.filter_note = label("", "Subtitle")
        layout.addWidget(
            card(
                self.filter_check,
                self.filter_points,
                self.filter_note,
                label(
                    "例: 鼻だけ選ぶと、顔がこちらを向いている人だけが枠・キーポイント・追跡の対象に"
                    "なります（person_keypoint_filter_names）。後ろ姿の人も残すときは OFF。",
                    "Subtitle",
                ),
            )
        )
        layout.addStretch()

    def _set_publish(self, names):
        self.publish_points.set_selected(names)
        self.system.set_keypoint_settings(keypoint_publish=names)

    def update_status(self, status):
        model = status.yolo_model or self.system.settings.yolo_model
        pose = is_pose_model(model)
        self.publish_points.setEnabled(pose)
        self.filter_check.setEnabled(pose)
        self.filter_points.setEnabled(pose and self.filter_check.isChecked())
        pose_note = (
            ""
            if pose
            else f"今のモデル（{model or '?'}）にはキーポイントがありません。"
            "yolo26m-pose.pt など名前に -pose が付くモデルにすると使えます。"
        )
        if pose and not self.publish_points.selected():
            self.publish_note.setText("点を選んでいないので、キーポイントは配信しません。")
        else:
            self.publish_note.setText(pose_note)
        if pose and self.filter_check.isChecked() and not self.filter_points.selected():
            self.filter_note.setText("点を1つ以上選んでください（選ばないと絞り込みません）。")
        else:
            self.filter_note.setText(pose_note)
