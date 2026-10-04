from PySide6.QtWidgets import QWidget
from yolo_ros.gui.catalog import is_pose_model
from yolo_ros.gui.controller import POSITION_NODES
from yolo_ros.gui.qt_utils import card, label, StatusRow
from yolo_ros.gui.widgets import page_layout, power_button, set_power

DESCRIPTIONS = {
    "bbox": ("枠 → 3D", "枠の中の点群から、物体の 3D 位置を出します。"),
    "keypoint": ("姿勢 → 3D", "キーポイントごとの 3D 位置を出します（-pose のモデル）。"),
    "mask": ("輪郭 → 3D", "輪郭の中の点群から 3D 位置を出します（-seg のモデル）。"),
}
OUTPUTS = {
    "bbox": "bbox_to_3d/object_3d_poses",
    "keypoint": "keypoint_3d_array",
    "mask": "mask_to_3d/object_3d_poses",
}


class PositionPage(QWidget):
    """image_to_position's bbox / keypoint / mask_to_3d nodes, connected to yolo_node."""

    def __init__(self, node):
        super().__init__()
        self.system = node.system
        layout = page_layout(self)
        layout.addWidget(label("3D 位置への変換（image_to_position）", "Section"))
        self.rows = {}
        self.buttons = {}
        items = []
        for kind in POSITION_NODES:
            title, text = DESCRIPTIONS[kind]
            row = StatusRow(title)
            button = power_button(lambda _=False, k=kind: self._toggle(k))
            row.addWidget(button)
            self.rows[kind] = row
            self.buttons[kind] = button
            items += [
                row,
                label(
                    f"{text}\n入力: {self.system.position_input_topic(kind)} → "
                    f"出力: {OUTPUTS[kind]}",
                    "Subtitle",
                ),
            ]
        self.model_note = label("", "Subtitle")
        layout.addWidget(card(*items, self.model_note))
        layout.addWidget(
            card(
                label(
                    "launch で use_bbox_to_3d:=true などとして起動したノードはその状態を表示し、"
                    "起動していないものは「起動」で image_to_position の launch を始めます。"
                    "点群・カメラ情報のトピックや座標系は image_to_position/config/*.yaml の"
                    "設定を使います（ロボットに合わせて変更してください）。",
                    "Subtitle",
                )
            )
        )
        layout.addStretch()

    def _toggle(self, kind):
        state = self.system.positions[kind].state
        if state == "active":
            self.system.stop_position(kind)
        else:
            self.system.start_position(kind)

    def update_status(self, status):
        for kind, row in self.rows.items():
            state = status.position_states.get(kind)
            process = self.system.position_processes[kind]
            if state == "active":
                row.set_state(True, "動作中")
            elif state in ("inactive", "unconfigured"):
                row.set_state(None, "OFF（ノードは起動済み）")
            elif process.running():
                row.set_state(False, "起動中…")
            elif process.exited_with_error():
                row.set_state(False, "起動に失敗しました（data_root/logs のログを確認）")
            else:
                row.set_state(None, "OFF")
            # While a launch started here is still coming up, a second press would start another.
            starting = process.running() and state is None
            set_power(self.buttons[kind], state == "active", enabled=not starting)
        model = status.yolo_model or self.system.settings.yolo_model or ""
        notes = []
        if not is_pose_model(model):
            notes.append("姿勢 → 3D には -pose のモデルが必要です")
        if "-seg" not in model:
            notes.append("輪郭 → 3D には -seg のモデルが必要です")
        self.model_note.setText(f"今のモデル {model or '?'}: " + "、".join(notes) if notes else "")
