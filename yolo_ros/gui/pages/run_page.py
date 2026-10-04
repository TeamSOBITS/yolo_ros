from PySide6.QtWidgets import QGridLayout, QPlainTextEdit, QWidget
from yolo_ros.gui.qt_utils import card, label, StatusRow
from yolo_ros.gui.widgets import page_layout, power_button, set_power

# stats key -> (name shown, unit, explanation in the tooltip)
METRICS = (
    ("processed_fps", "処理 FPS", "", "YOLO が1秒に処理したフレーム数"),
    ("received_fps", "受信 FPS", "", "カメラから1秒に届いたフレーム数（処理の上限で捨てた分も含む）"),
    ("inference_ms", "推論", " ms", "前処理・推論・後処理（Ultralytics の計測）"),
    ("callback_ms", "1フレーム", " ms", "推論に描画と配信を足した、1フレームの処理時間"),
    ("latency_ms", "遅延", " ms", "画像の撮影時刻（header.stamp）から結果を配信するまで"),
    ("detections", "検出数", "", "直近のフレームで配信した枠の数"),
)


def hz_text(hz):
    return f"{hz:.1f} Hz"


class RunPage(QWidget):
    """Start/stop camera and YOLO, processing figures, detected ids and the log."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.system = node.system
        layout = page_layout(self)

        layout.addWidget(label("接続状態", "Section"))
        self.camera_row = StatusRow("カメラ")
        self.yolo_row = StatusRow("YOLO")
        self.camera_button = power_button(self._toggle_camera)
        self.yolo_button = power_button(self._toggle_yolo)
        self.camera_row.addWidget(self.camera_button)
        self.yolo_row.addWidget(self.yolo_button)
        layout.addWidget(
            card(
                self.camera_row,
                self.yolo_row,
                label(
                    "YOLO の「停止」は推論だけを止めます（モデルは読み込んだまま）。"
                    "モデルやトラッカーの切り替えは各タブから行えます。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("処理状況（yolo_node/stats）", "Section"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        self.metric_values = {}
        for index, (key, name, _, tooltip) in enumerate(METRICS):
            value = label("-", "Metric")
            title = label(name, "MetricName")
            value.setToolTip(tooltip)
            title.setToolTip(tooltip)
            row, column = divmod(index, 3)
            grid.addWidget(value, row * 2, column)
            grid.addWidget(title, row * 2 + 1, column)
            self.metric_values[key] = value
        self.detections_label = label("", "Subtitle")
        layout.addWidget(
            card(
                grid,
                self.detections_label,
                label(
                    "処理 FPS が受信 FPS より低いときは、推論が追いついていないか処理の上限で"
                    "捨てています。遅延はカメラと同じ時計のときだけ出ます。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("ログ", "Section"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setMinimumHeight(160)
        layout.addWidget(self.log_view, 1)

    def _toggle_camera(self):
        self.system.stop_camera() if self.system.camera_on else self.system.start_camera()

    def _toggle_yolo(self):
        active = self.system.yolo.state == "active"
        self.system.stop_yolo() if active else self.system.start_yolo()

    def append_log(self, mark, message):
        self.log_view.appendPlainText(f"{mark} {message}")

    def update_status(self, status):
        rates = self.node.rates()
        if not status.camera_on:
            self.camera_row.set_state(None, f"OFF（{status.camera.label}）")
        elif status.camera_error:
            self.camera_row.set_state(False, status.camera_error)
        elif self.node.camera_alive():
            self.camera_row.set_state(True, f"{status.camera.label}  {hz_text(rates['camera'])}")
        else:
            self.camera_row.set_state(False, f"{status.camera.label} を待っています")
        set_power(self.camera_button, status.camera_on, enabled=True)

        settings = f"{status.yolo_model or '?'}（{status.yolo_mode or '?'}）"
        if status.busy:
            self.yolo_row.set_state(False, status.busy)
        elif status.yolo_state == "active":
            alive = self.node.detector_alive()
            self.yolo_row.set_state(
                alive, f"{settings}  {hz_text(rates['yolo'])}" if alive else f"{settings} 画像待ち"
            )
        elif status.yolo_state in ("inactive", "unconfigured"):
            self.yolo_row.set_state(None, f"OFF（{settings}）")
        else:
            self.yolo_row.set_state(False, "YOLO のノードがありません")
        set_power(
            self.yolo_button,
            status.yolo_state == "active",
            enabled=status.yolo_state is not None and not status.busy,
        )

        stats = self.node.latest_stats() if status.yolo_state == "active" else {}
        for key, _, unit, _ in METRICS:
            value = stats.get(key)
            self.metric_values[key].setText("-" if value is None else f"{value:g}{unit}")
        detections = self.node.latest_detections()
        if stats:
            extra = f"入力サイズ {stats.get('imgsz', '?')}" + (
                "・半精度（FP16）" if stats.get("half") else ""
            )
            ids = "、".join(detections[:12]) + ("…" if len(detections) > 12 else "")
            self.detections_label.setText(f"{extra}\n{ids}" if ids else extra)
        else:
            self.detections_label.setText("YOLO を起動すると処理状況が出ます。")
