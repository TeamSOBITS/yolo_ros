from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QPushButton, QWidget
from yolo_ros.gui.catalog import RELIABILITIES
from yolo_ros.gui.qt_utils import card, label
from yolo_ros.gui.system import camera_modes
from yolo_ros.gui.widgets import compact_combo, page_layout, set_combo_items


class CameraPage(QWidget):
    """Which camera to use, its resolution / frame rate, the processing cap and image QoS."""

    def __init__(self, node):
        super().__init__()
        self.system = node.system
        self.sources = []
        self.modes = []
        self.modes_device = None
        settings = self.system.settings
        layout = page_layout(self)

        layout.addWidget(label("使うカメラ", "Section"))
        self.camera_combo = compact_combo()
        self.camera_combo.activated.connect(self._camera_chosen)
        refresh = QPushButton("↻")
        refresh.setToolTip("カメラとトピックを探し直す")
        refresh.setFixedWidth(44)
        refresh.clicked.connect(self.refresh_sources)
        row = QHBoxLayout()
        row.addWidget(self.camera_combo, 1)
        row.addWidget(refresh)
        layout.addWidget(
            card(
                row,
                label(
                    "起動/停止は「起動」タブ。内蔵/USBカメラは yolo_camera で /image_raw に"
                    "出します。ロボットや外部のカメラは、そのトピックを選んでください"
                    "（yolo_node の image_topic_name になります）。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("内蔵/USBカメラの解像度とフレームレート", "Section"))
        self.mode_combo = compact_combo()
        self.mode_combo.activated.connect(lambda _: self._mode_changed())
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setDecimals(0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.editingFinished.connect(self._mode_changed)
        self.device_note = label("", "Subtitle")
        layout.addWidget(
            card(
                label("解像度", "Subtitle"),
                self.mode_combo,
                label("FPS", "Subtitle"),
                self.fps_spin,
                self.device_note,
            )
        )

        layout.addWidget(label("処理の上限（max_rate_hz）", "Section"))
        self.process_spin = QDoubleSpinBox()
        self.process_spin.setRange(0, 60)
        self.process_spin.setDecimals(0)
        self.process_spin.setSuffix(" Hz")
        self.process_spin.setSpecialValueText("制限なし")
        self.process_spin.setValue(float(settings.process_hz))
        self.process_spin.editingFinished.connect(
            lambda: self.system.set_process_hz(self.process_spin.value())
        )
        layout.addWidget(
            card(
                self.process_spin,
                label(
                    "YOLO が1秒に処理する回数の上限です。カメラの FPS を変えられないロボットでも"
                    " GPU / CPU の負荷を下げられます。超えた分のフレームは捨てます。0 で制限なし。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("画像の受信方法（image_reliability）", "Section"))
        self.reliabilities = list(RELIABILITIES)
        self.reliability_combo = compact_combo()
        set_combo_items(
            self.reliability_combo,
            list(RELIABILITIES.values()),
            self.reliabilities.index(settings.image_reliability)
            if settings.image_reliability in RELIABILITIES
            else 0,
        )
        self.reliability_combo.activated.connect(
            lambda i: self.system.set_image_reliability(self.reliabilities[i])
        )
        layout.addWidget(
            card(
                self.reliability_combo,
                label(
                    "自動のままで、reliable / best_effort どちらのカメラからも取りこぼさずに"
                    "受け取れます。変えると YOLO を一瞬止めて購読し直します。",
                    "Subtitle",
                ),
            )
        )
        layout.addStretch()
        self.refresh_sources()

    def refresh_sources(self):
        if self.camera_combo.view().isVisible():
            return
        sources = self.system.camera_sources()
        current = self.system.saved_camera()
        if current.key not in {source.key for source in sources}:
            sources.append(current)
        self.sources = sources
        index = next(i for i, source in enumerate(sources) if source.key == current.key)
        set_combo_items(self.camera_combo, [source.label for source in sources], index)
        self._refresh_modes(current)

    def _refresh_modes(self, camera):
        is_device = camera.kind == "device"
        self.mode_combo.setEnabled(is_device)
        self.fps_spin.setEnabled(is_device)
        if not is_device:
            self.device_note.setText(
                "トピックのカメラは配信側の設定で動きます（処理の上限で調整できます）。"
            )
            return
        if self.modes_device != camera.value:
            self.modes_device = camera.value
            self.modes = camera_modes(camera.value)
        settings = self.system.settings
        size = (int(settings.camera_width), int(settings.camera_height))
        sizes = [(w, h) for w, h, _ in self.modes]
        index = sizes.index(size) if size in sizes else 0
        set_combo_items(
            self.mode_combo, [f"{w} x {h}（最大 {fps:g} fps）" for w, h, fps in self.modes], index
        )
        max_fps = self.modes[index][2] if self.modes else 30.0
        self.fps_spin.setRange(1, max_fps)
        if not self.fps_spin.hasFocus():
            self.fps_spin.setValue(min(float(settings.camera_fps), max_fps))
        self.device_note.setText("変更すると、起動中のカメラはその場で再起動します。")

    def _camera_chosen(self, index):
        source = self.sources[index]
        if source.key != self.system.saved_camera().key:
            self.system.select_camera(source)
        self._refresh_modes(source)

    def _mode_changed(self):
        if not self.modes:
            return
        width, height, max_fps = self.modes[self.mode_combo.currentIndex()]
        fps = min(self.fps_spin.value(), max_fps)
        settings = self.system.settings
        if (width, height, fps) != (
            settings.camera_width,
            settings.camera_height,
            settings.camera_fps,
        ):
            self.system.set_camera_mode(width, height, fps)
        self._refresh_modes(self.system.saved_camera())
