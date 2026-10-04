"""Preview area split into 1 or 2 panes; each pane shows the camera or YOLO's drawing."""

import datetime
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QVBoxLayout, QWidget
from yolo_ros.gui.qt_utils import label
from yolo_ros.gui.widgets import compact_combo, set_combo_items

SOURCES = {
    "yolo": "YOLO（検出・導線）",
    "camera": "カメラ映像",
}
EMPTY_TEXT = {
    "yolo": "YOLO の映像がありません（「起動」タブでカメラと YOLO を起動してください）",
    "camera": "カメラは OFF です（「起動」タブの「起動」で開始します）",
}
LAYOUTS = (1, 2)


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def to_qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


def qimage_to_bgr(image):
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    rgb = buffer.reshape(height, image.bytesPerLine())[:, : width * 3].reshape(height, width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class PreviewPane(QWidget):
    source_changed = Signal()

    def __init__(self, source):
        super().__init__()
        self.sources = list(SOURCES)
        self.last_frame = None
        self.source_combo = compact_combo()
        set_combo_items(
            self.source_combo,
            list(SOURCES.values()),
            self.sources.index(source) if source in SOURCES else 0,
        )
        self.source_combo.activated.connect(lambda _: self.source_changed.emit())
        self.image = label("", "Preview")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(240, 160)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.image, 1)

    @property
    def source(self):
        return self.sources[self.source_combo.currentIndex()]

    def show_frame(self, frame):
        self.last_frame = frame
        if frame is None:
            self.image.setPixmap(QPixmap())
            self.image.setText(EMPTY_TEXT[self.source])
            return
        self.image.setPixmap(
            QPixmap.fromImage(to_qimage(frame)).scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class PreviewGrid(QWidget):
    sources_changed = Signal()

    def __init__(self, layout_count, sources):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        sources = (list(sources) + ["yolo", "camera"])[: max(LAYOUTS)]
        self.panes = [PreviewPane(source) for source in sources]
        for pane in self.panes:
            pane.source_changed.connect(self.sources_changed.emit)
        self.layout_count = 1
        self.set_layout_count(layout_count)

    def visible_panes(self):
        return self.panes[: self.layout_count]

    def set_layout_count(self, count):
        self.layout_count = count if count in LAYOUTS else 1
        for pane in self.panes:
            self.grid.removeWidget(pane)
            pane.hide()
        for column, pane in enumerate(self.visible_panes()):
            self.grid.addWidget(pane, 0, column)
            pane.show()
        self.sources_changed.emit()

    def sources(self):
        return [pane.source for pane in self.panes]

    def visible_sources(self):
        return {pane.source for pane in self.visible_panes()}

    def refresh(self, frame_for):
        """frame_for(source) returns the latest BGR frame for that source, or None."""
        for pane in self.visible_panes():
            pane.show_frame(frame_for(pane.source))

    def save_preview(self, folder):
        """Save the preview as displayed (all panes with their selectors) as one PNG."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{timestamp()}_preview.png")
        cv2.imwrite(path, qimage_to_bgr(self.grab().toImage()))
        return [path]

    def save_panes(self, folder):
        """Save each visible pane's latest frame at full resolution; returns the paths."""
        os.makedirs(folder, exist_ok=True)
        stamp = timestamp()
        saved = []
        for index, pane in enumerate(self.visible_panes(), start=1):
            if pane.last_frame is None:
                continue
            path = os.path.join(folder, f"{stamp}_pane{index}_{pane.source}.png")
            cv2.imwrite(path, pane.last_frame)
            saved.append(path)
        return saved
