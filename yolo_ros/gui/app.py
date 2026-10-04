"""yolo_gui: start/stop and configure yolo_node (and the camera / 3D nodes) from one window."""

import os
import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
import rclpy
from yolo_ros.gui.gui_node import YoloGuiNode
from yolo_ros.gui.pages.camera_page import CameraPage
from yolo_ros.gui.pages.detect_page import DetectPage
from yolo_ros.gui.pages.keypoint_page import KeypointPage
from yolo_ros.gui.pages.position_page import PositionPage
from yolo_ros.gui.pages.run_page import RunPage
from yolo_ros.gui.pages.tracking_page import TrackingPage
from yolo_ros.gui.preview import LAYOUTS, PreviewGrid
from yolo_ros.gui.qt_utils import apply_style, configure_qt_plugin_path, label, quit_on_signals
from yolo_ros.gui.widgets import scroll_page, segment, segment_group

LOG_MARKS = {"info": "•", "warn": "⚠", "error": "✖"}


class YoloWindow(QMainWindow):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.system = node.system
        self.screenshots_dir = os.path.join(node.data_root, "screenshots")
        self.setWindowTitle("YOLO ROS")
        self.resize(1400, 880)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        main = QWidget()
        main.setMinimumWidth(480)
        main.setLayout(self._build_main_area())
        splitter.addWidget(main)
        splitter.addWidget(self._build_sidebar())
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([960, 440])
        self.setCentralWidget(splitter)

        for interval_ms, callback in (
            (66, self.refresh_preview),
            (500, self.refresh_status),
            (3000, self.camera_page.refresh_sources),
        ):
            timer = QTimer(self)
            timer.timeout.connect(callback)
            timer.start(interval_ms)
        self._sources_changed()
        self.refresh_status()

    def _build_main_area(self):
        area = QVBoxLayout()
        area.setContentsMargins(24, 20, 16, 16)
        area.setSpacing(10)
        settings = self.system.settings

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("YOLO ROS", "Title"))
        titles.addWidget(label("物体検出・姿勢・追跡の設定と確認", "Subtitle"))
        header.addLayout(titles, 1)
        self.layout_buttons = []
        for count in LAYOUTS:
            button = segment(f"{count}画面")
            button.setChecked(count == settings.view_layout)
            button.clicked.connect(lambda _, c=count: self._set_layout(c))
            self.layout_buttons.append(button)
        header.addLayout(segment_group(self, *self.layout_buttons))
        screenshot_button = QPushButton("スクショ ▾")
        menu = QMenu(screenshot_button)
        menu.addAction("プレビュー全体（今の見た目のまま 1 枚）", lambda: self.save_screenshots("preview"))
        menu.addAction("画面それぞれの映像（元の解像度で）", lambda: self.save_screenshots("panes"))
        screenshot_button.setMenu(menu)
        screenshot_button.setToolTip(f"PNG で {self.screenshots_dir} に保存します")
        header.addSpacing(16)
        header.addWidget(screenshot_button)
        area.addLayout(header)

        self.preview = PreviewGrid(settings.view_layout, settings.view_sources)
        self.preview.sources_changed.connect(self._sources_changed)
        area.addWidget(self.preview, 1)
        return area

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(300)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(12)

        self.run_page = RunPage(self.node)
        self.camera_page = CameraPage(self.node)
        self.detect_page = DetectPage(self.node)
        self.tracking_page = TrackingPage(self.node)
        self.keypoint_page = KeypointPage(self.node)
        self.position_page = PositionPage(self.node)
        self.status_pages = (
            self.run_page,
            self.detect_page,
            self.tracking_page,
            self.keypoint_page,
            self.position_page,
        )

        self.pages = QStackedWidget()
        buttons = []
        for title, page in (
            ("起動", self.run_page),
            ("カメラ", self.camera_page),
            ("検出", self.detect_page),
            ("追跡", self.tracking_page),
            ("姿勢", self.keypoint_page),
            ("3D", self.position_page),
        ):
            scroll = scroll_page(page)
            self.pages.addWidget(scroll)
            button = segment(title)
            button.clicked.connect(lambda _, widget=scroll: self.pages.setCurrentWidget(widget))
            buttons.append(button)
        buttons[0].setChecked(True)
        layout.addLayout(segment_group(self, *buttons))
        layout.addWidget(self.pages, 1)
        return sidebar

    def _set_layout(self, count):
        self.preview.set_layout_count(count)
        self.system.save_settings(view_layout=count)

    def _sources_changed(self):
        self.node.show_yolo_image("yolo" in self.preview.visible_sources())
        self.system.save_settings(view_sources=self.preview.sources())

    def save_screenshots(self, kind):
        if kind == "preview":
            saved = self.preview.save_preview(self.screenshots_dir)
        else:
            saved = self.preview.save_panes(self.screenshots_dir)
        if saved:
            self.node.log("info", f"スクショを {len(saved)} 枚保存しました: {self.screenshots_dir}")
        else:
            self.node.log("warn", "保存できる映像がありません（カメラと YOLO を起動してください）。")

    def refresh_preview(self):
        self.preview.refresh(self.node.frame)

    def refresh_status(self):
        status = self.system.status()
        for page in self.status_pages:
            page.update_status(status)
        for level, message in self.node.pop_messages():
            self.run_page.append_log(LOG_MARKS.get(level, "•"), message)
            self.statusBar().showMessage(message, 8000)


def main(args=None):
    rclpy.init(args=args)
    node = YoloGuiNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    configure_qt_plugin_path()
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    signal_timer = quit_on_signals(app)  # noqa: F841 keep the timer alive while the app runs
    window = YoloWindow(node)
    window.show()
    try:
        app.exec()
    finally:
        node.system.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
