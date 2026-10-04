"""Qt helpers of the yolo_ros GUI (same look and behavior as the osnet_ros GUI)."""

import os
import signal

from PySide6.QtCore import QEvent, QLibraryInfo, QObject, QTimer
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from yolo_ros.gui.style import STYLE_SHEET


def configure_qt_plugin_path():
    """Point Qt back to PySide6's plugins; importing cv2 redirects them to its bundled Qt5 ones."""
    plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if plugin_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
        os.environ["QT_PLUGIN_PATH"] = plugin_path


class WheelGuard(QObject):
    """
    Stop the mouse wheel from changing combo boxes, spin boxes and sliders it passes over.

    The wheel goes to the enclosing scroll area instead, so the page keeps scrolling.
    """

    # Not QAbstractSlider: that would include QScrollBar, which should still scroll.
    INPUTS = (QComboBox, QAbstractSpinBox, QSlider)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and isinstance(watched, self.INPUTS):
            parent = watched.parentWidget()
            while parent is not None and not isinstance(parent, QAbstractScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent.viewport(), event)
            return True
        return False


def apply_style(app):
    app.setStyleSheet(STYLE_SHEET)
    app.installEventFilter(WheelGuard(app))


def quit_on_signals(app):
    """
    Close the Qt app on Ctrl+C / SIGTERM.

    Python signal handlers only run when the interpreter gets control, which never happens inside
    app.exec() without the periodic no-op timer.
    """
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: app.quit())
    timer = QTimer(app)
    timer.timeout.connect(lambda: None)
    timer.start(200)
    return timer


def label(text, object_name=""):
    widget = QLabel(text)
    widget.setWordWrap(True)
    if object_name:
        widget.setObjectName(object_name)
    return widget


def card(*items):
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        else:
            layout.addLayout(item)
    return frame


def restyle(widget, object_name):
    """Change objectName and re-apply the style sheet rules for it."""
    widget.setObjectName(object_name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class StatusRow(QHBoxLayout):
    """Colored-dot status row: green ok, yellow warning, gray when ok is None (off)."""

    def __init__(self, name):
        super().__init__()
        self.mark = QLabel("●")
        self.mark.setFixedWidth(18)
        self.name = QLabel(name)
        self.name.setFixedWidth(96)
        self.detail = label("", "Subtitle")
        self.addWidget(self.mark)
        self.addWidget(self.name)
        self.addWidget(self.detail, 1)

    def set_state(self, ok, detail):
        object_name = "RowOk" if ok else "RowOff" if ok is None else "RowWarn"
        restyle(self.mark, object_name)
        restyle(self.name, object_name)
        self.detail.setText(detail)
