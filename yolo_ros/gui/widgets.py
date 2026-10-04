"""Reusable widgets of the yolo_ros GUI."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from yolo_ros.gui.qt_utils import restyle


def compact_combo():
    """Combo box that does not widen the sidebar for long topic or model names."""
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    return combo


def set_combo_items(combo, labels, current_index):
    """Replace the items only when they changed (rebuilding closes an open popup)."""
    if [combo.itemText(i) for i in range(combo.count())] != list(labels):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(labels))
        combo.blockSignals(False)
    if 0 <= current_index < combo.count() and combo.currentIndex() != current_index:
        combo.blockSignals(True)
        combo.setCurrentIndex(current_index)
        combo.blockSignals(False)


def segment(text):
    button = QPushButton(text)
    button.setObjectName("Segment")
    button.setCheckable(True)
    return button


def segment_group(parent, *buttons):
    group = QButtonGroup(parent)
    group.setExclusive(True)
    row = QHBoxLayout()
    for button in buttons:
        group.addButton(button)
        row.addWidget(button)
    return row


def page_layout(widget):
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    return layout


def scroll_page(page, minimum_width=340):
    """Wrap a sidebar page so that both scroll bars appear when the window is too small."""
    page.setMinimumWidth(minimum_width)
    scroll = QScrollArea()
    scroll.setWidget(page)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return scroll


def power_button(callback):
    button = QPushButton("起動")
    button.setObjectName("Primary")
    button.setFixedWidth(64)
    button.clicked.connect(callback)
    return button


def set_power(button, on, enabled):
    """Show '停止' (gray) while running and '起動' (blurple) while stopped."""
    text = "停止" if on else "起動"
    if button.text() != text:
        button.setText(text)
        restyle(button, "" if on else "Primary")
    button.setEnabled(enabled)


class CheckGrid(QWidget):
    """Check boxes in columns for a {name: label} list that can be replaced (e.g. new model)."""

    changed = Signal(list)

    def __init__(self, columns=2, items=None, selected=()):
        super().__init__()
        self.columns = columns
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(12)
        self.boxes = {}
        if items:
            self.set_items(items, selected)

    def set_items(self, items, selected):
        """Rebuild only when the names changed; then just update the checks."""
        if list(items) != list(self.boxes):
            for box in self.boxes.values():
                self.grid.removeWidget(box)
                box.deleteLater()
            self.boxes = {}
            for index, (name, text) in enumerate(items.items()):
                box = QCheckBox(text)
                box.toggled.connect(lambda _: self.changed.emit(self.selected()))
                self.boxes[name] = box
                self.grid.addWidget(box, index // self.columns, index % self.columns)
        self.set_selected(selected)

    def selected(self):
        return [name for name, box in self.boxes.items() if box.isChecked()]

    def set_selected(self, names):
        for name, box in self.boxes.items():
            box.blockSignals(True)
            box.setChecked(name in names)
            box.blockSignals(False)


class KeypointChecks(CheckGrid):
    """Two columns of keypoint check boxes: nose alone on the first row, then left | right."""

    def __init__(self, keypoints, selected):
        super().__init__(columns=2)
        for index, (name, text) in enumerate(keypoints.items()):
            box = QCheckBox(text)
            box.toggled.connect(lambda _: self.changed.emit(self.selected()))
            self.boxes[name] = box
            row, column = (0, 0) if index == 0 else ((index + 1) // 2, (index + 1) % 2)
            self.grid.addWidget(box, row, column)
        self.set_selected(selected)


def all_none_buttons(on_all, on_none, all_text="すべて", none_text="なし"):
    row = QHBoxLayout()
    for text, callback in ((all_text, on_all), (none_text, on_none)):
        button = QPushButton(text)
        button.clicked.connect(callback)
        row.addWidget(button)
    return row
