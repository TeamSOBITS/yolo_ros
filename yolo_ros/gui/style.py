"""Discord-like dark theme, the same as the osnet_ros GUI so both look alike."""

BLURPLE = "#5865f2"
GREEN = "#23a55a"
YELLOW = "#f0b232"
RED = "#f23f43"

STYLE_SHEET = """
QMainWindow, QWidget, QDialog {
    background: #313338;
    color: #dbdee1;
    font-size: 14px;
}
QToolTip { background: #111214; color: #dbdee1; border: none; padding: 6px; }

QLabel#Title    { font-size: 22px; font-weight: bold; color: #f2f3f5; }
QLabel#Subtitle { font-size: 13px; color: #949ba4; }
QLabel#Section  { font-size: 12px; font-weight: bold; color: #949ba4; padding-top: 6px; }
QLabel#RowOk    { color: #23a55a; font-weight: bold; }
QLabel#RowWarn  { color: #f0b232; font-weight: bold; }
QLabel#RowOff   { color: #80848e; font-weight: bold; }
QLabel#Path     { font-family: monospace; font-size: 12px; color: #b5bac1; }

QWidget#Sidebar { background: #2b2d31; border-left: 1px solid #1e1f22; }
QFrame#Card {
    background: #2b2d31;
    border: 1px solid #1e1f22;
    border-radius: 10px;
}
QWidget#Sidebar QFrame#Card { background: #232428; }
QFrame#Card QLabel, QFrame#Card QRadioButton, QFrame#Card QCheckBox { background: transparent; }

QLabel#Preview {
    background: #1e1f22;
    border-radius: 12px;
    color: #80848e;
    font-size: 16px;
}

QPushButton {
    background: #4e5058;
    border: none;
    border-radius: 4px;
    padding: 9px 18px;
    color: #ffffff;
    font-weight: 500;
}
QPushButton:hover    { background: #6d6f78; }
QPushButton:disabled { background: #3a3c41; color: #80848e; }
QPushButton#Primary { background: #5865f2; font-weight: bold; }
QPushButton#Primary:hover    { background: #4752c4; }
QPushButton#Primary:disabled { background: #3c4270; color: #a0a4b8; }
QPushButton#Danger { background: #da373c; }
QPushButton#Danger:hover { background: #a12d2f; }

QPushButton#Segment {
    background: #1e1f22;
    color: #b5bac1;
    border-radius: 4px;
    padding: 7px 10px;
}
QPushButton#Segment:hover    { background: #35373c; color: #dbdee1; }
QPushButton#Segment:checked  { background: #5865f2; color: #ffffff; font-weight: bold; }
QPushButton#Segment:disabled { background: #2b2d31; color: #5c5e66; }

QLineEdit, QComboBox {
    background: #1e1f22;
    border: 1px solid #1e1f22;
    border-radius: 4px;
    padding: 6px 8px;
    color: #dbdee1;
    selection-background-color: #5865f2;
}
QLineEdit:focus, QComboBox:hover { border-color: #5865f2; }
QLineEdit:disabled, QComboBox:disabled { color: #5c5e66; }
QComboBox QAbstractItemView {
    background: #2b2d31;
    color: #dbdee1;
    selection-background-color: #5865f2;
    border: 1px solid #1e1f22;
    outline: none;
}

QRadioButton { padding: 4px; spacing: 10px; color: #dbdee1; }
QRadioButton::indicator {
    width: 16px; height: 16px; border: 1px solid #6d6f78; border-radius: 8px; background: #1e1f22;
}
QRadioButton::indicator:checked { background: #5865f2; border-color: #5865f2; }
QRadioButton:disabled { color: #5c5e66; }
QCheckBox { padding: 3px; spacing: 10px; color: #dbdee1; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #6d6f78; border-radius: 4px; background: #1e1f22;
}
QCheckBox::indicator:checked { background: #5865f2; border-color: #5865f2; }
QCheckBox:disabled { color: #5c5e66; }
QCheckBox::indicator:disabled { border-color: #3f4147; }

QListWidget {
    background: #2b2d31;
    border: 1px solid #1e1f22;
    border-radius: 8px;
    color: #dbdee1;
    outline: none;
}
QListWidget::item { padding: 6px 8px; border-radius: 4px; }
QListWidget::item:hover    { background: #35373c; }
QListWidget::item:selected { background: #404249; color: #ffffff; }

QPlainTextEdit {
    background: #1e1f22;
    border: 1px solid #1e1f22;
    border-radius: 8px;
    font-family: monospace;
    font-size: 12px;
    color: #b5bac1;
}

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: #1a1b1e; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }

QToolButton#More {
    background: transparent;
    color: #b5bac1;
    border: none;
    border-radius: 4px;
    padding: 0px 8px 4px 8px;
    font-size: 18px;
    font-weight: bold;
}
QToolButton#More:hover { background: #404249; color: #ffffff; }
QToolButton#More::menu-indicator { image: none; width: 0px; }
QPushButton::menu-indicator { image: none; width: 0px; }
QMenu {
    background: #111214;
    color: #dbdee1;
    border: 1px solid #1e1f22;
    border-radius: 6px;
    padding: 6px;
}
QMenu::item { padding: 7px 22px 7px 12px; border-radius: 4px; }
QMenu::item:selected { background: #5865f2; color: #ffffff; }
QMenu::separator { height: 1px; background: #2b2d31; margin: 4px 6px; }

QWidget#PersonRow, QWidget#PersonRow QLabel { background: transparent; }
QLabel#PersonName { font-size: 15px; font-weight: bold; color: #f2f3f5; }
QListWidget#Thumbnails { background: #1e1f22; }
QListWidget#Thumbnails::item { color: #949ba4; }

QLabel#Metric      { font-size: 20px; font-weight: bold; color: #f2f3f5; }
QLabel#MetricName  { font-size: 12px; color: #949ba4; }

QStatusBar { background: #232428; color: #949ba4; }
QMessageBox QLabel { color: #dbdee1; }
"""
