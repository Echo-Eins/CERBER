"""
CERBER Monitor — Dark theme with neon accents.

Consistent with the existing matplotlib energy landscape plots:
- Dark background (#0a0a0a)
- Cyan, magenta, lime accents
- Premium feel with subtle glows and rounded corners
"""

DARK_STYLESHEET = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: #0d0d1a;
    color: #e0e0e8;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

/* ── Tab Widget ── */
QTabWidget::pane {
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    background: #0d0d1a;
    padding: 4px;
}

QTabBar::tab {
    background: #151530;
    border: 1px solid #2a2a4a;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 10px 22px;
    margin-right: 3px;
    color: #8888aa;
    font-weight: 500;
    min-width: 120px;
}

QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a1a3e, stop:1 #0d0d1a);
    color: #00e5ff;
    border-bottom: 2px solid #00e5ff;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: #1a1a35;
    color: #bbbbdd;
}

/* ── Group Boxes ── */
QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    font-weight: 600;
    color: #00e5ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: #00e5ff;
}

/* ── Buttons ── */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1e1e45, stop:1 #141432);
    border: 1px solid #3a3a6a;
    border-radius: 6px;
    padding: 8px 18px;
    color: #c0c0e0;
    font-weight: 500;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a5a, stop:1 #1a1a40);
    border-color: #00e5ff;
    color: #ffffff;
}

QPushButton:pressed {
    background: #0a0a20;
    border-color: #ff00ff;
}

QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00bcd4, stop:1 #0097a7);
    border: none;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00e5ff, stop:1 #00bcd4);
}

/* ── Tables ── */
QTableWidget, QTableView {
    background: #0a0a18;
    alternate-background-color: #101028;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    gridline-color: #1a1a3a;
    selection-background-color: #1a3a5a;
    color: #d0d0e0;
}

QHeaderView::section {
    background: #151530;
    border: 1px solid #2a2a4a;
    padding: 6px 10px;
    color: #00e5ff;
    font-weight: 600;
}

/* ── Scroll Bars ── */
QScrollBar:vertical {
    background: #0a0a18;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #3a3a6a;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #00e5ff;
}

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
}

QScrollBar:horizontal {
    background: #0a0a18;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #3a3a6a;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #00e5ff;
}

/* ── Labels ── */
QLabel {
    color: #c0c0e0;
}

QLabel#title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
}

QLabel#subtitle {
    font-size: 14px;
    color: #8888aa;
}

QLabel#metric_value {
    font-size: 28px;
    font-weight: 800;
    color: #00ff88;
}

QLabel#metric_label {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QLabel#delta_positive {
    color: #00ff88;
    font-weight: 600;
}

QLabel#delta_negative {
    color: #ff4466;
    font-weight: 600;
}

/* ── Line Edits, Spinboxes ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0a0a18;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    padding: 6px 10px;
    color: #d0d0e0;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #00e5ff;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background: #0a0a18;
    border: 1px solid #2a2a4a;
    selection-background-color: #1a3a5a;
    color: #d0d0e0;
}

/* ── Splitter ── */
QSplitter::handle {
    background: #2a2a4a;
}

QSplitter::handle:hover {
    background: #00e5ff;
}

/* ── Progress Bar ── */
QProgressBar {
    background: #0a0a18;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e8;
    height: 16px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00bcd4, stop:0.5 #00e5ff, stop:1 #00bcd4);
    border-radius: 3px;
}

/* ── Status Bar ── */
QStatusBar {
    background: #0a0a12;
    border-top: 1px solid #2a2a4a;
    color: #8888aa;
    font-size: 12px;
}

/* ── Tool Tips ── */
QToolTip {
    background: #1a1a3e;
    border: 1px solid #00e5ff;
    border-radius: 4px;
    color: #e0e0e8;
    padding: 6px;
}

/* ── File Dialog ── */
QFileDialog {
    background: #0d0d1a;
}

/* ── Checkboxes ── */
QCheckBox {
    spacing: 8px;
    color: #c0c0e0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #3a3a6a;
    border-radius: 4px;
    background: #0a0a18;
}

QCheckBox::indicator:checked {
    background: #00e5ff;
    border-color: #00e5ff;
}

/* ── Slider ── */
QSlider::groove:horizontal {
    height: 4px;
    background: #2a2a4a;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #00e5ff;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #00ffff;
}

/* ── Text Browser ── */
QTextBrowser, QTextEdit {
    background: #0a0a18;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    color: #d0d0e0;
    padding: 8px;
}
"""

# Color constants for plots
COLORS = {
    "cyan": "#00e5ff",
    "magenta": "#ff00ff",
    "lime": "#00ff88",
    "yellow": "#ffff00",
    "orange": "#ff8800",
    "red": "#ff4466",
    "blue": "#4488ff",
    "purple": "#aa44ff",
    "white": "#ffffff",
    "gray": "#888888",
    "bg_dark": "#0a0a0a",
    "bg_panel": "#0d0d1a",
    "bg_input": "#0a0a18",
    "border": "#2a2a4a",
    "accent": "#00e5ff",
}

# Series colors for multi-line charts
SERIES_COLORS = [
    "#00e5ff",  # cyan
    "#ff00ff",  # magenta
    "#00ff88",  # lime
    "#ffff00",  # yellow
    "#ff8800",  # orange
    "#4488ff",  # blue
    "#aa44ff",  # purple
    "#ff4466",  # red
]
