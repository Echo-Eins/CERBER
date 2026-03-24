"""
Launch script for the CERBER Training Monitor application.
"""
import sys
import argparse
from pathlib import Path

# Fix Windows high DPI scaling
import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QPalette, QColor

from tools.cerber_monitor.theme import DARK_STYLESHEET
from tools.cerber_monitor.main_window import MainWindow

def main():
    parser = argparse.ArgumentParser(description="CERBER Monitor GUI")
    parser.add_argument("--metrics", type=str, default=None, help="Auto-load metrics file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Auto-load checkpoint file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("CERBER Monitor")
    
    # Apply theme
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Configure global dark palette for fallback (menus, dialogs)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0d0d1a"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e0e0e8"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0a0a18"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#101028"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1a1a3e"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e0e0e8"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e0e0e8"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#141432"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#c0c0e0"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#00e5ff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1a3a5a"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MainWindow()
    
    # Auto-load files if provided
    if args.metrics and Path(args.metrics).exists():
        window.load_metrics_file(args.metrics)
        window.tabs.setCurrentIndex(0)  # Curves tab
        
    if args.checkpoint and Path(args.checkpoint).exists():
        window.load_checkpoint_file(args.checkpoint)
        if not args.metrics:
            window.tabs.setCurrentIndex(1)  # Inspector tab

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
