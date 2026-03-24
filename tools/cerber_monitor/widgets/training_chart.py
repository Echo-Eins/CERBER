"""
Training Chart Widget - plots loss and evaluation metrics.
"""
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
from PyQt6.QtCore import Qt

class TrainingChartWidget(QWidget):
    def __init__(self, title="Training Metrics", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        self.header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setObjectName("subtitle")
        self.header.addWidget(self.title_label)
        self.header.addStretch()
        
        self.log_loss_cb = QCheckBox("Log Scale (Y)")
        self.log_loss_cb.setChecked(False)
        self.log_loss_cb.toggled.connect(self.toggle_log_scale)
        self.header.addWidget(self.log_loss_cb)
        
        self.layout.addLayout(self.header)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0d0d1a')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        
        # Custom axis styling
        styles = {'color': '#8888aa', 'font-size': '11px'}
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#2a2a4a', width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#8888aa'))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#2a2a4a', width=1))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#8888aa'))
        
        self.plot_widget.addLegend(offset=(10, 10), brush=pg.mkBrush(10, 10, 24, 200))
        self.layout.addWidget(self.plot_widget)
        
        self.curves = {}
        self.scatter = {}
        
    def toggle_log_scale(self, checked):
        self.plot_widget.setLogMode(y=checked)
        
    def add_line(self, name, x, y, color):
        if name in self.curves:
            self.curves[name].setData(x, y)
        else:
            pen = pg.mkPen(color=color, width=2)
            self.curves[name] = self.plot_widget.plot(x, y, name=name, pen=pen)
            
    def add_scatter(self, name, x, y, color, symbol='o'):
        if name in self.scatter:
            self.scatter[name].setData(x, y)
        else:
            pen = pg.mkPen(color=color, width=1)
            brush = pg.mkBrush(color=color)
            self.scatter[name] = pg.ScatterPlotItem(x=x, y=y, size=7, pen=pen, brush=brush, name=name, symbol=symbol)
            self.plot_widget.addItem(self.scatter[name])
            
    def clear(self):
        self.plot_widget.clear()
        self.curves = {}
        self.scatter = {}
