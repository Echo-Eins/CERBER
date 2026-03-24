"""
Checkpoint Comparison Widget - displays diff between multiple checkpoints
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
    QTableWidgetItem, QHeaderView, QPushButton, QLabel,
    QSplitter, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from tools.cerber_monitor.checkpoint_loader import CheckpointInfo

class CheckpointComparisonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Checkpoints list
        self.checkpoints: list[CheckpointInfo] = []
        
        # Header
        self.header = QHBoxLayout()
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_all)
        self.header.addWidget(self.btn_clear)
        self.header.addStretch()
        self.layout.addLayout(self.header)
        
        # Splitter
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.layout.addWidget(self.splitter, 1)
        
        # Top: Metrics comparison table
        self.metrics_group = QWidget()
        m_layout = QVBoxLayout(self.metrics_group)
        m_layout.setContentsMargins(0, 0, 0, 0)
        m_layout.addWidget(QLabel("Evaluation Metrics Comparison", objectName="subtitle"))
        
        self.metrics_table = QTableWidget(0, 1)
        self.metrics_table.setHorizontalHeaderLabels(["Metric"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        m_layout.addWidget(self.metrics_table)
        self.splitter.addWidget(self.metrics_group)
        
        # Bottom: Parameter diff radar / graph
        self.graph_group = QWidget()
        g_layout = QVBoxLayout(self.graph_group)
        g_layout.setContentsMargins(0, 0, 0, 0)
        g_layout.addWidget(QLabel("Denoising Performance (Cosine Diff)", objectName="subtitle"))
        
        self.plot = pg.PlotWidget(background='#0d0d1a')
        self.plot.showGrid(x=False, y=True, alpha=0.2)
        self.plot.addLegend(offset=(10, 10))
        g_layout.addWidget(self.plot)
        self.splitter.addWidget(self.graph_group)
        
        self.splitter.setSizes([300, 300])
        
    def add_checkpoint(self, ckpt: CheckpointInfo):
        self.checkpoints.append(ckpt)
        self.update_view()
        
    def clear_all(self):
        self.checkpoints.clear()
        self.update_view()
        
    def update_view(self):
        self.metrics_table.clearContents()
        self.plot.clear()
        
        if not self.checkpoints:
            self.metrics_table.setColumnCount(1)
            self.metrics_table.setRowCount(0)
            return
            
        # Update Table Headers
        headers = ["Metric"] + [f"Epoch {c.epoch}\n({c.model_type})" for c in self.checkpoints]
        self.metrics_table.setColumnCount(len(headers))
        self.metrics_table.setHorizontalHeaderLabels(headers)
        
        # Collect all unique metric keys
        all_metrics = set()
        for c in self.checkpoints:
            if c.eval_metrics:
                all_metrics.update(c.eval_metrics.keys())
                
        metrics = sorted(list(all_metrics))
        rows = []
        
        for m in metrics:
            if m.startswith("noise_"):
                rows.append((m, "improvement"))
                rows.append((m, "success_rate"))
                rows.append((m, "cos_before_mean"))
                rows.append((m, "cos_after_mean"))
            elif m == "samples":
                rows.append((m, "norm_mean"))
                rows.append((m, "pairwise_cos_mean"))
                
        self.metrics_table.setRowCount(len(rows))
        
        # Fill table
        for r, (cat, key) in enumerate(rows):
            name = f"{cat} - {key}"
            self.metrics_table.setItem(r, 0, QTableWidgetItem(name))
            
            for c_idx, ckpt in enumerate(self.checkpoints):
                val_item = QTableWidgetItem("--")
                if ckpt.eval_metrics and cat in ckpt.eval_metrics:
                    val = ckpt.eval_metrics[cat].get(key)
                    if val is not None:
                        val_item.setText(f"{val:.4f}")
                        
                        # Add colors for deltas if > 1 checkpoint
                        if c_idx > 0:
                            prev_val_raw = self.checkpoints[c_idx-1].eval_metrics.get(cat, {}).get(key)
                            if prev_val_raw is not None:
                                delta = val - prev_val_raw
                                if abs(delta) > 1e-4:
                                    # Green if positive delta is good
                                    is_good = (delta > 0)
                                    color = "#00ff88" if is_good else "#ff4466"
                                    val_item.setForeground(pg.mkColor(color))
                                    val_item.setText(f"{val:.4f} ({delta:+.4f})")
                
                self.metrics_table.setItem(r, c_idx + 1, val_item)
                
        self.metrics_table.resizeColumnsToContents()
        
        # Update Plot (plot noise improvements)
        colors = ["#00e5ff", "#ff00ff", "#00ff88", "#ffff00", "#ff8800"]
        
        noise_levels = [m for m in metrics if m.startswith("noise_")]
        noise_levels.sort(key=lambda x: float(x.split("_")[1]))
        
        x_ticks = [(i, n) for i, n in enumerate(noise_levels)]
        self.plot.getAxis('bottom').setTicks([x_ticks])
        
        for c_idx, ckpt in enumerate(self.checkpoints):
            y_vals = []
            x_vals = []
            if ckpt.eval_metrics:
                for i, n in enumerate(noise_levels):
                    if n in ckpt.eval_metrics:
                        y_vals.append(ckpt.eval_metrics[n].get("improvement", 0.0))
                        x_vals.append(i)
                
                if x_vals:
                    c = colors[c_idx % len(colors)]
                    name = f"Epoch {ckpt.epoch}"
                    self.plot.plot(x_vals, y_vals, name=name, pen=pg.mkPen(color=c, width=2), symbol='o', symbolBrush=c)
