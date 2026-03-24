"""
CERBER Training Monitor - Main Window
"""
import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QLabel, QPushButton, QFileDialog, QSplitter, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QFileSystemWatcher, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
import numpy as np
import torch
import math

from tools.cerber_monitor.checkpoint_loader import load_checkpoint, load_model_from_checkpoint, CheckpointInfo
from tools.cerber_monitor.metrics_parser import load_metrics_streaming, TrainingRun
from tools.cerber_monitor.widgets.training_chart import TrainingChartWidget
from tools.cerber_monitor.widgets.landscape_3d import Landscape3DWidget
from tools.cerber_monitor.widgets.checkpoint_comparison import CheckpointComparisonWidget
import csv
from cebcm.data.dataset import SONARVectorDataset
from cebcm.visualization.energy_landscape import scan_energy_landscape
from cebcm.inference.langevin import run_langevin



class MetricCard(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 20, 15, 15)
        
        self.val_label = QLabel("--")
        self.val_label.setObjectName("metric_value")
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.val_label)
        
    def set_value(self, val, fmt="{:.4f}", color_override=None):
        if val is None:
            self.val_label.setText("--")
            self.val_label.setStyleSheet("")
            return
            
        if isinstance(val, (float, int)):
            self.val_label.setText(fmt.format(val))
        else:
            self.val_label.setText(str(val))
            
        if color_override:
            self.val_label.setStyleSheet(f"color: {color_override};")
        else:
            self.val_label.setStyleSheet("")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CERBER Monitor")
        self.setMinimumSize(1200, 800)
        
        # Loaded data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.model = None

        
        # File watcher for live updates
        self.watcher = QFileSystemWatcher()
        self.watcher.fileChanged.connect(self.on_file_changed)
        
        self.init_ui()
        
    def init_ui(self):
        # Central widget
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.main_layout = QVBoxLayout(self.central)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Top toolbar
        self.toolbar_layout = QHBoxLayout()
        
        self.title_label = QLabel("CERBER Training Monitor")
        self.title_label.setObjectName("title")
        self.toolbar_layout.addWidget(self.title_label)
        self.toolbar_layout.addStretch()
        
        self.btn_load_ckpt = QPushButton("Load Checkpoint")
        self.btn_load_ckpt.clicked.connect(self.load_checkpoint_dialog)
        self.toolbar_layout.addWidget(self.btn_load_ckpt)
        
        self.btn_load_metrics = QPushButton("Load Metrics JSON")
        self.btn_load_metrics.clicked.connect(self.load_metrics_dialog)
        self.toolbar_layout.addWidget(self.btn_load_metrics)
        
        self.btn_export = QPushButton("Export Metrics CSV")
        self.btn_export.clicked.connect(self.export_metrics_csv)
        self.toolbar_layout.addWidget(self.btn_export)
        
        self.main_layout.addLayout(self.toolbar_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # --- Tab 1: Training Curves ---
        self.tab_curves = QWidget()
        self.tabs.addTab(self.tab_curves, "Training Curves")
        self.init_curves_tab()
        
        # --- Tab 2: Checkpoint Inspector ---
        self.tab_inspector = QWidget()
        self.tabs.addTab(self.tab_inspector, "Checkpoint Inspector")
        self.init_inspector_tab()
        
        # --- Tab 3: Checkpoint Comparison ---
        self.tab_compare = QWidget()
        self.tabs.addTab(self.tab_compare, "Checkpoint Comparison")
        self.init_compare_tab()
        
        # --- Tab 4: Energy Landscape ---
        self.tab_landscape = QWidget()
        self.tabs.addTab(self.tab_landscape, "Energy Landscape 3D")
        self.init_landscape_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def init_curves_tab(self):
        layout = QVBoxLayout(self.tab_curves)
        
        # Top metrics summary
        self.summary_layout = QHBoxLayout()
        self.card_epoch = MetricCard("Epochs")
        self.card_loss = MetricCard("Final Loss")
        self.card_best_imp = MetricCard("Best Improvement")
        self.card_time = MetricCard("Total Time (s)")
        
        self.summary_layout.addWidget(self.card_epoch)
        self.summary_layout.addWidget(self.card_loss)
        self.summary_layout.addWidget(self.card_best_imp)
        self.summary_layout.addWidget(self.card_time)
        layout.addLayout(self.summary_layout)
        
        # Charts splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)
        
        # Chart 1: Loss
        self.chart_loss = TrainingChartWidget("Training Loss")
        splitter.addWidget(self.chart_loss)
        
        # Chart 2: Cosine Improvement
        self.chart_cos = TrainingChartWidget("Cosine Improvement (Eval)")
        splitter.addWidget(self.chart_cos)
        
        # Chart 3: Sample Quality (if available)
        self.chart_samples = TrainingChartWidget("Sample Norm / Gap")
        splitter.addWidget(self.chart_samples)
        
        # Set default splitter sizes
        splitter.setSizes([400, 300, 200])
        
    def init_inspector_tab(self):
        layout = QVBoxLayout(self.tab_inspector)
        
        # Header info
        self.ckpt_header = QLabel("No checkpoint loaded")
        self.ckpt_header.setObjectName("subtitle")
        layout.addWidget(self.ckpt_header)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        
        # Left side: Properties list
        prop_widget = QWidget()
        prop_layout = QVBoxLayout(prop_widget)
        prop_layout.setContentsMargins(0, 0, 0, 0)
        
        self.props_table = QTableWidget(0, 2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        prop_layout.addWidget(self.props_table)
        
        splitter.addWidget(prop_widget)
        
        # Right side: Layer stats
        layer_widget = QWidget()
        layer_layout = QVBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.layers_table = QTableWidget(0, 6)
        self.layers_table.setHorizontalHeaderLabels(["Layer", "Shape", "Params", "Mean", "Std", "Norm"])
        self.layers_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.layers_table.verticalHeader().setVisible(False)
        self.layers_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layer_layout.addWidget(self.layers_table)
        
        splitter.addWidget(layer_widget)
        splitter.setSizes([300, 700])
        
    def init_compare_tab(self):
        layout = QVBoxLayout(self.tab_compare)
        
        self.btn_add_compare = QPushButton("Add Checkpoint for Comparison")
        self.btn_add_compare.clicked.connect(self.add_checkpoint_for_comparison)
        layout.addWidget(self.btn_add_compare)
        
        self.compare_widget = CheckpointComparisonWidget()
        layout.addWidget(self.compare_widget, 1)

    def init_landscape_tab(self):
        layout = QHBoxLayout(self.tab_landscape)
        
        # Left controls
        controls = QWidget()
        controls.setFixedWidth(250)
        c_layout = QVBoxLayout(controls)
        
        c_layout.addWidget(QLabel("1. Load Checkpoint first"))
        c_layout.addWidget(QLabel("2. Load Data (.pt):"))
        
        self.btn_load_data = QPushButton("Select Data")
        self.btn_load_data.clicked.connect(self.load_data_dialog)
        c_layout.addWidget(self.btn_load_data)
        
        c_layout.addWidget(QLabel("Sample Index:"))
        from PyQt6.QtWidgets import QSpinBox
        self.sample_idx_spin = QSpinBox()
        self.sample_idx_spin.setRange(0, 0)
        self.sample_idx_spin.setEnabled(False)
        c_layout.addWidget(self.sample_idx_spin)
        
        self.btn_scan = QPushButton("Scan Landscape")
        self.btn_scan.setObjectName("primary")
        self.btn_scan.setEnabled(False)
        self.btn_scan.clicked.connect(self.scan_current_sample)
        c_layout.addWidget(self.btn_scan)
        
        c_layout.addStretch()
        layout.addWidget(controls)
        
        # Right 3D View
        self.landscape_view = Landscape3DWidget()
        layout.addWidget(self.landscape_view, 1)
        
    # --- Interactivity ---
    
    def load_metrics_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Metrics", 
            str(Path.cwd() / "experiments"), 
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.load_metrics_file(path)
            
    def load_metrics_file(self, path):
        self.current_metrics_path = path
        self.statusBar().showMessage(f"Loading {path}...")
        
        run_data = load_metrics_streaming(path)
        if run_data is None:
            self.statusBar().showMessage(f"Failed to load {path}")
            return
            
        self.metrics_data = run_data
        
        # Add to watcher
        if path not in self.watcher.files():
            self.watcher.addPath(path)
            
        self.update_curves_ui()
        self.statusBar().showMessage(f"Loaded {path} successfully.")
        
    def load_checkpoint_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", 
            str(Path.cwd() / "experiments"), 
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.load_checkpoint_file(path)
            
    def load_checkpoint_file(self, path):
        self.current_ckpt_path = path
        self.statusBar().showMessage(f"Loading {path}...")
        
        try:
            self.model, info = load_model_from_checkpoint(path, self.device)
            self.ckpt_data = info
            self.update_inspector_ui()
            
            if self.dataset is not None:
                self.btn_scan.setEnabled(True)
                
            self.statusBar().showMessage(f"Loaded {path} successfully ({info.model_type}).")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading checkpoint: {str(e)}")
            
    def add_checkpoint_for_comparison(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Checkpoint to Comparison", 
            str(Path.cwd() / "experiments"), 
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)"
        )
        if path:
            try:
                info = load_checkpoint(path)
                self.compare_widget.add_checkpoint(info)
                self.statusBar().showMessage(f"Added {Path(path).name} to comparison.")
            except Exception as e:
                self.statusBar().showMessage(f"Error adding checkpoint: {str(e)}")
                
    def export_metrics_csv(self):
        if not self.metrics_data:
            self.statusBar().showMessage("No metrics loaded to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics CSV", 
            str(Path.cwd() / "metrics_export.csv"), 
            "CSV Files (*.csv)"
        )
        if not path:
            return
            
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = ["Epoch", "Phase", "Loss"]
                for n in self.metrics_data.noise_levels:
                    header.append(f"{n}_cos_improve")
                    header.append(f"{n}_success")
                writer.writerow(header)
                
                # Data
                metrics = self.metrics_data
                for i, ep in enumerate(metrics.loss_epochs):
                    row = [ep, metrics.phase_labels[i], metrics.loss_values[i]]
                    
                    # Fill eval metrics if this was an eval epoch
                    for n in metrics.noise_levels:
                        if ep in metrics.eval_epochs:
                            idx = metrics.eval_epochs.index(ep)
                            if idx < len(metrics.improvements[n]):
                                row.append(metrics.improvements[n][idx])
                                row.append(metrics.success_rates[n][idx])
                            else:
                                row.extend(["", ""])
                        else:
                            row.extend(["", ""])
                            
                    writer.writerow(row)
                    
            self.statusBar().showMessage(f"Exported metrics to {path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting CSV: {str(e)}")
            
    def on_file_changed(self, path):
        if path == self.current_metrics_path:
            self.statusBar().showMessage(f"Live update from {path}...")
            # Small delay to ensure file write is complete
            QTimer.singleShot(100, lambda: self.load_metrics_file(path))
            
    def load_data_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load SONAR Dataset", 
            str(Path.cwd() / "data"), 
            "PyTorch Binaries (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.statusBar().showMessage(f"Loading dataset {path}...")
            try:
                self.dataset = SONARVectorDataset(path)
                self.sample_idx_spin.setRange(0, len(self.dataset) - 1)
                self.sample_idx_spin.setEnabled(True)
                self.btn_scan.setEnabled(self.model is not None)
                self.statusBar().showMessage(f"Loaded {len(self.dataset)} samples from {Path(path).name}")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading dataset: {str(e)}")

    def scan_current_sample(self):
        if not self.model or not self.dataset:
            return
            
        idx = self.sample_idx_spin.value()
        v_clean = self.dataset[idx].unsqueeze(0).to(self.device)
        
        # Add noise
        noise_scale = 0.15
        norms = v_clean.norm(dim=-1, keepdim=True)
        v_noisy = v_clean + torch.randn_like(v_clean) * noise_scale * norms
        
        self.statusBar().showMessage(f"Scanning landscape for sample {idx}... This may take a few seconds.")
        self.btn_scan.setEnabled(False)
        
        # We need a background thread for this to not freeze UI
        # For now, do it synchronously (can be optimized later)
        try:
            is_uncond = self.ckpt_data.model_type == "unconditional_energy"
            data = scan_energy_landscape(
                energy_fn=self.model,
                v_clean=v_clean,
                v_noisy=v_noisy,
                grid_size=40,  # Lower res for faster UI response
                unconditional=is_uncond
            )
            
            # Update 3D view
            x = data.grid_x.numpy()
            y = data.grid_y.numpy()
            z = data.energy.numpy()
            
            self.landscape_view.clear()
            self.landscape_view.set_surface(x, y, z)
            
            # Map points onto surface (using empirical Z height offset to ensure visibility)
            # Find the actual scaled Z max from the plotted surface
            z_scaled = self.landscape_view.surface.zData
            clean_z = np.max(z_scaled) + (np.max(z_scaled) - np.min(z_scaled)) * 0.05
            noisy_z = clean_z
            
            self.landscape_view.add_point("clean", (0, 0, clean_z), (0.0, 1.0, 0.0, 1.0))
            self.landscape_view.add_point("noisy", (data.v_noisy_xy[0], data.v_noisy_xy[1], noisy_z), (1.0, 0.0, 0.0, 1.0))
            
            mid = data.cosine_sim.shape[0] // 2
            self.statusBar().showMessage(f"Scan complete. Cos(clean, noisy): {data.cosine_sim[mid, mid]:.4f}")
        except Exception as e:
            self.statusBar().showMessage(f"Error scanning landscape: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_scan.setEnabled(True)

    # --- UI Updates ---
    
    def update_curves_ui(self):
        data = self.metrics_data
        if not data: return
        
        # Update cards
        self.card_epoch.set_value(data.num_epochs, "{}")
        if data.loss_values:
            self.card_loss.set_value(data.loss_values[-1])
        self.card_best_imp.set_value(data.best_improvement, "{:+.4f}", "#00ff88" if data.best_improvement > 0 else "#ff4466")
        self.card_time.set_value(data.total_time_seconds, "{:.0f}")
        
        # Update Loss Chart
        self.chart_loss.clear()
        
        # Split loss by phase if possible
        if any(p == "nce" for p in data.phase_labels):
            nce_x = [e for e, p in zip(data.loss_epochs, data.phase_labels) if p == "nce"]
            nce_y = [v for v, p in zip(data.loss_values, data.phase_labels) if p == "nce"]
            em_x = [e for e, p in zip(data.loss_epochs, data.phase_labels) if p == "em"]
            em_y = [v for v, p in zip(data.loss_values, data.phase_labels) if p == "em"]
            
            if nce_x: self.chart_loss.add_line("NCE Loss", nce_x, nce_y, "#8888aa")
            if em_x: self.chart_loss.add_line("EM Loss", em_x, em_y, "#ff00ff")
        else:
            self.chart_loss.add_line("Loss", data.loss_epochs, data.loss_values, "#ff00ff")
            
        # Update Eval Chart
        self.chart_cos.clear()
        colors = ["#00e5ff", "#00ff88", "#ff8800", "#ff4466", "#ffff00"]
        
        for i, noise in enumerate(data.noise_levels):
            if noise in data.improvements:
                color = colors[i % len(colors)]
                self.chart_cos.add_line(f"Δ {noise}", data.eval_epochs, data.improvements[noise], color)
                # Mark evaluations with scatter points
                self.chart_cos.add_scatter(f"pts_{noise}", data.eval_epochs, data.improvements[noise], color)
                
        # Update Samples Chart
        self.chart_samples.clear()
        if data.sample_epochs:
            self.chart_samples.add_line("Sample Norm", data.sample_epochs, data.sample_norm_mean, "#00e5ff")
            self.chart_samples.add_line("Pairwise Cos", data.sample_epochs, data.sample_pairwise_cos, "#ff00ff")
            
    def update_inspector_ui(self):
        info = self.ckpt_data
        if not info: return
        
        self.ckpt_header.setText(f"{Path(info.path).name} — {info.model_type} (Epoch {info.epoch})")
        
        # Properties
        props = [
            ("Arch Type", info.model_type),
            ("Dim", info.config.get("energy_dim", "1024")),
            ("Hidden Dims", str(info.config.get("energy_hidden_dims", "[]"))),
            ("Norm Mode", info.config.get("norm_mode", "orthonorm")),
            ("Activation", info.config.get("activation", "groupsort")),
            ("Total Params", f"{info.total_params:,}"),
        ]
        if info.log_energy_scale is not None:
            props.append(("E_scale (exp)", f"{math.exp(info.log_energy_scale):.4f}"))
            
        if info.eval_metrics:
            props.append(("", ""))
            props.append(("--- EVAL METRICS ---", "---"))
            for noise, m in info.eval_metrics.items():
                if noise.startswith("noise_") and isinstance(m, dict):
                    imp = m.get('improvement', 0.0)
                    succ = m.get('success_rate', 0.0)
                    props.append((f"{noise} Δ Cos", f"{imp:+.4f}"))
                    props.append((f"{noise} Success", f"{succ:.0%}"))
                    
        self.props_table.setRowCount(len(props))
        for r, (k, v) in enumerate(props):
            self.props_table.setItem(r, 0, QTableWidgetItem(k))
            self.props_table.setItem(r, 1, QTableWidgetItem(str(v)))
            
        # Layer Stats
        self.layers_table.setRowCount(len(info.layer_stats))
        for r, stat in enumerate(info.layer_stats):
            self.layers_table.setItem(r, 0, QTableWidgetItem(stat.name))
            self.layers_table.setItem(r, 1, QTableWidgetItem(str(stat.shape)))
            self.layers_table.setItem(r, 2, QTableWidgetItem(f"{stat.num_params:,}"))
            self.layers_table.setItem(r, 3, QTableWidgetItem(f"{stat.mean:.6f}"))
            self.layers_table.setItem(r, 4, QTableWidgetItem(f"{stat.std:.6f}"))
            self.layers_table.setItem(r, 5, QTableWidgetItem(f"{stat.norm:.4f}"))

