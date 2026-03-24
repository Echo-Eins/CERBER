"""
3D Energy Landscape Widget using pyqtgraph's opengl module.
"""
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QWidget, QVBoxLayout

class Landscape3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#0a0a0a')
        self.view.setCameraPosition(distance=3.0, elevation=30, azimuth=45)
        self.layout.addWidget(self.view)
        
        # Grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=4, y=4, z=0)
        self.grid.setSpacing(x=0.5, y=0.5, z=0)
        self.view.addItem(self.grid)
        
        # Items
        self.surface = None
        self.trajectory_line = None
        self.points = {}
        
    def set_surface(self, x, y, z):
        if self.surface is not None:
            self.view.removeItem(self.surface)
            
        # Normalize Z for colormap
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)
        
        # Simple inferno-like colormap mapping
        colors = np.zeros((*z.shape, 4))
        colors[..., 0] = z_norm  # Red
        colors[..., 1] = z_norm * 0.5  # Green
        colors[..., 2] = 0.2  # Blue
        colors[..., 3] = 0.8  # Alpha
        
        self.surface = gl.GLSurfacePlotItem(x=x, y=y, z=z, colors=colors, computeNormals=False)
        self.view.addItem(self.surface)
        
        # Auto-center camera
        self.view.pan(-x.mean(), -y.mean(), -(z_max+z_min)/2)
        self.view.setCameraPosition(distance=max(x.max()-x.min(), y.max()-y.min()) * 1.5)
        
    def add_point(self, name, pos, color, size=10):
        if name in self.points:
            self.view.removeItem(self.points[name])
            
        pt = gl.GLScatterPlotItem(pos=np.array([pos]), color=color, size=size)
        self.view.addItem(pt)
        self.points[name] = pt
        
    def set_trajectory(self, points, color=(0.0, 1.0, 1.0, 1.0)):
        if self.trajectory_line is not None:
            self.view.removeItem(self.trajectory_line)
            
        if not points or len(points) < 2:
            return
            
        pts = np.array(points)
        self.trajectory_line = gl.GLLinePlotItem(pos=pts, color=color, width=2.0)
        self.view.addItem(self.trajectory_line)
        
    def clear(self):
        if self.surface is not None:
            self.view.removeItem(self.surface)
            self.surface = None
        if self.trajectory_line is not None:
            self.view.removeItem(self.trajectory_line)
            self.trajectory_line = None
        for pt in self.points.values():
            self.view.removeItem(pt)
        self.points.clear()
