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
        
        # Scaling parameters
        self._z_min = 0.0
        self._z_range = 1.0
        self._xy_span = 1.0
        
    def scale_z(self, z_val):
        """Scale a true Z value to the visual Z space used by the surface plot."""
        return (z_val - self._z_min) / self._z_range * (self._xy_span * 0.2)
        
    def set_surface(self, x, y, z):
        if self.surface is not None:
            self.view.removeItem(self.surface)
            
        # Z Scaling: to make the surface appropriately hilly without being flat or needle-thin
        # we scale Z to match the roughly ~1.0 XY range (or half the grid span)
        self._xy_span = max(x.max() - x.min(), y.max() - y.min(), 1.0)
        self._z_min = z.min()
        z_max = z.max()
        self._z_range = max(z_max - self._z_min, 1e-8)
        
        # Scale Z to be about 20% of the maximum XY span
        z_scaled = self.scale_z(z)
        
        # Use z_scaled for height and color calculations
        z_norm = (z_scaled - z_scaled.min()) / (z_scaled.max() - z_scaled.min() + 1e-8)
        
        # Simple inferno-like colormap mapping
        colors = np.zeros((*z.shape, 4))
        colors[..., 0] = z_norm  # Red
        colors[..., 1] = z_norm * 0.5  # Green
        colors[..., 2] = 0.2  # Blue
        colors[..., 3] = 0.8  # Alpha
        
        self.surface = gl.GLSurfacePlotItem(x=x, y=y, z=z_scaled, colors=colors, computeNormals=False)
        self.view.addItem(self.surface)
        
        # Auto-center camera
        self.view.pan(-x.mean(), -y.mean(), -(z_scaled.max() + z_scaled.min())/2)
        
        # Adjust zoom: higher distance means viewing from further away
        self.view.setCameraPosition(distance=self._xy_span * 1.5, elevation=30, azimuth=45)
        
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
