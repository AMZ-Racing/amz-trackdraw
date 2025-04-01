import os
import math
import csv
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap
import yaml
from scipy.interpolate import splprep, splev
from utils_qt import (create_closed_spline, generate_offset_boundaries, sample_cones)
from track_canvas_qt import TrackCanvas


class FSTrackDraw(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrackDraw - PyQt")
        
        # Global configuration file
        self.config_file = "config/track_config.yaml"
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.track_width = config.get('track_width', 3.0)
            self.default_cone_distance = config.get('cone_distance', 3.5)
            self.min_boundary_backoff = config.get('min_boundary_backoff', 10.0)
            self.n_points_midline = config.get('n_points_midline', 300)
            self.location_name = config.get('standard_location', "empty")

        # Load the location-specific details
        self.folderpath_location = "location_images/" + self.location_name
        self.filename_location_config = self.location_name + "_config.yaml"
        self.fpath_location_config = os.path.join(self.folderpath_location, self.filename_location_config)
        with open(self.fpath_location_config, 'r') as file:
            config = yaml.safe_load(file)
            self.px_per_m = config.get('px_per_m', 10.0)
            sat_img_file = config.get('sat_img_path', '')
            occup_img_file = config.get('occ_img_path', '')
            self.fpath_location_sat_img = os.path.join(self.folderpath_location, sat_img_file)
            self.fpath_location_occup_img = os.path.join(self.folderpath_location, occup_img_file)
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create canvas (handles its own obstacle loading with proper scaling)
        self.canvas = TrackCanvas(self)
        self.canvas.setMinimumSize(600, 600)
        self.main_layout.addWidget(self.canvas, 1)
        
        # Right-side UI
        self.ui_frame = QWidget()
        self.ui_frame.setFixedWidth(250)
        self.ui_layout = QVBoxLayout(self.ui_frame)
        self.ui_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(self.ui_frame)
        
        # Mode label
        self.mode_label = QLabel("Mode: Add")
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.ui_layout.addWidget(self.mode_label)
        
        # Mode buttons
        self.add_button = QPushButton("Add Control Point")
        self.add_button.clicked.connect(self.activate_add_mode)
        self.ui_layout.addWidget(self.add_button)
        
        self.remove_button = QPushButton("Remove Control Point")
        self.remove_button.clicked.connect(self.activate_remove_mode)
        self.ui_layout.addWidget(self.remove_button)
        
        self.move_button = QPushButton("Move Control Point")
        self.move_button.clicked.connect(self.activate_move_mode)
        self.ui_layout.addWidget(self.move_button)
        
        self.swap_button = QPushButton("Swap Boundaries")
        self.swap_button.clicked.connect(self.swap_boundaries)
        self.ui_layout.addWidget(self.swap_button)
        
        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.ui_layout.addWidget(self.export_button)
        
        # Cone spacing input
        self.ui_layout.addWidget(QLabel("Cone spacing (m):"))
        self.cone_spacing_entry = QLineEdit(str(self.default_cone_distance))
        self.cone_spacing_entry.returnPressed.connect(self.redraw)
        self.ui_layout.addWidget(self.cone_spacing_entry)
        
        # Backoff input
        self.ui_layout.addWidget(QLabel("Backoff (m):"))
        self.backoff_entry = QLineEdit(str(self.min_boundary_backoff))
        self.backoff_entry.returnPressed.connect(self.redraw)
        self.ui_layout.addWidget(self.backoff_entry)
        
        # Track width input
        self.ui_layout.addWidget(QLabel("Track width (m):"))
        self.track_width_entry = QLineEdit(str(self.track_width))
        self.track_width_entry.returnPressed.connect(self.redraw)
        self.ui_layout.addWidget(self.track_width_entry)
        
        # Statistics labels
        self.track_length_label = QLabel("Track Length: --")
        self.track_length_label.setStyleSheet("font-weight: bold;")
        self.ui_layout.addWidget(self.track_length_label)
        
        self.min_radius_label = QLabel("Min Radius: --")
        self.min_radius_label.setStyleSheet("font-weight: bold;")
        self.ui_layout.addWidget(self.min_radius_label)
        
        self.cone_count_label = QLabel("Cones: \n Blue: -, Yellow: - \n Total: -")
        self.cone_count_label.setStyleSheet("font-weight: bold;")
        self.ui_layout.addWidget(self.cone_count_label)
        
        # Add stretch to push everything up
        self.ui_layout.addStretch(1)
        
        # Data for control points and track
        self.control_points = []  # List of QPointF
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries_swapped = False
        
        # GUI control variables
        self.mode = "add"  # Modes: "add", "remove", "move"
        self.selected_point_index = None
        self.dragging = False

        # Add logo at the bottom right
        self.logo_label = QLabel()
        self.logo_pixmap = QPixmap("TrackDraw_Logo.png")  # Load your logo image
        # Add the logo label to the layout
        self.ui_layout.addStretch(1)  # Push everything up
        self.ui_layout.addWidget(self.logo_label, 0, Qt.AlignRight|Qt.AlignBottom)
        # Handle window resize events
        self.main_widget.resizeEvent = self.on_resize
        # Initial logo setup
        self.update_logo_size()
        
        # Initialize
        self.redraw()

    def on_resize(self, event):
      self.update_logo_size()
      super().resizeEvent(event)  # Call parent's resize handler

    def update_logo_size(self):
      """Adjusts logo size when window is resized"""
      if hasattr(self, 'logo_pixmap') and self.logo_pixmap and not self.logo_pixmap.isNull():
          # Increased base size (adjust these values as needed)
          max_width = min(200, self.ui_frame.width() - 20)  # 20px padding
          scaled_pixmap = self.logo_pixmap.scaledToWidth(
              max_width, 
              Qt.SmoothTransformation
          )
          self.logo_label.setPixmap(scaled_pixmap)
          self.logo_label.setFixedSize(scaled_pixmap.size())  # Prevent layout shifting
    
    def activate_add_mode(self):
        self.mode = "add"
        self.mode_label.setText("Mode: Add")
        
    def activate_remove_mode(self):
        self.mode = "remove"
        self.mode_label.setText("Mode: Remove")
        
    def activate_move_mode(self):
        self.mode = "move"
        self.mode_label.setText("Mode: Move")
        
    def swap_boundaries(self):
        self.boundaries_swapped = not self.boundaries_swapped
        self.redraw()
        
    def handle_canvas_click(self, pos):
        x, y = pos.x(), pos.y()
        if self.mode == "add":
            self.control_points.append(QPointF(x, y))
            self.redraw()
        elif self.mode == "remove":
            idx = self.find_near_control_point(x, y)
            if idx is not None:
                del self.control_points[idx]
                self.redraw()
        elif self.mode == "move":
            idx = self.find_near_control_point(x, y, threshold=20)
            if idx is not None:
                self.selected_point_index = idx
                self.dragging = True
                
    def handle_canvas_drag(self, pos):
        if self.dragging and self.selected_point_index is not None and self.mode == "move":
            self.control_points[self.selected_point_index] = QPointF(pos.x(), pos.y())
            self.redraw()
            
    def handle_canvas_release(self, pos):
        self.dragging = False
        self.selected_point_index = None
        
    def find_near_control_point(self, x, y, threshold=10):
        for i, pt in enumerate(self.control_points):
            if (pt.x() - x) ** 2 + (pt.y() - y) ** 2 < threshold ** 2:
                return i
        return None
        
    def export_csv(self):
        """Export the cone positions as CSV in meters."""
        try:
            cone_spacing = float(self.cone_spacing_entry.text())
        except ValueError:
            cone_spacing = self.default_cone_distance
            
        if self.centerline is None or self.left_boundary is None or self.right_boundary is None:
            QMessageBox.critical(self, "Export Error", "No track defined yet!")
            return
            
        # Convert boundaries to numpy arrays
        left_boundary = np.array([[p.x(), p.y()] for p in self.left_boundary])
        right_boundary = np.array([[p.x(), p.y()] for p in self.right_boundary])
        
        left_cones = sample_cones(left_boundary, cone_spacing, self.px_per_m)
        right_cones = sample_cones(right_boundary, cone_spacing, self.px_per_m)
        
        # Define new coordinate system
        origin = np.array([self.centerline[0].x(), self.centerline[0].y()])
        if len(self.centerline) < 2:
            QMessageBox.critical(self, "Export Error", "Not enough centerline points!")
            return
            
        tangent = np.array([self.centerline[1].x(), self.centerline[1].y()]) - origin
        theta = math.atan2(tangent[1], tangent[0])
        
        # Rotation matrix for -theta
        R = np.array([[math.cos(-theta), -math.sin(-theta)],
                      [math.sin(-theta),  math.cos(-theta)]])
                      
        def transform(pt):
            local = np.array(pt) - origin
            local_rot = R.dot(local)
            return local_rot / self.px_per_m  # convert pixels to meters
            
        left_cones_m = [transform(pt) for pt in left_cones]
        right_cones_m = [transform(pt) for pt in right_cones]
        
        if self.boundaries_swapped:
            left_tag = "yellow"
            right_tag = "blue"
        else:
            left_tag = "blue"
            right_tag = "yellow"
            
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not filename:
            return
            
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["tag", "x", "y"])
            for pt in right_cones_m:
                writer.writerow([right_tag, pt[0], pt[1]])
            for pt in left_cones_m:
                writer.writerow([left_tag, pt[0], pt[1]])
                
        QMessageBox.information(self, "Export", f"Track exported to {filename}")
        
    def redraw(self):
        self.canvas.update_drawing(
            self.control_points,
            self.centerline,
            self.left_boundary,
            self.right_boundary,
            self.boundaries_swapped
        )
        
        # Update track parameters
        try:
            self.track_width = float(self.track_width_entry.text())
        except ValueError:
            pass
            
        try:
            backoff_val = float(self.backoff_entry.text())
        except ValueError:
            backoff_val = self.min_boundary_backoff
        self.min_boundary_backoff = backoff_val
        
        # If at least 3 control points, compute and draw track
        if len(self.control_points) > 3:
            # Convert QPointF to numpy array for spline calculation
            pts = np.array([[p.x(), p.y()] for p in self.control_points])
            centerline = create_closed_spline(pts, num_points=self.n_points_midline)
            self.centerline = [QPointF(p[0], p[1]) for p in centerline]
            
            boundaries = generate_offset_boundaries(centerline, self.track_width, self.px_per_m)
            if boundaries[0] is None or boundaries[1] is None:
                return
                
            left_boundary, right_boundary = boundaries
            self.left_boundary = [QPointF(p[0], p[1]) for p in left_boundary]
            self.right_boundary = [QPointF(p[0], p[1]) for p in right_boundary]
            
            # Calculate statistics
            track_length = self.calculate_track_length()
            min_radius = self.calculate_min_radius()
            blue_cones, yellow_cones, total_cones = self.count_cones()
            
            # Update the labels with calculated statistics
            self.track_length_label.setText(f"Track Length: {track_length:.2f} m")
            self.min_radius_label.setText(f"Min Radius: {min_radius:.2f} m")
            self.cone_count_label.setText(f"Cones: \n Blue: {blue_cones}, Yellow: {yellow_cones} \n Total: {total_cones}")
            
    def calculate_track_length(self):
        """Calculate the total length of the track (centerline B-spline)."""
        if len(self.centerline) < 2:
            return 0
        total_length = 0
        for i in range(1, len(self.centerline)):
            x1, y1 = self.centerline[i - 1].x(), self.centerline[i - 1].y()
            x2, y2 = self.centerline[i].x(), self.centerline[i].y()
            total_length += math.hypot(x2 - x1, y2 - y1)
        return total_length / self.px_per_m  # Convert from pixels to meters
        
    def calculate_min_radius(self):
        """Calculate the minimum radius of curvature of the centerline B-spline."""
        if len(self.centerline) < 3:
            return float('inf')  # No valid curvature can be calculated
            
        # Extract x and y coordinates from the centerline
        x_coords = [p.x() for p in self.centerline]
        y_coords = [p.y() for p in self.centerline]
        
        # Parametrize the centerline
        tck, u = splprep([x_coords, y_coords], s=0, per=True)  # Use per=True for closed splines
        
        min_radius = float('inf')
        
        # Calculate curvature at each point on the B-spline
        for t in np.linspace(0, 1, len(self.centerline)):
            # First derivatives (x', y')
            dx, dy = splev(t, tck, der=1)
            
            # Second derivatives (x'', y'')
            ddx, ddy = splev(t, tck, der=2)
            
            # Curvature formula
            numerator = abs(dx * ddy - dy * ddx)
            denominator = (dx**2 + dy**2)**(3/2)
            
            if denominator == 0:
                continue  # Skip if denominator is zero
                
            curvature = numerator / denominator
            radius = 1 / curvature
            
            # Track the minimum radius
            min_radius = min(min_radius, radius)
            
        return min_radius / self.px_per_m  # Convert from pixels to meters
        
    def count_cones(self):
        """Count the number of blue, yellow, and total cones."""
        try:
            cone_spacing = float(self.cone_spacing_entry.text())
        except ValueError:
            cone_spacing = self.default_cone_distance
            
        if self.left_boundary is None or self.right_boundary is None:
            return 0, 0, 0
            
        left_boundary = np.array([[p.x(), p.y()] for p in self.left_boundary])
        right_boundary = np.array([[p.x(), p.y()] for p in self.right_boundary])
        
        left_cones = sample_cones(left_boundary, cone_spacing, self.px_per_m)
        right_cones = sample_cones(right_boundary, cone_spacing, self.px_per_m)
        
        if self.boundaries_swapped:
            return len(right_cones), len(left_cones), len(right_cones) + len(left_cones)
        else:
            return len(left_cones), len(right_cones), len(left_cones) + len(right_cones)