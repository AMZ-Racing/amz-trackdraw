import os
import math
import csv
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
from PIL import Image
import cv2
import yaml
from shapely.geometry import Polygon, LineString
from scipy.interpolate import splprep, splev

def create_closed_spline(control_points, num_points=100):
    """
    Compute a smooth, closed B-spline from a list of control points.
    """
    pts = np.array(control_points)
    # add first point to end to close the loop
    pts = np.vstack((pts, pts[0]))
    # if we have only one point, return it
    if len(pts) <= 2:
        return pts
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    return np.column_stack((x_fine, y_fine))


def robust_parallel_offset(ls, distance, side, join_style=2):
    """
    Compute a parallel offset of a LineString using shapely's parallel_offset.
    If the result is a MultiLineString, return the longest LineString.
    """
    offset = ls.parallel_offset(distance, side, join_style=join_style)
    if offset.is_empty:
        return None
    if offset.geom_type == "MultiLineString":
        lines = list(offset)
        lines.sort(key=lambda l: l.length, reverse=True)
        return lines[0]
    return offset


def generate_offset_boundaries(track_points, track_width_meters, px_per_m):
    """
    Compute left and right boundaries as parallel offsets from the centerline.
    """
    track_width_px = track_width_meters * px_per_m
    ls = LineString(track_points)
    left_offset = robust_parallel_offset(ls, track_width_px / 2.0, 'left', join_style=2)
    right_offset = robust_parallel_offset(ls, track_width_px / 2.0, 'right', join_style=2)
    if left_offset is None or right_offset is None:
        return None, None
    left_coords = list(left_offset.coords)
    right_coords = list(right_offset.coords)
    return np.array(left_coords), np.array(right_coords)


def sample_cones(boundary, cone_spacing_meters, px_per_m):
    """Sample points along a boundary so that they are approximately cone_spacing_meters apart."""
    cone_spacing_px = cone_spacing_meters * px_per_m
    pts = boundary
    if len(pts) < 2:
        return pts
    distances = [0]
    for i in range(1, len(pts)):
        d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        distances.append(distances[-1] + d)
    total_length = distances[-1]
    num_cones = max(2, int(total_length // cone_spacing_px))
    sample_d = np.linspace(0, total_length, num_cones)
    sampled = []
    for sd in sample_d:
        for i in range(1, len(distances)):
            if distances[i] >= sd:
                t = (sd - distances[i-1]) / (distances[i] - distances[i-1])
                x = pts[i-1][0] + t * (pts[i][0] - pts[i-1][0])
                y = pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])
                sampled.append((x, y))
                break
    return np.array(sampled)


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

        # Load images
        self.sat_image = Image.open(self.fpath_location_sat_img)
        self.sat_qimage = QImage(self.fpath_location_sat_img)
        self.sat_pixmap = QPixmap.fromImage(self.sat_qimage)
        
        # Load occupancy map and extract obstacles
        occ_img = cv2.imread(self.fpath_location_occup_img, cv2.IMREAD_GRAYSCALE)
        _, self.binary_map = cv2.threshold(occ_img, 128, 255, cv2.THRESH_BINARY)
        self.obstacle_polygons = self.extract_obstacle_polygons(self.binary_map)
        self.free_region = self.extract_free_region(self.binary_map)
        
        # Data for control points and track
        self.control_points = []  # List of QPointF
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries_swapped = False  # False: left=blue, right=yellow
        
        # GUI control variables
        self.mode = "add"  # Modes: "add", "remove", "move"
        self.selected_point_index = None
        self.dragging = False
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Canvas for drawing
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
        
        self.cone_count_label = QLabel("Cones: Blue: -- Yellow: -- Total: --")
        self.cone_count_label.setStyleSheet("font-weight: bold;")
        self.ui_layout.addWidget(self.cone_count_label)
        
        # Add stretch to push everything up
        self.ui_layout.addStretch(1)
        
        # Initialize canvas
        self.redraw()
    
    def extract_obstacle_polygons(self, binary_map):
        """Extract obstacle contours from the occupancy map."""
        contours, _ = cv2.findContours(binary_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        if contours is not None:
            for cnt in contours:
                pts = cnt.squeeze()
                if pts.ndim == 1:
                    continue
                polys.append(pts.tolist())
        return polys
        
    def extract_free_region(self, binary_map):
        """Extract the largest external contour from the occupancy map as the free region."""
        contours, _ = cv2.findContours(binary_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        pts = largest.squeeze()
        if pts.ndim == 1 or len(pts) < 3:
            return None
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        try:
            poly = Polygon(pts)
            if not poly.is_valid or poly.is_empty:
                return None
            return poly
        except Exception as e:
            print("Error creating free region polygon:", e)
            return None
    
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
            self.boundaries_swapped,
            self.obstacle_polygons,
            self.free_region
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
            self.cone_count_label.setText(f"Cones: Blue: {blue_cones} Yellow: {yellow_cones} Total: {total_cones}")
            
        self.canvas.update()
        
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


class TrackCanvas(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        
        # Load the satellite image
        self.sat_image = QImage(parent.fpath_location_sat_img)
        self.sat_pixmap = QPixmap.fromImage(self.sat_image)
        
        # Drawing elements
        self.control_points = []
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries_swapped = False
        self.obstacle_polygons = []
        self.free_region = None
        
    def update_drawing(self, control_points, centerline, left_boundary, right_boundary, 
                      boundaries_swapped, obstacle_polygons, free_region):
        self.control_points = control_points
        self.centerline = centerline
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.boundaries_swapped = boundaries_swapped
        self.obstacle_polygons = obstacle_polygons
        self.free_region = free_region
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw satellite image
        painter.drawPixmap(0, 0, self.sat_pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))
        
        # Draw obstacles
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        for poly in self.obstacle_polygons:
            qpoly = QPolygonF([QPointF(p[0], p[1]) for p in poly])
            painter.drawPolygon(qpoly)
        
        # Draw safe backoff region
        try:
            backoff_val = float(self.parent.backoff_entry.text())
        except ValueError:
            backoff_val = self.parent.min_boundary_backoff
            
        backoff_px = backoff_val * self.parent.px_per_m
        if self.free_region is not None:
            safe_region = self.free_region.buffer(-backoff_px)
            if safe_region and not safe_region.is_empty and safe_region.exterior is not None:
                coords = list(safe_region.exterior.coords)
                if len(coords) >= 4:
                    qpoly = QPolygonF([QPointF(p[0], p[1]) for p in coords])
                    pen = QPen(QColor(255, 0, 255), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawPolygon(qpoly)
        
        # Draw control points
        for i, pt in enumerate(self.control_points):
            if i == 0 and len(self.control_points) > 2:
                painter.setBrush(QColor(255, 0, 0))  # Red for first point
            elif i == len(self.control_points) - 1 and len(self.control_points) > 2:
                painter.setBrush(QColor(0, 0, 255))  # Blue for last point
            else:
                painter.setBrush(QColor(0, 255, 0))  # Green for others
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(pt, 5, 5)
        
        # Draw centerline
        if self.centerline and len(self.centerline) > 1:
            pen = QPen(QColor(0, 255, 0), 2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            for i in range(1, len(self.centerline)):
                painter.drawLine(self.centerline[i-1], self.centerline[i])
        
        # Draw boundaries
        if self.left_boundary and len(self.left_boundary) > 1:
            left_color = QColor(255, 255, 0) if self.boundaries_swapped else QColor(0, 0, 255)
            pen = QPen(left_color, 2)
            painter.setPen(pen)
            for i in range(1, len(self.left_boundary)):
                painter.drawLine(self.left_boundary[i-1], self.left_boundary[i])
        
        if self.right_boundary and len(self.right_boundary) > 1:
            right_color = QColor(0, 0, 255) if self.boundaries_swapped else QColor(255, 255, 0)
            pen = QPen(right_color, 2)
            painter.setPen(pen)
            for i in range(1, len(self.right_boundary)):
                painter.drawLine(self.right_boundary[i-1], self.right_boundary[i])
        
        # Draw cones
        try:
            cone_spacing = float(self.parent.cone_spacing_entry.text())
        except ValueError:
            cone_spacing = self.parent.default_cone_distance
            
        if self.left_boundary and len(self.left_boundary) > 1:
            left_boundary = np.array([[p.x(), p.y()] for p in self.left_boundary])
            left_cones = sample_cones(left_boundary, cone_spacing, self.parent.px_per_m)
            
            left_color = QColor(255, 255, 0) if self.boundaries_swapped else QColor(0, 0, 255)
            painter.setBrush(left_color)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for pt in left_cones:
                painter.drawEllipse(QPointF(pt[0], pt[1]), 3, 3)
                
        if self.right_boundary and len(self.right_boundary) > 1:
            right_boundary = np.array([[p.x(), p.y()] for p in self.right_boundary])
            right_cones = sample_cones(right_boundary, cone_spacing, self.parent.px_per_m)
            
            right_color = QColor(0, 0, 255) if self.boundaries_swapped else QColor(255, 255, 0)
            painter.setBrush(right_color)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for pt in right_cones:
                painter.drawEllipse(QPointF(pt[0], pt[1]), 3, 3)
        
    def mousePressEvent(self, event):
        self.parent.handle_canvas_click(event.pos())
        
    def mouseMoveEvent(self, event):
        if self.parent.dragging:
            self.parent.handle_canvas_drag(event.pos())
            
    def mouseReleaseEvent(self, event):
        self.parent.handle_canvas_release(event.pos())