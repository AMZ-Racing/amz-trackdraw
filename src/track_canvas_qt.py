import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
import cv2
from shapely.geometry import Polygon
from utils_qt import sample_cones


class TrackCanvas(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Zoom/pan variables
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.pan_start = QPointF(0, 0)
        self.is_panning = False
        
        # Initialize scaling factors
        self.map_scale_x = 1.0
        self.map_scale_y = 1.0
        self.map_offset_x = 0
        self.map_offset_y = 0
        
        # Load the satellite image
        self.sat_image = QImage(parent.fpath_location_sat_img)
        self.sat_pixmap = QPixmap.fromImage(self.sat_image)
        
        # Load occupancy map and extract obstacles with proper scaling
        occ_img = cv2.imread(parent.fpath_location_occup_img, cv2.IMREAD_GRAYSCALE)
        _, self.binary_map = cv2.threshold(occ_img, 128, 255, cv2.THRESH_BINARY)
        self.obstacle_polygons = self.extract_obstacle_polygons(self.binary_map)
        self.free_region = self.extract_free_region(self.binary_map)
        
        # Store original obstacle polygons before scaling
        self.original_obstacle_polygons = self.extract_obstacle_polygons(self.binary_map)
        self.original_free_region = self.extract_free_region(self.binary_map)

        # Calculate scaling factors to match occupancy map with satellite image
        sat_width = self.sat_image.width()
        sat_height = self.sat_image.height()
        occ_height, occ_width = self.binary_map.shape
        self.map_scale_x = sat_width / occ_width
        self.map_scale_y = sat_height / occ_height
        
        # Drawing elements
        self.control_points = []
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries_swapped = False

    def wheelEvent(self, event):
        # Zoom factor (10% per wheel step)
        zoom_factor = 1.1
        if event.angleDelta().y() < 0:
            zoom_factor = 1 / zoom_factor
        
        # Get mouse position before zoom
        mouse_pos = event.pos()
        old_scene_pos = self.screen_to_map(mouse_pos)
        
        # Apply zoom
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(self.zoom, 10.0))  # Limit zoom range
        
        # Get mouse position after zoom
        new_screen_pos = self.map_to_screen(old_scene_pos)
        
        # Adjust pan to zoom toward mouse
        delta = mouse_pos - new_screen_pos
        self.pan += delta
        
        self.update()
        
    def transform_point(self, map_x, map_y):
        """Convert from map coordinates to display coordinates"""
        return (map_x * self.map_scale_x + self.map_offset_x,
                map_y * self.map_scale_y + self.map_offset_y)

    def inverse_transform_point(self, display_x, display_y):
        """Convert from display coordinates to map coordinates"""
        return ((display_x - self.map_offset_x) / self.map_scale_x,
                (display_y - self.map_offset_y) / self.map_scale_y)
    
    def transform_polygon(self, polygon):
        """Transform a polygon from occupancy to display coordinates"""
        return [self.transform_point(p[0], p[1]) for p in polygon]
    
    def screen_to_map(self, screen_point):
        """Convert screen coordinates to map coordinates"""
        # Remove pan and zoom, then remove image scaling
        image_x = (screen_point.x() - self.pan.x()) / self.zoom
        image_y = (screen_point.y() - self.pan.y()) / self.zoom
        map_x = (image_x - self.map_offset_x) / self.map_scale_x
        map_y = (image_y - self.map_offset_y) / self.map_scale_y
        return QPointF(map_x, map_y)

    def map_to_screen(self, map_point):
        """Convert map coordinates to screen coordinates"""
        # Apply image scaling, then zoom and pan
        image_x = map_point.x() * self.map_scale_x + self.map_offset_x
        image_y = map_point.y() * self.map_scale_y + self.map_offset_y
        screen_x = image_x * self.zoom + self.pan.x()
        screen_y = image_y * self.zoom + self.pan.y()
        return QPointF(int(screen_x), int(screen_y))
    
    def screen_to_scene(self, screen_point):
        """Convert screen coordinates to image coordinates (including zoom/pan)"""
        return QPointF(
            (screen_point.x() - self.pan.x()) / self.zoom,
            (screen_point.y() - self.pan.y()) / self.zoom
        )

    def scene_to_screen(self, scene_point):
        """Convert image coordinates to screen coordinates (including zoom/pan)"""
        return QPointF(
            scene_point.x() * self.zoom + self.pan.x(),
            scene_point.y() * self.zoom + self.pan.y()
        )
        
    def extract_obstacle_polygons(self, binary_map):
        """Extract obstacle contours from the occupancy map with proper scaling."""
        contours, _ = cv2.findContours(binary_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        if contours is not None:
            for cnt in contours:
                pts = cnt.squeeze()
                if pts.ndim == 1:
                    continue
                # Scale points to match satellite image
                scaled_pts = [self.transform_point(p[0], p[1]) for p in pts]
                polys.append(scaled_pts)
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
            # Scale points to match satellite image
            scaled_pts = [self.transform_point(p[0], p[1]) for p in pts]
            poly = Polygon(scaled_pts)
            if not poly.is_valid or poly.is_empty:
                return None
            return poly
        except Exception as e:
            print("Error creating free region polygon:", e)
            return None
            
    def update_drawing(self, control_points, centerline, left_boundary, right_boundary, 
                      boundaries_swapped):
        self.control_points = control_points
        self.centerline = centerline
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.boundaries_swapped = boundaries_swapped
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Apply zoom/pan transformation
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)

        # Draw scaled satellite image
        scaled_pixmap = self.sat_pixmap.scaled(
            self.width(), self.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(0, 0, scaled_pixmap)
        
        # Calculate scaling factors based on actual displayed image size
        self.map_scale_x = scaled_pixmap.width() / self.binary_map.shape[1]
        self.map_scale_y = scaled_pixmap.height() / self.binary_map.shape[0]
        
        # Draw obstacles with proper scaling
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        for poly in self.original_obstacle_polygons:
            scaled_poly = self.transform_polygon(poly)
            qpoly = QPolygonF([QPointF(p[0], p[1]) for p in scaled_poly])
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
                    # Transform each coordinate point to display space
                    scaled_coords = [self.transform_point(p[0], p[1]) for p in coords]
                    qpoly = QPolygonF([QPointF(p[0], p[1]) for p in scaled_coords])
                    pen = QPen(QColor(255, 0, 255), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawPolygon(qpoly)
        
        # Draw control points (in display coordinates)
        for i, pt in enumerate(self.control_points):
            display_pt = QPointF(*self.transform_point(pt.x(), pt.y()))
            if i == 0 and len(self.control_points) > 2:
                painter.setBrush(QColor(255, 0, 0))  # Red for first point
            elif i == len(self.control_points) - 1 and len(self.control_points) > 2:
                painter.setBrush(QColor(0, 0, 255))  # Blue for last point
            else:
                painter.setBrush(QColor(0, 255, 0))  # Green for others
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(display_pt, 5, 5)

        # Draw centerline (transform from map to display coords)
        if self.centerline and len(self.centerline) > 1:
            pen = QPen(QColor(0, 255, 0), 2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            
            # Convert all centerline points to display coordinates
            display_points = []
            for pt in self.centerline:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:  # Assume raw coordinates
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            
            # Draw connected lines
            for i in range(1, len(display_points)):
                painter.drawLine(display_points[i-1], display_points[i])

        # Draw boundaries (transform from map to display coords)
        if self.left_boundary and len(self.left_boundary) > 1:
            left_color = QColor(0, 0, 255) if not self.boundaries_swapped else QColor(255, 255, 0)
            painter.setPen(QPen(left_color, 2))
            
            # Convert boundary points to display coordinates
            display_points = []
            for pt in self.left_boundary:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:  # Assume raw coordinates
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            
            painter.drawPolyline(QPolygonF(display_points))

        if self.right_boundary and len(self.right_boundary) > 1:
            right_color = QColor(255, 255, 0) if not self.boundaries_swapped else QColor(0, 0, 255)
            painter.setPen(QPen(right_color, 2))
            
            # Convert boundary points to display coordinates
            display_points = []
            for pt in self.right_boundary:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:  # Assume raw coordinates
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            
            painter.drawPolyline(QPolygonF(display_points))

        # Draw cones (transform from map to display coords)
        try:
            cone_spacing = float(self.parent.cone_spacing_entry.text())
        except ValueError:
            cone_spacing = self.parent.default_cone_distance

        if self.left_boundary and len(self.left_boundary) > 1:
            # Get boundary points in map coordinates
            boundary_points = []
            for pt in self.left_boundary:
                if isinstance(pt, QPointF):
                    boundary_points.append([pt.x(), pt.y()])
                else:
                    boundary_points.append([pt[0], pt[1]])
            
            left_cones = sample_cones(np.array(boundary_points), cone_spacing, self.parent.px_per_m)
            
            left_color = QColor(255, 255, 0) if self.boundaries_swapped else QColor(0, 0, 255)
            painter.setBrush(left_color)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for pt in left_cones:
                display_pt = QPointF(*self.transform_point(pt[0], pt[1]))
                painter.drawEllipse(display_pt, 3, 3)
                
        if self.right_boundary and len(self.right_boundary) > 1:
            # Get boundary points in map coordinates
            boundary_points = []
            for pt in self.right_boundary:
                if isinstance(pt, QPointF):
                    boundary_points.append([pt.x(), pt.y()])
                else:
                    boundary_points.append([pt[0], pt[1]])
            
            right_cones = sample_cones(np.array(boundary_points), cone_spacing, self.parent.px_per_m)
            
            right_color = QColor(0, 0, 255) if self.boundaries_swapped else QColor(255, 255, 0)
            painter.setBrush(right_color)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for pt in right_cones:
                display_pt = QPointF(*self.transform_point(pt[0], pt[1]))
                painter.drawEllipse(display_pt, 3, 3)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            # Convert click to scene coordinates
            scene_pos = self.screen_to_scene(event.pos())
            # Then convert to map coordinates
            map_x, map_y = self.inverse_transform_point(scene_pos.x(), scene_pos.y())
            self.parent.handle_canvas_click(QPointF(map_x, map_y))

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.pan_start
            self.pan += delta
            self.pan_start = event.pos()
            self.update()
        elif self.parent.dragging:
            map_pos = self.screen_to_map(event.pos())
            self.parent.handle_canvas_drag(QPointF(map_pos.x(), map_pos.y()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            self.parent.handle_canvas_release(event.pos())