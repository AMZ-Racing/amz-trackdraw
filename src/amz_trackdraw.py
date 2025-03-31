import os
import math
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from shapely.geometry import Polygon
from utils import create_closed_spline, generate_offset_boundaries, sample_cones
import cv2
import yaml
from scipy.interpolate import splprep, splev

# ---------------------------- GUI CLASS ---------------------------- #
class AMZTrackDraw(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)

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

        # Load and display the satellite image.
        self.image = Image.open(self.fpath_location_sat_img)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(self, width=self.photo.width(), height=self.photo.height(), bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Load occupancy map and extract obstacles.
        occ_img = cv2.imread(self.fpath_location_occup_img, cv2.IMREAD_GRAYSCALE)
        _, self.binary_map = cv2.threshold(occ_img, 128, 255, cv2.THRESH_BINARY)
        self.obstacle_polygons = self.extract_obstacle_polygons(self.binary_map)
        self.draw_obstacles()
        
        # Extract free region for backoff visualization.
        self.free_region = self.extract_free_region(self.binary_map)
        
        # Data for control points and track.
        self.control_points = []  # List of (x,y)
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries_swapped = False  # False: left=blue, right=yellow
        
        # GUI control variables.
        self.mode = "add"  # Modes: "add", "remove", "move"
        self.selected_point_index = None
        self.dragging = False
        
        # Right-side UI
        self.ui_frame = tk.Frame(self)
        self.ui_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mode label
        self.mode_label = tk.Label(self.ui_frame, text="Mode: Add", font=("Arial", 12))
        self.mode_label.pack(pady=5)
        
        self.add_button = tk.Button(self.ui_frame, text="Add Control Point", command=self.activate_add_mode)
        self.add_button.pack(pady=5)
        self.remove_button = tk.Button(self.ui_frame, text="Remove Control Point", command=self.activate_remove_mode)
        self.remove_button.pack(pady=5)
        self.move_button = tk.Button(self.ui_frame, text="Move Control Point", command=self.activate_move_mode)
        self.move_button.pack(pady=5)
        self.swap_button = tk.Button(self.ui_frame, text="Swap Boundaries", command=self.swap_boundaries)
        self.swap_button.pack(pady=5)
        self.export_button = tk.Button(self.ui_frame, text="Export CSV", command=self.export_csv)
        self.export_button.pack(pady=5)
        
        tk.Label(self.ui_frame, text="Cone spacing (m):").pack(pady=5)
        self.cone_spacing_entry = tk.Entry(self.ui_frame)
        self.cone_spacing_entry.insert(0, str(self.default_cone_distance))
        self.cone_spacing_entry.pack(pady=5)
        self.cone_spacing_entry.bind('<Return>', lambda event: self.redraw())
        
        tk.Label(self.ui_frame, text="Backoff (m):").pack(pady=5)
        self.backoff_entry = tk.Entry(self.ui_frame)
        self.backoff_entry.insert(0, str(self.min_boundary_backoff))
        self.backoff_entry.pack(pady=5)
        self.backoff_entry.bind('<Return>', lambda event: self.redraw())

        tk.Label(self.ui_frame, text="Track width (m):").pack(pady=5)
        self.track_width_entry = tk.Entry(self.ui_frame)
        self.track_width_entry.insert(0, str(self.track_width))
        self.track_width_entry.pack(pady=5)
        self.track_width_entry.bind('<Return>', lambda event: self.redraw())
        
        # Bind mouse events on the canvas.
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Storage for canvas drawn items.
        self.point_items = []
        self.centerline_item = None
        self.left_boundary_item = None
        self.right_boundary_item = None
        self.cone_items = []
        self.backoff_item = None  # For drawing safe backoff region

        # Adding labels for the statistics
        self.track_length_label = tk.Label(self.ui_frame, text="Track Length: --", font=("Arial", 12))
        self.track_length_label.pack(pady=5)
        self.min_radius_label = tk.Label(self.ui_frame, text="Min Radius: --", font=("Arial", 12))
        self.min_radius_label.pack(pady=5)
        self.cone_count_label = tk.Label(self.ui_frame, text="Cones: Blue: -- Yellow: -- Total: --", font=("Arial", 12))
        self.cone_count_label.pack(pady=5)
        
        # Load and create a PhotoImage for the logo
        logo_path = "logo.png"  # Update with the actual path
        logo_image = Image.open(logo_path)
        # Get the original dimensions
        original_width, original_height = logo_image.size
        # Define a scaling factor (e.g., 0.1 for 10% of the original size)
        scale_factor = 0.1
        # Calculate the new dimensions while maintaining the aspect ratio
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        # Resize the image
        logo_image = logo_image.resize((new_width, new_height))
        self.logo_photo = ImageTk.PhotoImage(logo_image)
        # Create a Label to hold the logo
        self.logo_label = tk.Label(self.master, image=self.logo_photo, borderwidth=0, highlightthickness=0)
        # Place the logo at the bottom-right corner of the main window
        padding = 10  # The desired padding in pixels
        self.logo_label.place(
            relx=1.0, rely=1.0,
            x=-padding,  # negative offset pushes the label left from the right edge
            y=-padding,  # negative offset pushes the label up from the bottom edge
            anchor='se'
        )
        
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
        
    def draw_obstacles(self):
        """Draw obstacles (in red) on the canvas."""
        for poly in self.obstacle_polygons:
            flat_coords = []
            for pt in poly:
                flat_coords.extend(pt)
            self.canvas.create_polygon(flat_coords, outline="red", fill="", width=2)
            
    def activate_add_mode(self):
        self.mode = "add"
        self.mode_label.config(text="Mode: Add")
    def activate_remove_mode(self):
        self.mode = "remove"
        self.mode_label.config(text="Mode: Remove")
    def activate_move_mode(self):
        self.mode = "move"
        self.mode_label.config(text="Mode: Move")
        
    def swap_boundaries(self):
        self.boundaries_swapped = not self.boundaries_swapped
        self.redraw()
        
    def on_canvas_click(self, event):
        x, y = event.x, event.y
        if self.mode == "add":
            self.control_points.append((x, y))
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
                
    def on_canvas_drag(self, event):
        if self.dragging and self.selected_point_index is not None and self.mode == "move":
            self.control_points[self.selected_point_index] = (event.x, event.y)
            self.redraw()
            
    def on_canvas_release(self, event):
        self.dragging = False
        self.selected_point_index = None
        
    def find_near_control_point(self, x, y, threshold=10):
        for i, pt in enumerate(self.control_points):
            if (pt[0] - x) ** 2 + (pt[1] - y) ** 2 < threshold ** 2:
                return i
        return None
        
    def export_csv(self):
        """
        Export the cone positions as CSV in meters with the following structure:
        tag,x,y
        where the (0,0) is defined at the first point of the centerline and the x-axis 
        is along the tangent of the centerline at that point.
        """
        try:
            cone_spacing = float(self.cone_spacing_entry.get())
        except ValueError:
            cone_spacing = self.default_cone_distance
        if self.centerline is None or self.left_boundary is None or self.right_boundary is None:
            messagebox.showerror("Export Error", "No track defined yet!")
            return
        left_cones = sample_cones(self.left_boundary, cone_spacing, self.px_per_m)
        right_cones = sample_cones(self.right_boundary, cone_spacing, self.px_per_m)
        # Define new coordinate system:
        origin = np.array(self.centerline[0])
        if len(self.centerline) < 2:
            messagebox.showerror("Export Error", "Not enough centerline points!")
            return
        tangent = np.array(self.centerline[1]) - origin
        theta = math.atan2(tangent[1], tangent[0])
        # Rotation matrix for -theta.
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
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filename:
            return
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["tag", "x", "y"])
            for pt in right_cones_m:
                writer.writerow([right_tag, pt[0], pt[1]])
            for pt in left_cones_m:
                writer.writerow([left_tag, pt[0], pt[1]])
        messagebox.showinfo("Export", f"Track exported to {filename}")
        
    def export_csv_data(self, track_data, filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["tag", "x", "y"])
            for data in track_data:
                writer.writerow(data)
        
    def redraw(self):
        # Clear canvas overlays.
        for item in self.point_items:
            self.canvas.delete(item)
        self.point_items = []
        if self.centerline_item:
            self.canvas.delete(self.centerline_item)
            self.centerline_item = None
        if self.left_boundary_item:
            self.canvas.delete(self.left_boundary_item)
            self.left_boundary_item = None
        if self.right_boundary_item:
            self.canvas.delete(self.right_boundary_item)
            self.right_boundary_item = None
        for item in self.cone_items:
            self.canvas.delete(item['item'])
        self.cone_items = []
        if self.backoff_item:
            self.canvas.delete(self.backoff_item)
            self.backoff_item = None
        
        # Draw control points.
        for pt in self.control_points:
            item = self.canvas.create_oval(pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5, fill="green")
            self.point_items.append(item)
        # redraw the first control point in red
        if len(self.control_points) > 2:
            item = self.canvas.create_oval(self.control_points[0][0]-5, self.control_points[0][1]-5,
                                           self.control_points[0][0]+5, self.control_points[0][1]+5, fill="red")
            self.point_items.append(item)

        # redraw the last control point in blue
        if len(self.control_points) > 2:
            item = self.canvas.create_oval(self.control_points[-1][0]-5, self.control_points[-1][1]-5,
                                           self.control_points[-1][0]+5, self.control_points[-1][1]+5, fill="blue")
            self.point_items.append(item)
        
        try:
            self.track_width = float(self.track_width_entry.get())
        except ValueError:
            pass  # Optionally, you can set a default value or alert the user

        # Draw safe backoff region.
        try:
            backoff_val = float(self.backoff_entry.get())
        except ValueError:
            backoff_val = self.min_boundary_backoff
        self.min_boundary_backoff = backoff_val
        backoff_px = self.min_boundary_backoff * self.px_per_m
        if self.free_region is not None:
            safe_region = self.free_region.buffer(-backoff_px)
            if safe_region and not safe_region.is_empty and safe_region.exterior is not None:
                coords = list(safe_region.exterior.coords)
                if len(coords) >= 4:
                    flat_coords = []
                    for pt in coords:
                        flat_coords.extend(pt)
                    self.backoff_item = self.canvas.create_line(*flat_coords, fill="magenta", dash=(2,2), width=2)
        
        # If at least 3 control points, compute and draw track.
        if len(self.control_points) > 3:
            self.centerline = create_closed_spline(self.control_points, 
                                                   num_points=self.n_points_midline)
            coords = []
            for pt in self.centerline:
                coords.extend(pt.tolist())
            self.centerline_item = self.canvas.create_line(*coords, fill="green", dash=(4,2), width=2)
            boundaries = generate_offset_boundaries(self.centerline, self.track_width, self.px_per_m)
            if boundaries[0] is None or boundaries[1] is None:
                return
            self.left_boundary, self.right_boundary = boundaries
            left_coords = []
            for pt in self.left_boundary:
                left_coords.extend(pt.tolist())
            right_coords = []
            for pt in self.right_boundary:
                right_coords.extend(pt.tolist())
            if self.boundaries_swapped:
                self.left_boundary_item = self.canvas.create_line(*left_coords, fill="yellow", width=2)
                self.right_boundary_item = self.canvas.create_line(*right_coords, fill="blue", width=2)
            else:
                self.left_boundary_item = self.canvas.create_line(*left_coords, fill="blue", width=2)
                self.right_boundary_item = self.canvas.create_line(*right_coords, fill="yellow", width=2)
            
            # Draw cones along boundaries.
            try:
                cone_spacing = float(self.cone_spacing_entry.get())
            except ValueError:
                cone_spacing = self.default_cone_distance
            left_cones = sample_cones(self.left_boundary, cone_spacing, self.px_per_m)
            right_cones = sample_cones(self.right_boundary, cone_spacing, self.px_per_m)
            if self.boundaries_swapped:
                for pt in left_cones:
                    item = self.canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3, fill="yellow")
                    self.cone_items.append({'item': item, 'color': 'yellow'})
                for pt in right_cones:
                    item = self.canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3, fill="blue")
                    self.cone_items.append({'item': item, 'color': 'blue'})
            else:
                for pt in left_cones:
                    item = self.canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3, fill="blue")
                    self.cone_items.append({'item': item, 'color': 'blue'})
                for pt in right_cones:
                    item = self.canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3, fill="yellow")
                    self.cone_items.append({'item': item, 'color': 'yellow'})

            # Calculate statistics
            track_length = self.calculate_track_length()
            min_radius = self.calculate_min_radius()
            blue_cones, yellow_cones, total_cones = self.count_cones()
        
            # Update the labels with calculated statistics
            self.track_length_label.config(text=f"Track Length: {track_length:.2f} m")
            self.min_radius_label.config(text=f"Min Radius: {min_radius:.2f} m")
            self.cone_count_label.config(text=f"Cones: Blue: {blue_cones} Yellow: {yellow_cones} Total: {total_cones}")
                    
    def find_near_control_point(self, x, y, threshold=10):
        for i, pt in enumerate(self.control_points):
            if (pt[0] - x) ** 2 + (pt[1] - y) ** 2 < threshold ** 2:
                return i
        return None


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
    

    def calculate_track_length(self):
        """Calculate the total length of the track (centerline B-spline)."""
        if len(self.centerline) < 2:
            return 0
        total_length = 0
        for i in range(1, len(self.centerline)):
            x1, y1 = self.centerline[i - 1]
            x2, y2 = self.centerline[i]
            total_length += math.hypot(x2 - x1, y2 - y1)
        return total_length / self.px_per_m  # Convert from pixels to meters
    

    def calculate_min_radius(self):
        """Calculate the minimum radius of curvature of the centerline B-spline."""
        if len(self.centerline) < 3:
            return float('inf')  # No valid curvature can be calculated
        
        # Extract x and y coordinates from the centerline
        x_coords, y_coords = zip(*self.centerline)  # This unpacks the list of tuples into x and y arrays
        
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
                continue  # Skip if denominator is zero (to avoid division by zero)
            
            curvature = numerator / denominator
            radius = 1 / curvature
            
            # Track the minimum radius
            min_radius = min(min_radius, radius)
        
        return min_radius / self.px_per_m  # Convert from pixels to meters

    
    def count_cones(self):
        """Count the number of blue, yellow, and total cones."""
        blue_cones = 0
        yellow_cones = 0
        total_cones = 0
        for pt in self.cone_items:
            if pt['color'] == 'blue':
                blue_cones += 1
            elif pt['color'] == 'yellow':
                yellow_cones += 1
            total_cones += 1
        return blue_cones, yellow_cones, total_cones