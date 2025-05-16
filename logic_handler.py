import numpy as np
import pyautogui
import cv2
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from mediapipe_module import BodyTracker
from astra_depth import AstraDepthReader
import config

class LogicHandler:
    def __init__(self, calibration_ui):
        self.calibration_ui = calibration_ui
        self.tracker = BodyTracker()
        self.depth_reader = AstraDepthReader()
        self.screen_w, self.screen_h = pyautogui.size()
        self.virtual_planes = []  # List of dicts for each calibrated face plane data
        self.merged_planes = []   # Merged polygons and info for proportional mapping
        self.prev_touch_states = {}
        self.last_mode = None

    def calculate_plane_depths(self):
        self.virtual_planes.clear()
        faces = self.calibration_ui.get_faces()

        # Prepare planes with depth info
        for face_points in faces:
            depths = []
            for (x, y) in face_points:
                depth = self.depth_reader.get_smoothed_depth(self.depth_reader.get_depth_frame(), x, y)
                if depth is None:
                    depth = 1000
                depths.append(depth)

            plane = {
                "points": np.array(face_points, dtype=np.int32),
                "depths": depths
            }
            self.virtual_planes.append(plane)

        # After collecting planes, merge overlapping faces
        self.merge_overlapping_faces()

    def merge_overlapping_faces(self):
        # Convert virtual_planes polygons to shapely Polygons
        shapely_polys = [Polygon(plane["points"]) for plane in self.virtual_planes]

        # Merge all overlapping polygons into continuous surfaces
        merged = unary_union(shapely_polys)

        # unary_union can return a Polygon or MultiPolygon
        # Normalize to list of polygons for consistent handling
        if isinstance(merged, Polygon):
            merged_polygons = [merged]
        else:
            merged_polygons = list(merged.geoms)

        # Clear previous merged_planes info
        self.merged_planes.clear()

        # For each merged polygon, approximate points and assign average depths
        for poly in merged_polygons:
            # Approximate polygon points as integer numpy array
            pts = np.array(poly.exterior.coords[:-1], dtype=np.int32)

            # For depths, approximate by sampling original planes inside this polygon
            # We'll average depths from the nearest points from the original planes corners

            depths = []
            for (x, y) in pts:
                # For each vertex, find closest depth from all original planes corners
                min_dist = float('inf')
                chosen_depth = 1000
                for plane in self.virtual_planes:
                    for (px, py), d in zip(plane["points"], plane["depths"]):
                        dist = (px - x) ** 2 + (py - y) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            chosen_depth = d
                depths.append(chosen_depth)

            self.merged_planes.append({
                "points": pts,
                "depths": depths,
                "polygon": poly,
                "area": poly.area
            })

        # Sort merged_planes left to right by centroid.x (for consistent horizontal slicing)
        self.merged_planes.sort(key=lambda p: p["polygon"].centroid.x)

    def point_in_any_merged_face(self, x, y):
        point = (x, y)
        for i, plane in enumerate(self.merged_planes):
            # Use shapely polygon contains method
            if plane["polygon"].contains(Point(point)):
                return i
        return None

    def interpolate_depth(self, x, y, plane):
        pts = plane["points"]
        depths = plane["depths"]

        min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return None

        nx = (x - min_x) / (max_x - min_x) if (max_x - min_x) > 0 else 0
        ny = (y - min_y) / (max_y - min_y) if (max_y - min_y) > 0 else 0

        depth = (
            depths[0] * (1 - nx) * (1 - ny) +
            depths[1] * nx * (1 - ny) +
            depths[2] * nx * ny +
            depths[3] * (1 - nx) * ny
        )
        return depth

    def process_frame(self, webcam_frame):
        frame = webcam_frame.copy()
        color_frame = self.depth_reader.get_color_frame()
        depth_map = self.depth_reader.get_depth_frame()

        if color_frame is None or depth_map is None:
            return frame

        pose_frame = self.tracker.find_pose(color_frame.copy())
        pose_landmarks = self.tracker.get_pose_landmarks(pose_frame.shape)
        selected_frame = pose_frame
        pointer_indices = [19, 20, 31, 32]  # Left & right hand, left & right foot

        # Draw virtual planes
        for plane in self.merged_planes:
            cv2.polylines(selected_frame, [plane["points"]], isClosed=True, color=(0, 255, 255), thickness=2)

        if pose_landmarks and self.merged_planes:
            for landmarks in pose_landmarks:
                for idx in pointer_indices:
                    if idx >= len(landmarks):
                        continue
                    px, py = int(landmarks[idx][0]), int(landmarks[idx][1])
                    if not (0 <= px < depth_map.shape[1] and 0 <= py < depth_map.shape[0]):
                        continue

                    depth_mm = self.depth_reader.get_smoothed_depth(depth_map, px, py)
                    if depth_mm is None:
                        continue

                    over_face_idx, virtual_depth = None, None
                    for i, plane in enumerate(self.merged_planes):
                        if plane["polygon"].contains(Point(px, py)):
                            virtual_depth = self.interpolate_depth(px, py, plane)
                            over_face_idx = i
                            break

                    color = (200, 200, 200)
                    if virtual_depth is not None:
                        # Show depth text
                        cv2.putText(selected_frame, f"Depth: {depth_mm} mm", (px + 10, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(selected_frame, f"Plane: {virtual_depth:.1f} mm", (px + 10, py - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 2)

                        # Map to screen coordinates
                        total_area = sum(p["area"] for p in self.merged_planes)
                        cumulative_width = 0
                        for i, plane in enumerate(self.merged_planes):
                            slice_width = (plane["area"] / total_area) * self.screen_w
                            if i == over_face_idx:
                                pts = plane["points"]
                                min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
                                min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
                                norm_x = np.interp(px, [min_x, max_x], [0, 1])
                                norm_y = np.interp(py, [min_y, max_y], [0, 1])
                                screen_x = int(cumulative_width + norm_x * slice_width)
                                screen_y = int(norm_y * self.screen_h)
                                break
                            cumulative_width += slice_width

                        # Hover / Touch with click-once logic
                        hover_thresh = virtual_depth - config.HOVER_DEPTH_THRESHOLD
                        touch_thresh = virtual_depth - config.TOUCH_DEPTH_THRESHOLD

                        landmark_id = f"{idx}_{over_face_idx}"  # Unique per landmark+plane

                        if depth_mm > hover_thresh:
                            color = (0, 255, 0)  # Hover
                            pyautogui.moveTo(screen_x, screen_y)
                            self.prev_touch_states[landmark_id] = False  # Reset
                        elif depth_mm < touch_thresh:
                            color = (0, 0, 255)  # Touch
                            pyautogui.moveTo(screen_x, screen_y)

                            if not self.prev_touch_states.get(landmark_id, False):
                                pyautogui.click()
                                self.prev_touch_states[landmark_id] = True
                        else:
                            self.prev_touch_states[landmark_id] = False  # Reset if in between

                    cv2.circle(selected_frame, (px, py), 5, color, cv2.FILLED)

        return selected_frame

    def shutdown(self):
        self.depth_reader.shutdown()
