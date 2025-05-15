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
        self.touched = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.virtual_planes = []  # List of dicts for each calibrated face plane data
        self.merged_planes = []   # Merged polygons and info for proportional mapping
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

        # STEP 1: Get both hand and pose frames/landmarks
        hands_frame = self.tracker.find_hands(color_frame.copy())
        hands_landmarks = self.tracker.get_hand_landmarks(hands_frame.shape)

        pose_frame = self.tracker.find_pose(color_frame.copy())
        pose_landmarks = self.tracker.get_pose_landmarks(pose_frame.shape)

        use_hands = False
        active_landmarks = []
        selected_frame = frame
        current_depth = None

        # STEP 2: Use pose depth silently for current_depth stability
        if pose_landmarks:
            px, py = pose_landmarks[0][19]  # pose index fingertip
            pose_depth = self.depth_reader.get_smoothed_depth(depth_map, int(px), int(py))
            if pose_depth:
                current_depth = pose_depth
        elif hands_landmarks:
            hx, hy = hands_landmarks[0][8]  # hand index fingertip fallback
            hand_depth = self.depth_reader.get_smoothed_depth(depth_map, int(hx), int(hy))
            if hand_depth:
                current_depth = hand_depth

        # STEP 3: Apply hysteresis for smooth mode switching
        if current_depth is not None:
            upper_limit = config.DEPTH_HAND_THRESHOLD + config.HYSTERESIS_MARGIN
            lower_limit = config.DEPTH_HAND_THRESHOLD - config.HYSTERESIS_MARGIN

            if getattr(self, "last_mode", None) == "hands":
                use_hands = current_depth < upper_limit
            elif getattr(self, "last_mode", None) == "pose":
                use_hands = current_depth < lower_limit
            else:
                use_hands = current_depth < config.DEPTH_HAND_THRESHOLD

        # STEP 4: Select active landmarks and frame for display
        if use_hands and hands_landmarks:
            active_landmarks = hands_landmarks
            selected_frame = hands_frame
            self.last_mode = "hands"
        elif pose_landmarks:
            active_landmarks = pose_landmarks
            selected_frame = pose_frame
            self.last_mode = "pose"
        else:
            self.last_mode = None

        # STEP 5: Draw face polygons
        for plane in self.merged_planes:
            cv2.polylines(selected_frame, [plane["points"]], isClosed=True, color=(0, 255, 255), thickness=2)

        # STEP 6: Interaction logic with depth display
        if active_landmarks and self.merged_planes:
            for landmarks in active_landmarks:
                pointer = landmarks[8] if use_hands else landmarks[20]
                x, y = int(pointer[0]), int(pointer[1])

                if not (0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]):
                    continue

                depth_mm = self.depth_reader.get_smoothed_depth(depth_map, x, y)
                if depth_mm is None:
                    continue

                over_face_idx, virtual_depth = None, None
                for i, plane in enumerate(self.merged_planes):
                    if plane["polygon"].contains(Point(x, y)):
                        virtual_depth = self.interpolate_depth(x, y, plane)
                        over_face_idx = i
                        break

                if virtual_depth is None:
                    color = (200, 200, 200)
                else:
                    color = (0, 0, 255)
                    # Show pointer depth
                    cv2.putText(selected_frame, f"Depth: {depth_mm} mm", (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # Show virtual plane depth (distance to virtual surface)
                    cv2.putText(selected_frame, f"Plane: {virtual_depth:.1f} mm", (x + 10, y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 2)

                    # Compute proportional screen coordinates (unchanged)
                    total_area = sum(p["area"] for p in self.merged_planes)
                    cumulative_width = 0
                    for idx, plane in enumerate(self.merged_planes):
                        slice_width = (plane["area"] / total_area) * self.screen_w
                        if idx == over_face_idx:
                            pts = plane["points"]
                            min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
                            min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
                            norm_x = np.interp(x, [min_x, max_x], [0, 1])
                            norm_y = np.interp(y, [min_y, max_y], [0, 1])

                            screen_x = int(cumulative_width + norm_x * slice_width)
                            screen_y = int(norm_y * self.screen_h)
                            break
                        cumulative_width += slice_width

                    # Hover/hold logic (unchanged)
                    if depth_mm > virtual_depth - config.HOVER_DEPTH_THRESHOLD:
                        color = (0, 255, 0)
                        pyautogui.moveTo(screen_x, screen_y)
                        if self.touched:
                            pyautogui.mouseUp()
                            self.touched = False
                    elif depth_mm < virtual_depth - config.TOUCH_DEPTH_THRESHOLD:
                        color = (0, 0, 255)
                        if not self.touched:
                            pyautogui.mouseDown()
                            self.touched = True

                cv2.circle(selected_frame, (x, y), 5, color, cv2.FILLED)

        return selected_frame

    def shutdown(self):
        self.depth_reader.shutdown()
