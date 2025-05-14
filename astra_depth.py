import cv2
from openni import openni2
import numpy as np

class AstraDepthReader:
    def __init__(self):
        openni2.initialize()
        self.dev = openni2.Device.open_any()

        # Depth stream
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

        # Color stream
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.start()

        # Resolution
        vm = self.depth_stream.get_video_mode()
        self.resolution = (vm.resolutionX, vm.resolutionY)

        # Calibration data
        self.virtual_plane_set = False
        self.corner_points = []        # [(x, y), ...]
        self.corner_depths = []        # [z1, z2, z3, z4]
        self.depth_buffer = []

    def get_depth_frame(self):
        frame = self.depth_stream.read_frame()
        data = frame.get_buffer_as_uint16()
        return np.frombuffer(data, dtype=np.uint16).reshape(self.resolution[1], self.resolution[0])

    def get_color_frame(self):
        frame = self.color_stream.read_frame()
        data = frame.get_buffer_as_uint8()
        rgb = np.frombuffer(data, dtype=np.uint8).reshape((self.resolution[1], self.resolution[0], 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def shutdown(self):
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()

    def capture_virtual_screen_depth(self, corners):
        if len(corners) != 4:
            raise ValueError("Please provide exactly 4 corners for calibration.")

        self.corner_points = corners
        self.corner_depths = []

        depth_map = self.get_depth_frame()

        for (x, y) in corners:
            z = self._get_median_depth(depth_map, x, y, size=5)
            self.corner_depths.append(z)

        if len(self.corner_depths) == 4 and all(z > 0 for z in self.corner_depths):
            self.virtual_plane_set = True
        else:
            raise ValueError("Depth capture failed: Ensure all corner depths are valid.")

    def _get_median_depth(self, depth_map, x, y, size=5):
        h, w = depth_map.shape
        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(w, x + size // 2 + 1), min(h, y + size // 2 + 1)
        patch = depth_map[y1:y2, x1:x2]
        valid = patch[patch > 0]
        return int(np.median(valid)) if valid.size > 0 else 0

    def get_smoothed_depth(self, depth_map, x, y):
        z = self._get_median_depth(depth_map, x, y)
        if z > 0:
            self.depth_buffer.append(z)
        if len(self.depth_buffer) > 5:
            self.depth_buffer.pop(0)
        return int(np.mean(self.depth_buffer)) if self.depth_buffer else 0

    def get_interpolated_virtual_depth(self, x, y):
        if not self.virtual_plane_set or len(self.corner_points) != 4:
            return None

        # Map to virtual space: assume rectangle and use bilinear interpolation
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.corner_points
        z1, z2, z3, z4 = self.corner_depths

        # Convert input (x, y) to relative position (u, v) within quad
        rect = np.array(self.corner_points, dtype=np.float32)
        u, v = self._get_relative_uv((x, y), rect)

        # Bilinear interpolation
        z_interp = (1 - u) * (1 - v) * z1 + u * (1 - v) * z2 + u * v * z3 + (1 - u) * v * z4
        return int(z_interp)

    def _get_relative_uv(self, point, rect):
        # Assuming rect is in order: TL, TR, BR, BL
        tl, tr, br, bl = rect

        # Estimate horizontal ratio u
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        center_top = tl + (tr - tl) * 0.5
        center_bottom = bl + (br - bl) * 0.5
        height = np.linalg.norm(center_bottom - center_top)

        # Project to edges to estimate u and v
        def project(p, a, b):
            ab = b - a
            ap = p - a
            return np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)

        u = project(point, tl, tr)
        v = project(point, tl, bl)

        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        return u, v
