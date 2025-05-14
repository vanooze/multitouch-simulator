# test.py

import cv2
import numpy as np
from mediapipe_module import HandTracker
from astra_depth import AstraDepthReader

# --- Smoothing Helpers ---

depth_buffer = []

def get_median_depth(depth_map, x, y, size=5):
    h, w = depth_map.shape
    x1 = max(0, x - size // 2)
    y1 = max(0, y - size // 2)
    x2 = min(w, x + size // 2 + 1)
    y2 = min(h, y + size // 2 + 1)
    patch = depth_map[y1:y2, x1:x2]
    patch = patch[patch > 0]  # Filter invalid (zero) values
    if patch.size == 0:
        return 0
    return int(np.median(patch))

def smooth_depth(new_val, buffer, max_size=5):
    if new_val > 0:
        buffer.append(new_val)
    if len(buffer) > max_size:
        buffer.pop(0)
    return int(np.mean(buffer)) if buffer else 0

# --- Main Program ---

def main():
    tracker = HandTracker()
    depth_reader = AstraDepthReader()

    print("Running Astra color + depth + smoothed hand tracking...")

    while True:
        # Get color and depth frames
        color_frame = depth_reader.get_color_frame()
        depth_map = depth_reader.get_depth_frame()

        # Track hands
        tracked_frame = tracker.find_hands(color_frame.copy())
        hand_landmarks = tracker.get_landmarks(tracked_frame.shape)

        for landmarks in hand_landmarks:
            index_tip = landmarks[8]  # Index fingertip (x, y)
            x, y = index_tip

            if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                raw_depth = get_median_depth(depth_map, x, y, size=5)
                smoothed_depth = smooth_depth(raw_depth, depth_buffer)

                # Draw marker + depth text
                cv2.circle(tracked_frame, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(tracked_frame, f"{smoothed_depth} mm", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert depth to color image
        depth_8u = cv2.convertScaleAbs(depth_map, alpha=0.03)
        depth_colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        # Resize if needed
        if depth_colored.shape[:2] != tracked_frame.shape[:2]:
            depth_colored = cv2.resize(depth_colored, (tracked_frame.shape[1], tracked_frame.shape[0]))

        # Show both views
        cv2.imshow("Astra Color + Smoothed Depth", tracked_frame)
        cv2.imshow("Astra Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    depth_reader.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
