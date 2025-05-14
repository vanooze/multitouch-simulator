import cv2
import pyautogui
import numpy as np
import time
from config import HOVER_DEPTH_THRESHOLD
from mediapipe_module import BodyTracker  # Import BodyTracker instead of HandTracker
from astra_depth import AstraDepthReader
import config

# --- Globals ---
touched = False
corners = []
calibration_complete = False
last_click_time = 0

def on_mouse_click(event, x, y, flags, param):
    global corners, calibration_complete
    if event == cv2.EVENT_LBUTTONDOWN and not calibration_complete:
        if len(corners) < 4:
            corners.append((x, y))
        if len(corners) == 4:
            calibration_complete = True

def main():
    global corners, calibration_complete, touched, last_click_time

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(3, config.FRAME_WIDTH)
    cap.set(4, config.FRAME_HEIGHT)

    screen_w, screen_h = pyautogui.size()
    tracker = BodyTracker()  # Initialize BodyTracker for full-body tracking
    depth_reader = AstraDepthReader()

    cv2.namedWindow("Virtual Touchscreen")
    cv2.setMouseCallback("Virtual Touchscreen", on_mouse_click)

    while True:
        success, webcam_frame = cap.read()
        if not success:
            break

        color_frame = depth_reader.get_color_frame()
        depth_map = depth_reader.get_depth_frame()

        frame = tracker.find_pose(color_frame.copy())  # Use find_pose for body tracking
        all_landmarks = tracker.get_landmarks(frame.shape)

        # Handle calibration
        if not calibration_complete:
            for pt in corners:
                cv2.circle(frame, pt, 5, (0, 255, 255), cv2.FILLED)
            cv2.putText(frame, f"Click {4 - len(corners)} more corners...",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif calibration_complete and not depth_reader.virtual_plane_set:
            depth_reader.capture_virtual_screen_depth(corners)
            print("Virtual plane calibrated.")

        # Only proceed with the calibration logic if the corners are available
        if calibration_complete and len(corners) == 4:
            # Draw calibrated area
            cv2.polylines(frame, [np.array(corners, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, f"Touch Threshold Active ({config.TOUCH_DEPTH_THRESHOLD}mm)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate the width and height of the calibrated virtual space
            min_x = min(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
            max_x = max(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
            min_y = min(corners[0][1], corners[1][1], corners[2][1], corners[3][1])
            max_y = max(corners[0][1], corners[1][1], corners[2][1], corners[3][1])

            virtual_width = max_x - min_x
            virtual_height = max_y - min_y

            # Resize the webcam frame to fit within the calibrated virtual space
            resized_frame = cv2.resize(webcam_frame, (virtual_width, virtual_height))

            # Process body landmarks
            if all_landmarks:
                for landmarks in all_landmarks:
                    # Use landmark 20 (right index) for finger pointer replacement
                    index_tip = landmarks[20]  # Landmark 20 corresponds to the right index finger
                    x, y = index_tip

                    # Default color for hovering (not touching)
                    color = (0, 0, 255)

                    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                        raw_depth = depth_reader._get_median_depth(depth_map, x, y)
                        depth_mm = depth_reader.get_smoothed_depth(depth_map, x, y)
                        virtual_depth = depth_reader.get_interpolated_virtual_depth(x, y)
                        diff = virtual_depth - depth_mm
                        cv2.putText(frame, f"Hand: {depth_mm} mm", (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
                        cv2.putText(frame, f"Plane: {virtual_depth if virtual_depth else 0} mm",
                                    (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 2)

                       # Check if the body part(Index) is inside the calibrated virtual space
                        if virtual_depth and 0 < diff < config.TOUCH_DEPTH_THRESHOLD:
                            if cv2.pointPolygonTest(np.array(corners, dtype=np.int32), (x, y), False) >= 0:
                                color = (0, 0, 255)  # Touching

                                # Normalize the finger's position to calibrated virtual space
                                norm_x = np.interp(x, [min_x, max_x], [0, 1])
                                norm_y = np.interp(y, [min_y, max_y], [0, 1])

                                # Map to the screen coordinates
                                screen_x = int(norm_x * screen_w)
                                screen_y = int(norm_y * screen_h)

                                pyautogui.mouseDown(screen_x, screen_y)
                                print("Touch (mouse down) at:", screen_x, screen_y)

                                touched = True
                        # Check if the body part (index) is inside the calibrated virtual space
                        elif virtual_depth and config.TOUCH_DEPTH_THRESHOLD <= diff < config.HOVER_DEPTH_THRESHOLD:

                            if cv2.pointPolygonTest(np.array(corners, dtype=np.int32), (x, y), False) >= 0:
                                if touched:
                                    pyautogui.mouseUp()
                                    color = (0, 255, 0)  # Hovering
                                    touched = False

                                    # Normalize the finger's position to calibrated virtual space
                                    norm_x = np.interp(x, [min_x, max_x], [0, 1])
                                    norm_y = np.interp(y, [min_y, max_y], [0, 1])

                                    # Now map the normalized coordinates to the screen
                                    screen_x = int(norm_x * screen_w)
                                    screen_y = int(norm_y * screen_h)

                                    # Move the mouse to the new position
                                    pyautogui.moveTo(screen_x, screen_y)
                                    print("Hovering at:", screen_x, screen_y)
                    cv2.circle(frame, (x, y), 10, color, cv2.FILLED)

        cv2.imshow("Virtual Touchscreen", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    depth_reader.shutdown()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()