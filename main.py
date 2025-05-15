import cv2
import pyautogui
import numpy as np
import traceback
from mediapipe_module import BodyTracker
from astra_depth import AstraDepthReader
import config

# --- Globals ---
touched = False
corners = []
calibration_complete = False

def on_mouse_click(event, x, y, flags, param):
    global corners, calibration_complete
    if event == cv2.EVENT_LBUTTONDOWN and not calibration_complete:
        if len(corners) < 4:
            corners.append((x, y))
        if len(corners) == 4:
            calibration_complete = True

def main():
    global corners, calibration_complete, touched

    try:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam.")

        cap.set(3, config.FRAME_WIDTH)
        cap.set(4, config.FRAME_HEIGHT)

        screen_w, screen_h = pyautogui.size()
        tracker = BodyTracker()
        depth_reader = AstraDepthReader()

        cv2.namedWindow("Virtual Touchscreen")
        cv2.setMouseCallback("Virtual Touchscreen", on_mouse_click)

        while True:
            success, webcam_frame = cap.read()
            if not success or webcam_frame is None:
                print("Error: Failed to read frame from webcam.")
                continue

            try:
                color_frame = depth_reader.get_color_frame()
                depth_map = depth_reader.get_depth_frame()

                if color_frame is None or depth_map is None:
                    print("Warning: Failed to get color or depth frame. Skipping frame.")
                    continue

                frame = tracker.find_pose(color_frame.copy())
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

                if calibration_complete and len(corners) == 4:
                    # Draw calibrated area
                    cv2.polylines(frame, [np.array(corners, dtype=np.int32)], isClosed=True,
                                  color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, f"Touch Threshold Active ({config.TOUCH_DEPTH_THRESHOLD}mm)",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    min_x = min(pt[0] for pt in corners)
                    max_x = max(pt[0] for pt in corners)
                    min_y = min(pt[1] for pt in corners)
                    max_y = max(pt[1] for pt in corners)

                    virtual_width = max_x - min_x
                    virtual_height = max_y - min_y

                    resized_frame = cv2.resize(webcam_frame, (virtual_width, virtual_height))

                    if all_landmarks:
                        for landmarks in all_landmarks:
                            if len(landmarks) <= 20:
                                print("Warning: Landmark 20 not found.")
                                continue

                            index_tip = landmarks[20]
                            x, y = index_tip
                            if not (isinstance(x, int) or isinstance(x, float)) or not (isinstance(y, int) or isinstance(y, float)):
                                print(f"Invalid landmark coordinates: {index_tip}")
                                continue

                            x, y = int(x), int(y)
                            color = (0, 0, 255)

                            if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                                depth_mm = depth_reader.get_smoothed_depth(depth_map, x, y)
                                virtual_depth = depth_reader.get_interpolated_virtual_depth(x, y)

                                if depth_mm is None or virtual_depth is None:
                                    print("Warning: Invalid depth values.")
                                    continue

                                cv2.putText(frame, f"Hand: {depth_mm} mm", (x + 10, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
                                cv2.putText(frame, f"Plane: {virtual_depth:.1f} mm", (x + 10, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 2)

                                inside_area = cv2.pointPolygonTest(np.array(corners, dtype=np.int32), (x, y), False) >= 0
                                if inside_area:
                                    norm_x = np.interp(x, [min_x, max_x], [0, 1])
                                    norm_y = np.interp(y, [min_y, max_y], [0, 1])
                                    screen_x = int(norm_x * screen_w)
                                    screen_y = int(norm_y * screen_h)

                                    if depth_mm > virtual_depth - config.HOVER_DEPTH_THRESHOLD:
                                        color = (0, 255, 0)
                                        pyautogui.moveTo(screen_x, screen_y)
                                        touched = False
                                    elif depth_mm < virtual_depth - config.TOUCH_DEPTH_THRESHOLD:
                                        color = (0, 0, 255)
                                        if not touched:
                                            pyautogui.click(screen_x, screen_y)
                                            touched = True

                            cv2.circle(frame, (x, y), 5, color, cv2.FILLED)

                cv2.imshow("Virtual Touchscreen", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            except Exception as e:
                print("Error during frame processing:", e)
                traceback.print_exc()
                continue

    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()

    finally:
        cap.release()
        depth_reader.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
