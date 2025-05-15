import cv2
import traceback
import config
from ui_handler import CalibrationUI
from logic_handler import LogicHandler

def main():
    calibration_ui = CalibrationUI()
    logic = LogicHandler(calibration_ui)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    cap.set(3, config.FRAME_WIDTH)
    cap.set(4, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Ensure RGB conversion

    cv2.namedWindow("Virtual Touchscreen")
    cv2.setMouseCallback("Virtual Touchscreen", calibration_ui.handle_mouse_click)

    while True:
        webcam_frame = logic.depth_reader.get_color_frame()
        if webcam_frame is None:
            print("Failed to read from Astra color stream.")
            continue
        try:
            if calibration_ui.calibrated:
                # Process logic with depth etc
                frame = logic.process_frame(webcam_frame)
            else:
                # Ensure color is correct (just in case webcam outputs YUV/other)
                frame = webcam_frame.copy()
                calibration_ui.draw_ui(frame)

            cv2.imshow("Virtual Touchscreen", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('r'):
                calibration_ui.reset()
            elif key == ord('s'):
                if calibration_ui.is_ready():
                    calibration_ui.calibrated = True
                    logic.calculate_plane_depths()
                    print(f"Calibration finished with {len(calibration_ui.get_faces())} faces.")
                else:
                    print("Define at least one face before starting.")

        except Exception as e:
            print("Error in main loop:", e)
            traceback.print_exc()

    cap.release()
    logic.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
