import cv2
import numpy as np


class CalibrationUI:
    def __init__(self):
        self.faces = []  # List of faces, each face is a list of 4 points [(x,y,z), ...]
        self.current_face_points = []  # Points for the face currently being defined (in 2D)
        self.calibrated = False
        self.message = "Left-click 4 points to define a face. Right-click to save face."

    def handle_mouse_click(self, event, x, y, flags, param):
        if self.calibrated:
            return  # Ignore clicks after calibration is done

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_face_points) < 4:
                self.current_face_points.append((x, y))
            if len(self.current_face_points) == 4:
                self.message = "Right-click to confirm this face or add points again."

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_face_points) == 4:
                # Confirm current face
                self.faces.append(self.current_face_points.copy())
                self.current_face_points.clear()
                self.message = f"Face {len(self.faces)} saved! Left-click 4 points for next face or press 's' to start."
            else:
                self.message = "Need exactly 4 points to save a face."

    def draw_ui(self, frame):
        # Draw all confirmed faces
        for i, face in enumerate(self.faces):
            pts = np.array(face, np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            # Label face number at centroid
            cx = int(sum(p[0] for p in face) / 4)
            cy = int(sum(p[1] for p in face) / 4)
            cv2.putText(frame, f"Face {i + 1}", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw points of current face in progress
        for pt in self.current_face_points:
            cv2.circle(frame, pt, 6, (0, 255, 0), cv2.FILLED)
        # If less than 4 points, draw lines connecting them
        if len(self.current_face_points) > 1:
            pts = np.array(self.current_face_points, np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

        # Show instructions/message
        cv2.putText(frame, self.message, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def reset(self):
        self.faces.clear()
        self.current_face_points.clear()
        self.calibrated = False
        self.message = "Calibration reset. Left-click 4 points to define a face."

    def is_ready(self):
        return len(self.faces) > 0

    def get_faces(self):
        return self.faces
