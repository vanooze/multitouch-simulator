import cv2
import mediapipe as mp

class BodyTracker:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
        self.pose_results = None

    def find_pose(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.pose_results = self.pose.process(rgb)
        if self.pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                image,
                self.pose_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        return image

    def get_pose_landmarks(self, shape):
        if self.pose_results and self.pose_results.pose_landmarks:
            h, w, _ = shape
            return [[(int(lm.x * w), int(lm.y * h)) for lm in self.pose_results.pose_landmarks.landmark]]
        return []
