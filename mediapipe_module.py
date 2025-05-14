import cv2
import mediapipe as mp

class BodyTracker:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return image

    def get_landmarks(self, shape):
        landmarks_list = []
        if self.results.pose_landmarks:
            h, w, _ = shape
            landmarks = []
            for lm in self.results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))
            landmarks_list.append(landmarks)
        return landmarks_list
