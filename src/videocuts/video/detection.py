import os
import cv2
from typing import List, Dict, Optional

try:
    import mediapipe as mp
except ImportError:
    mp = None

def _mouth_open_metric(face_landmarks) -> float:
    """Measure mouth opening using multiple lip landmarks for accuracy."""
    # Primary: vertical distance between inner lip centers
    upper_center = face_landmarks.landmark[13]
    lower_center = face_landmarks.landmark[14]
    vertical_opening = abs(lower_center.y - upper_center.y)
    
    # Secondary: horizontal lip stretch
    left_corner = face_landmarks.landmark[78]
    right_corner = face_landmarks.landmark[308]
    horizontal_stretch = abs(right_corner.x - left_corner.x)
    
    BASELINE_LIP_WIDTH = 0.12
    stretch_delta = max(0, horizontal_stretch - BASELINE_LIP_WIDTH)
    
    return vertical_opening * 0.8 + stretch_delta * 0.2

def detect_faces_with_landmarker(landmarker, rgb_frame) -> List[Dict]:
    """Detect faces using MediaPipe FaceLandmarker."""
    detections = []
    if landmarker is None or mp is None:
        return detections

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    h, w, _ = rgb_frame.shape

    if not result.face_landmarks:
        return detections

    for face_landmarks in result.face_landmarks:
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        xmin = max(min(xs), 0.0)
        xmax = min(max(xs), 1.0)
        ymin = max(min(ys), 0.0)
        ymax = min(max(ys), 1.0)
        width = max(xmax - xmin, 1e-3)
        height = max(ymax - ymin, 1e-3)
        center = (xmin + xmax) / 2.0

        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        left_corner = face_landmarks[78]
        right_corner = face_landmarks[308]

        vertical_opening = abs(lower_lip.y - upper_lip.y)
        horizontal_stretch = abs(right_corner.x - left_corner.x)
        
        BASELINE_LIP_WIDTH = 0.12
        stretch_delta = max(0, horizontal_stretch - BASELINE_LIP_WIDTH)
        mouth_open = vertical_opening * 0.8 + stretch_delta * 0.2

        mouth_open_normalized = mouth_open / height if height > 0.01 else mouth_open

        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": min(width, 1.0),
            "mouth_open": mouth_open_normalized,
            "activity": 0.0
        })

    return detections

def detect_faces_with_tasks(detector, rgb_frame) -> List[Dict]:
    """Detect faces using MediaPipe FaceDetector."""
    detections = []
    if detector is None or mp is None:
        return detections

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    h, w, _ = rgb_frame.shape

    if not result.detections:
        return detections

    for det in result.detections:
        bbox = det.bounding_box
        center = (bbox.origin_x + bbox.width / 2.0) / max(w, 1)
        width = bbox.width / max(w, 1)
        score = det.categories[0].score if det.categories else 0.0
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": max(min(width, 1.0), 1e-3),
            "activity": score
        })

    return detections

def detect_faces_with_cascade(cascade, frame_bgr) -> List[Dict]:
    """Fallback face detection using Haar cascade."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    detections = []
    for (x, y, fw, fh) in faces:
        center = (x + fw / 2) / max(w, 1)
        width = fw / max(w, 1)
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": max(min(width, 1.0), 1e-3),
            "mouth_open": 0.0,
            "activity": 0.0
        })
    return detections

def load_haar_cascade():
    cascade_path = getattr(cv2.data, "haarcascades", "")
    cascade_file = os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        return None
    return cascade
