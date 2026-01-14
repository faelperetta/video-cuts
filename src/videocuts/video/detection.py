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
        width = max(xmax - xmin, 1e-3)
        height = max(ymax - ymin, 1e-3)
        center = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

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
        center_y = (bbox.origin_y + bbox.height / 2.0) / max(h, 1)
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "center_y": max(0.0, min(1.0, center_y)),
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
        center_y = (y + fh / 2) / max(h, 1)
        width = fw / max(w, 1)
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "center_y": max(0.0, min(1.0, center_y)),
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
    
_yolo_model = None

def detect_faces_with_yolo(model_path: str, frame_bgr, device: str = "cpu") -> List[Dict]:
    """
    Detect faces using YOLOv8-Face.
    Returns: List of dicts with {center, center_y, width, activity, box_int}
    """
    global _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        return []

    if _yolo_model is None:
        if not os.path.exists(model_path):
            return []
        _yolo_model = YOLO(model_path)

    h, w = frame_bgr.shape[:2]
    
    # Run inference
    # conf=0.5 default, classes=0 (if standard yolo), but this is face specific so usually class 0 is face
    results = _yolo_model(frame_bgr, verbose=False, conf=0.4, iou=0.5, device=device)
    
    detections = []
    if not results:
        return detections
        
    for result in results:
        for box in result.boxes:
            # box.xyxy is [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords
            
            # Normalize
            width_px = x2 - x1
            height_px = y2 - y1
            center_x_px = x1 + width_px / 2
            center_y_px = y1 + height_px / 2
            
            center = center_x_px / max(w, 1)
            center_y = center_y_px / max(h, 1)
            norm_width = width_px / max(w, 1)
            
            detections.append({
                "center": max(0.0, min(1.0, center)),
                "center_y": max(0.0, min(1.0, center_y)),
                "width": max(min(norm_width, 1.0), 1e-3),
                "mouth_open": 0.0, # Will be filled by Stage 2
                "activity": float(box.conf[0]),
                # Store raw integer box for Stage 2 cropping
                "box_int": [int(x1), int(y1), int(width_px), int(height_px)] 
            })
            
    return detections

_ov_model = None

def detect_faces_with_openvino(model_path: str, frame_bgr, device: str = "GPU") -> List[Dict]:
    """
    Detect faces using OpenVINO (optimized for Intel Arc).
    """
    global _ov_model
    try:
        import openvino as ov
    except ImportError:
        return []

    try:
        from ultralytics import YOLO
        ov_dir = os.path.dirname(model_path)
        
        if _ov_model is None:
            if not os.path.exists(ov_dir):
                return []
            _ov_model = YOLO(ov_dir, task="detect")
            
        results = _ov_model(frame_bgr, device=device.lower(), verbose=False)
        
        detections = []
        if not results:
            return detections
            
        h, w = frame_bgr.shape[:2]
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                width_px = x2 - x1
                height_px = y2 - y1
                center_x_px = x1 + width_px / 2
                center_y_px = y1 + height_px / 2
                
                detections.append({
                    "center": max(0.0, min(1.0, center_x_px / max(w, 1))),
                    "center_y": max(0.0, min(1.0, center_y_px / max(h, 1))),
                    "width": max(min(width_px / max(w, 1), 1.0), 1e-3),
                    "mouth_open": 0.0,
                    "activity": float(box.conf[0]),
                    "box_int": [int(x1), int(y1), int(width_px), int(height_px)] 
                })
        return detections
    except Exception:
        return []
