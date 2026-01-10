import os
import cv2
import numpy as np
import statistics
import logging
from typing import List, Dict, Optional, Tuple
from videocuts.config import Config
from videocuts.video.frames import frame_generator_from_ffmpeg
from videocuts.video.detection import (
    detect_faces_with_landmarker,
    detect_faces_with_tasks,
    detect_faces_with_openvino,
    _mouth_open_metric
)
from videocuts.video.models import ensure_face_landmarker_model
from videocuts.utils.device import is_intel_accel_enabled

logger = logging.getLogger(__name__)

def analyze_clip_faces(
    input_video: str,
    clip_start: float,
    clip_end: float,
    cfg: Config
) -> Dict:
    """
    Unified face analysis that extracts frames ONCE and returns all data needed for:
    - Speaker tracking (single active speaker with smooth transitions)
    - Multi-speaker layout detection (all faces per frame)
    """
    analysis_fps = cfg.face_tracking.analysis_fps
    
    # Initialize detector
    detector_backend = "none"
    face_landmarker = None

    # We assume MP_HAS_TASKS is checked by importing mediapipe earlier or using discovery
    # For now, we'll try to load the landmarker
    model_path = ensure_face_landmarker_model(cfg.model.face_landmarker_url, cfg.paths.face_landmarker_model)
    if model_path:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision as mp_tasks_vision
            
            base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
            landmarker_options = mp_tasks_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=4,
                min_face_detection_confidence=cfg.face_tracking.min_confidence,
                min_face_presence_confidence=cfg.face_tracking.min_confidence,
                min_tracking_confidence=0.5
            )
            face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(landmarker_options)
            detector_backend = "landmarker"
        except Exception as exc:
            logger.warning(f"FaceLandmarker unavailable ({exc}); falling back.")

    if cfg.face_tracking.use_openvino and is_intel_accel_enabled():
        if os.path.exists(cfg.paths.openvino_face_model):
            detector_backend = "openvino"
        else:
            logger.warning(f"OpenVINO model not found at {cfg.paths.openvino_face_model}. Falling back.")
    elif cfg.face_tracking.use_yolo:
        detector_backend = "yolo"

    if detector_backend == "none":
        logger.error("No face detector available (MediaPipe failed to load).")
        return {
            "speaker_samples": [],
            "speaker_segments": [{"start": clip_start, "end": clip_end, "center": 0.5}],
            "multi_samples": [],
            "track_detections": {},
            "detector_backend": "none"
        }
            
    logger.info(f"Using '{detector_backend}' for clip {clip_start:.2f}-{clip_end:.2f}s")

    # Tracking state
    tracks: List[Dict] = []
    next_track_id = 0
    track_detections: Dict[int, List[Dict]] = {}
    track_mouth_history: Dict[int, List[float]] = {}
    track_lip_activity: Dict[int, float] = {}
    multi_samples: List[Dict] = []
    speaker_samples: List[Dict] = []
    last_center: Optional[float] = None
    last_track_id: Optional[int] = None
    last_detection_time = clip_start
    locked_track_id: Optional[int] = None
    lock_start_time: float = clip_start
    smoothed_center: Optional[float] = None
    
    lip_speaking_detections = 0
    total_multi_face_frames = 0

    frame_gen = frame_generator_from_ffmpeg(input_video, clip_start, clip_end, analysis_fps)
    
    for ts, frame in frame_gen:
        detections = []
        
        # STAGE 1: Face Detection
        if detector_backend == "openvino":
            # OpenVINO (Highest performance on Intel Arc)
            detections = detect_faces_with_openvino(cfg.paths.openvino_face_model, frame)
            
            # STAGE 2: Landmark Extraction on Crop (MediaPipe)
            if face_landmarker is not None and detections:
                h_frame, w_frame = frame.shape[:2]
                for det in detections:
                    if "box_int" not in det:
                        continue
                    x, y, w, h = det["box_int"]
                    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                    x2, y2 = min(w_frame, x + w + pad_x), min(h_frame, y + h + pad_y)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0: continue
                    import mediapipe as mp
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    result = face_landmarker.detect(mp_crop)
                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                        det["mouth_open"] = _mouth_open_metric(type('obj', (object,), {'landmark': landmarks}))

        elif detector_backend == "yolo" or cfg.face_tracking.use_yolo:
            # YOLO v8 (High Recall)
            from videocuts.video.detection import detect_faces_with_yolo
            detections = detect_faces_with_yolo(cfg.face_tracking.yolo_model_path, frame)
            
            # STAGE 2: Landmark Extraction (MediaPipe) on Crop
            # Only run if we have a landmarker model loaded
            if face_landmarker is not None and detections:
                h_frame, w_frame = frame.shape[:2]
                mp_image = None
                
                for det in detections:
                    if "box_int" not in det:
                        continue
                        
                    x, y, w, h = det["box_int"]
                    # Add padding for better landmarking
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.1)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w_frame, x + w + pad_x)
                    y2 = min(h_frame, y + h + pad_y)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                        
                    # Run MP on crop
                    # Note: We must create a new MP Image for each crop
                    import mediapipe as mp
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    result = face_landmarker.detect(mp_crop)
                    
                    if result.face_landmarks:
                        # Calculate mouth open using landmarks from the first face in crop
                        # Landmarks are normalized relative to CROP, which is fine for "mouth_open" ratio
                        # We don't need absolute frame coordinates for mouth metric
                        landmarks = result.face_landmarks[0]
                        det["mouth_open"] = _mouth_open_metric(type('obj', (object,), {'landmark': landmarks}))

        elif detector_backend == "landmarker":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detect_faces_with_landmarker(face_landmarker, rgb)
        elif detector_backend == "tasks": 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            detections = detect_faces_with_tasks(face_landmarker, rgb) 
        else:
            detections = []

        # Filter out faces:
        # 1. Too small (width < min_face_width)
        # 2. Too low (center_y > 0.6) - likely objects on table (e.g. soda can with eye logo)
        detections = [
            d for d in detections 
            if d.get("width", 0.1) >= cfg.face_tracking.min_face_width
            and d.get("center_y", 0.3) <= 0.6
        ]

        frame_faces = []
        assigned_entries = []
        
        for det in detections:
            best_track = None
            best_dist = 1.0
            for track in tracks:
                if ts - track["last_time"] > cfg.face_tracking.track_max_gap:
                    continue
                dist = abs(det["center"] - track["center"])
                if dist < cfg.face_tracking.track_distance and dist < best_dist:
                    best_track = track
                    best_dist = dist

            if best_track is None:
                best_track = {"id": next_track_id, "center": det["center"], "last_time": ts}
                tracks.append(best_track)
                track_detections[next_track_id] = []
                next_track_id += 1

            best_track["center"] = det["center"]
            best_track["last_time"] = ts
            tid = best_track["id"]
            det["track_id"] = tid
            
            mouth_open = det.get("mouth_open", 0.0)
            if tid not in track_mouth_history:
                track_mouth_history[tid] = []
            
            history = track_mouth_history[tid]
            lip_movement = 0.0
            if len(history) >= 1:
                lip_movement = abs(mouth_open - history[-1])
                if lip_movement > cfg.lip_detection.min_delta:
                    track_lip_activity[tid] = track_lip_activity.get(tid, 0.0) + lip_movement
            
            history.append(mouth_open)
            if len(history) > cfg.lip_detection.history_frames:
                history.pop(0)
            
            det["lip_movement"] = lip_movement
            det["activity"] = lip_movement
            
            face_info = {
                "track_id": tid,
                "center": det["center"],
                "width": det.get("width", 0.1),
                "activity": lip_movement,
                "time": ts
            }
            frame_faces.append(face_info)
            track_detections[tid].append(face_info)
            assigned_entries.append(det)
        
        multi_samples.append({"time": ts, "faces": frame_faces, "num_faces": len(frame_faces)})

        active_entry = None
        if assigned_entries:
            if len(assigned_entries) == 1:
                active_entry = assigned_entries[0]
                if locked_track_id is None:
                    locked_track_id = active_entry.get("track_id")
                    lock_start_time = ts
            else:
                total_multi_face_frames += 1
                locked_entry = None
                if locked_track_id is not None:
                    for entry in assigned_entries:
                        if entry.get("track_id") == locked_track_id:
                            locked_entry = entry
                            break
                
                time_with_current = ts - lock_start_time
                should_switch = False
                best_other_entry = None
                
                if locked_entry is None:
                    should_switch = True
                    best_other_score = -1.0
                    for entry in assigned_entries:
                        tid = entry.get("track_id")
                        lip_score = track_lip_activity.get(tid, 0.0)
                        size_bonus = entry.get("width", 0.1) * 0.1
                        score = lip_score + size_bonus
                        if score > best_other_score:
                            best_other_score = score
                            best_other_entry = entry
                elif time_with_current >= cfg.speaker_lock.min_duration:
                    current_lip_score = track_lip_activity.get(locked_track_id, 0.0)
                    for entry in assigned_entries:
                        tid = entry.get("track_id")
                        if tid != locked_track_id:
                            other_lip_score = track_lip_activity.get(tid, 0.0)
                            if other_lip_score > current_lip_score * cfg.speaker_lock.switch_threshold:
                                should_switch = True
                                lip_speaking_detections += 1
                                if best_other_entry is None or other_lip_score > track_lip_activity.get(best_other_entry.get("track_id"), 0.0):
                                    best_other_entry = entry
                
                if should_switch and best_other_entry is not None:
                    active_entry = best_other_entry
                    locked_track_id = active_entry.get("track_id")
                    lock_start_time = ts
                    track_lip_activity = {locked_track_id: track_lip_activity.get(locked_track_id, 0.0)}
                else:
                    if locked_entry:
                        active_entry = locked_entry
                    else:
                        def get_score(d):
                            lip = track_lip_activity.get(d.get("track_id"), 0.0)
                            size = d.get("width", 0.1) * 0.1
                            return lip + size
                        active_entry = max(assigned_entries, key=get_score)

        if active_entry is None:
            if last_center is not None:
                # Hold last known position instead of drifting to center
                # This prevents showing empty space in split-layout/podcast scenarios
                if smoothed_center is None:
                    smoothed_center = last_center
                else:
                    smoothed_center = smoothed_center * cfg.speaker_lock.position_smoothing + last_center * (1.0 - cfg.speaker_lock.position_smoothing)
                speaker_samples.append({"time": ts, "center": smoothed_center, "track_id": last_track_id})
        else:
            last_center = active_entry["center"]
            last_track_id = active_entry.get("track_id")
            last_detection_time = ts
            if smoothed_center is None:
                smoothed_center = last_center
            else:
                # Simple Smoothing
                # The "Can" false positive is fixed, so we don't need complex whip pans.
                # Just float gently to the new target.
                smoothing_factor = cfg.speaker_lock.position_smoothing
                smoothed_center = smoothed_center * smoothing_factor + last_center * (1.0 - smoothing_factor)

            speaker_samples.append({"time": ts, "center": smoothed_center, "track_id": last_track_id})

    if face_landmarker is not None and hasattr(face_landmarker, "close"):
        face_landmarker.close()

    if not speaker_samples:
        speaker_samples = [{"time": clip_start, "center": 0.5, "track_id": None}, {"time": clip_end, "center": 0.5, "track_id": None}]
    else:
        # Backfilling: If the first detection happens late (e.g. at 2.0s), the time between 0.0s and 2.0s
        # would otherwise be empty (or covered by build_speaker_segments logic).
        # We explicitly enforce a sample at clip_start with the first known position to ensure
        # the camera starts ALREADY focused on the speaker, not checking "center" then panning.
        first_sample = speaker_samples[0]
        if first_sample["time"] > clip_start + 0.05:
            # Insert a "start" sample that matches the first actual detection
            speaker_samples.insert(0, {
                "time": clip_start,
                "center": first_sample["center"],
                "track_id": first_sample["track_id"]
            })
    
    speaker_segments = build_speaker_segments(speaker_samples, clip_start, clip_end, cfg)

    return {
        "speaker_samples": speaker_samples,
        "speaker_segments": speaker_segments,
        "multi_samples": multi_samples,
        "track_detections": track_detections,
        "detector_backend": detector_backend
    }

def build_speaker_segments(
    samples: List[Dict],
    clip_start: float,
    clip_end: float,
    cfg: Config
) -> List[Dict]:
    if not samples:
        return [{"start": clip_start, "end": clip_end, "center": 0.5}]

    samples = sorted(samples, key=lambda s: s["time"])
    
    if samples[0]["time"] > clip_start + 0.01:
        samples.insert(0, {"time": clip_start, "center": samples[0]["center"], "track_id": samples[0].get("track_id")})

    current_start = clip_start
    current_centers = [samples[0]["center"]]
    current_track = samples[0].get("track_id")
    segments: List[Dict] = []

    for sample in samples[1:]:
        ts = min(max(sample["time"], clip_start), clip_end)
        if ts <= current_start:
            current_centers.append(sample["center"])
            continue

        sample_track = sample.get("track_id")
        if sample_track != current_track and current_centers:
            center_val = statistics.median(current_centers)
            segments.append({"start": current_start, "end": ts, "center": center_val})
            current_start = ts
            current_centers = [sample["center"]]
            current_track = sample_track
        else:
            current_centers.append(sample["center"])

    if current_start < clip_end:
        center_val = statistics.median(current_centers) if current_centers else 0.5
        segments.append({"start": current_start, "end": clip_end, "center": center_val})

    # Phase 1: Absorb
    min_duration = cfg.segment_smoothing.min_duration
    absorb_threshold = cfg.segment_smoothing.absorb_threshold
    
    while True:
        changed = False
        new_segments: List[Dict] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_duration = seg["end"] - seg["start"]
            
            if seg_duration < min_duration and len(segments) > 1:
                prev_seg = new_segments[-1] if new_segments else None
                next_seg = segments[i + 1] if i + 1 < len(segments) else None
                
                prev_dist = abs(seg["center"] - prev_seg["center"]) if prev_seg else float('inf')
                next_dist = abs(seg["center"] - next_seg["center"]) if next_seg else float('inf')
                
                if prev_dist <= absorb_threshold and prev_dist <= next_dist and prev_seg:
                    prev_seg["end"] = seg["end"]
                    changed = True
                elif next_dist <= absorb_threshold and next_seg:
                    next_seg["start"] = seg["start"]
                    next_seg["center"] = (next_seg["center"] * 0.8 + seg["center"] * 0.2)
                    changed = True
                else:
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
            i += 1
        segments = new_segments
        if not changed:
            break
    
    # Phase 2: Merge
    merge_threshold = cfg.segment_smoothing.merge_threshold
    merged_segments: List[Dict] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        prev = merged_segments[-1]
        if abs(seg["center"] - prev["center"]) < merge_threshold:
            prev_dur = prev["end"] - prev["start"]
            seg_dur = seg["end"] - seg["start"]
            total_dur = prev_dur + seg_dur
            if total_dur > 0:
                prev["center"] = (prev["center"] * prev_dur + seg["center"] * seg_dur) / total_dur
            prev["end"] = seg["end"]
        else:
            merged_segments.append(seg)

    # Phase 3: Final cleanup
    final_segments: List[Dict] = []
    for seg in merged_segments:
        if seg["end"] - seg["start"] < 0.1 and final_segments:
            final_segments[-1]["end"] = seg["end"]
        else:
            final_segments.append(seg)

    return final_segments

def crop_x_expression_for_segments(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    target_w: int
) -> Optional[str]:
    if not segments:
        return None

    duration = clip_end - clip_start
    base_expr = f"(in_w-{target_w})/2"

    valid_segments = []
    for seg in segments:
        rel_start = max(seg["start"] - clip_start, 0.0)
        rel_end = min(seg["end"] - clip_start, duration)
        if rel_end > rel_start:
            valid_segments.append({"rel_start": rel_start, "rel_end": rel_end, "center": seg["center"]})
    
    if not valid_segments:
        return None

    expr = base_expr
    for seg in reversed(valid_segments):
        center_expr = f"clip(({seg['center']:.4f})*in_w-{target_w}/2,0,in_w-{target_w})"
        expr = f"if(between(t,{seg['rel_start']:.3f},{seg['rel_end']:.3f}),{center_expr},{expr})"

    return expr

def determine_layout_segments(
    samples: List[Dict],
    track_detections: Dict[int, List[Dict]],
    clip_start: float,
    clip_end: float,
    cfg: Config
) -> List[Dict]:
    layout_mode = cfg.layout.mode
    
    if not samples or layout_mode == "single":
        return [{"start": clip_start, "end": clip_end, "layout": "single", "active_track": None}]
    
    if layout_mode in ("split", "wide"):
        return [{"start": clip_start, "end": clip_end, "layout": layout_mode, "faces": []}]
    
    layout_samples = []
    track_activity: Dict[int, float] = {}
    for tid, detections in track_detections.items():
        track_activity[tid] = sum(d.get("activity", 0) for d in detections)
    
    sorted_tracks = sorted(track_activity.items(), key=lambda x: x[1], reverse=True)
    main_tracks = [t[0] for t in sorted_tracks[:2]] if len(sorted_tracks) >= 2 else []
    
    if len(main_tracks) < 2:
        return [{"start": clip_start, "end": clip_end, "layout": "single", "active_track": main_tracks[0] if main_tracks else None}]
    
    for sample in samples:
        ts = sample["time"]
        faces = sample["faces"]
        main_faces = [f for f in faces if f.get("track_id") in main_tracks]
        
        if len(main_faces) < 2:
            visible_tracks = [f.get("track_id") for f in faces]
            active = main_tracks[0] if main_tracks[0] in visible_tracks else (main_tracks[1] if len(main_tracks) > 1 and main_tracks[1] in visible_tracks else None)
            layout_samples.append({"time": ts, "layout": "single", "active_track": active, "faces": faces})
        else:
            # Ensure we actually have both unique tracks visible
            face1 = next((f for f in main_faces if f.get("track_id") == main_tracks[0]), None)
            face2 = next((f for f in main_faces if f.get("track_id") == main_tracks[1]), None)
            
            if not face1 or not face2:
                 # Fallback if we have 2+ faces but they map to the same track (rare duplicate tracking artifact)
                visible_tracks = [f.get("track_id") for f in faces]
                active = main_tracks[0] if main_tracks[0] in visible_tracks else (main_tracks[1] if len(main_tracks) > 1 and main_tracks[1] in visible_tracks else None)
                layout_samples.append({"time": ts, "layout": "single", "active_track": active, "faces": faces})
            else:
                distance = abs(face1["center"] - face2["center"])
                act1 = face1.get("activity", 0)
                act2 = face2.get("activity", 0)
                max_act = max(act1, act2, 0.001)
                both_active = (act1 / max_act > cfg.layout.both_speaking_ratio and act2 / max_act > cfg.layout.both_speaking_ratio)
                
                if distance < cfg.layout.wide_threshold or both_active:
                    layout_samples.append({"time": ts, "layout": "wide", "faces": main_faces, "center": (face1["center"] + face2["center"]) / 2})
                elif distance > cfg.layout.split_threshold:
                    layout_samples.append({"time": ts, "layout": "split", "faces": sorted(main_faces, key=lambda f: f["center"]), "left_center": min(face1["center"], face2["center"]), "right_center": max(face1["center"], face2["center"])})
                else:
                    active_track = main_tracks[0] if act1 >= act2 else main_tracks[1]
                    layout_samples.append({"time": ts, "layout": "single", "active_track": active_track, "faces": faces})
    
    if not layout_samples:
        return [{"start": clip_start, "end": clip_end, "layout": "single", "active_track": None}]
    
    segments = []
    current_layout = layout_samples[0]["layout"]
    current_start = clip_start
    current_data = layout_samples[0]
    
    for sample in layout_samples[1:]:
        if sample["layout"] != current_layout:
            segments.append({"start": current_start, "end": sample["time"], "layout": current_layout, "data": current_data})
            current_layout = sample["layout"]
            current_start = sample["time"]
            current_data = sample
            
    segments.append({"start": current_start, "end": clip_end, "layout": current_layout, "data": current_data})
    return segments

def analyze_best_layout(
    samples: List[Dict],
    track_detections: Dict[int, List[Dict]],
    clip_start: float,
    clip_end: float,
    cfg: Config
) -> Dict:
    """Analyze multi-speaker samples and recommend the best layout for this clip."""
    if not samples or not track_detections:
        return {"recommended_layout": "single", "confidence": 1.0, "metrics": {}, "reason": "No face data available"}
    
    total_frames = len(samples)
    frames_2plus = sum(1 for s in samples if s["num_faces"] >= 2)
    pct_2plus = frames_2plus / max(total_frames, 1)
    
    distances, both_active_count = [], 0
    for sample in samples:
        faces = sample.get("faces", [])
        if len(faces) >= 2:
            sorted_faces = sorted(faces, key=lambda f: f.get("activity", 0), reverse=True)[:2]
            dist = abs(sorted_faces[0]["center"] - sorted_faces[1]["center"])
            distances.append(dist)
            act1, act2 = sorted_faces[0].get("activity", 0), sorted_faces[1].get("activity", 0)
            if act1 / max(act1, act2, 0.001) > 0.3 and act2 / max(act1, act2, 0.001) > 0.3: both_active_count += 1
    
    avg_dist = sum(distances) / len(distances) if distances else 0
    pct_both_active = both_active_count / max(len(distances), 1) if distances else 0
    
    track_activities = {tid: sum(d.get("activity", 0) for d in detections) for tid, detections in track_detections.items()}
    sorted_tracks = sorted(track_activities.items(), key=lambda x: x[1], reverse=True)
    activity_ratio = sorted_tracks[1][1] / max(sorted_tracks[0][1], 0.001) if len(sorted_tracks) >= 2 else 0
    
    metrics = {"num_tracks": len(track_detections), "pct_2plus_faces": round(pct_2plus * 100, 1), "avg_face_distance": round(avg_dist, 3), "pct_both_speaking": round(pct_both_active * 100, 1), "activity_ratio": round(activity_ratio, 2)}
    
    if pct_2plus < 0.3:
        return {"recommended_layout": "single", "confidence": 0.9, "metrics": metrics, "reason": "Single speaker dominant"}
    
    if avg_dist > cfg.layout.split_threshold:
        if pct_both_active > 0.4: return {"recommended_layout": "wide", "confidence": 0.75, "metrics": metrics, "reason": "Both active, far apart"}
        return {"recommended_layout": "split", "confidence": 0.85, "metrics": metrics, "reason": "Two speakers far apart"}
    
    if avg_dist < cfg.layout.wide_threshold: return {"recommended_layout": "wide", "confidence": 0.8, "metrics": metrics, "reason": "Speakers close together"}
    
    if activity_ratio > 0.5 and pct_both_active > 0.3: return {"recommended_layout": "wide", "confidence": 0.65, "metrics": metrics, "reason": "Balanced conversation"}
    return {"recommended_layout": "single", "confidence": 0.6, "metrics": metrics, "reason": "One dominant speaker"}
