"""
Video Ingestion and Computer Vision Landmarking Engine for PulseGuard AI Pro.
Uses Google MediaPipe FaceLandmarker (478 landmarks) and OpenCV for high-precision
ROI tracking (Forehead, Left Cheek, Right Cheek), Eye Aspect Ratio (EAR), and motion analysis.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")


def ensure_model_exists(target_path: str = MODEL_PATH) -> str:
    """Ensures the MediaPipe Face Landmarker model binary exists locally."""
    target_path = os.path.abspath(target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 1000:
        print(f"Downloading MediaPipe Face Landmarker to {target_path}...")
        urllib.request.urlretrieve(MODEL_URL, target_path)
    return target_path


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass
class ExtractionResult:
    metadata: VideoMetadata
    forehead_rgb: np.ndarray    # (N_frames, 3)
    left_cheek_rgb: np.ndarray  # (N_frames, 3)
    right_cheek_rgb: np.ndarray # (N_frames, 3)
    ear_series: np.ndarray      # (N_frames,)
    motion_series: np.ndarray   # (N_frames,)
    preview_frames: List[np.ndarray] # Sample annotated BGR frames for UI preview
    face_detected_ratio: float


class FaceMeshTracker:
    """Tracks facial landmarks and extracts skin ROIs per frame."""

    # Key landmark indices in MediaPipe Face Mesh
    FOREHEAD_INDICES = [10, 67, 109, 108, 151, 337, 338, 297]
    LEFT_CHEEK_INDICES = [117, 118, 101, 205, 50, 123, 147, 187, 207]
    RIGHT_CHEEK_INDICES = [346, 347, 330, 425, 280, 352, 376, 411, 427]

    # Eye landmarks for EAR calculation
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # p1, p2, p3, p4, p5, p6
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    def __init__(self, model_path: Optional[str] = None):
        actual_path = ensure_model_exists(model_path or MODEL_PATH)
        base_options = python.BaseOptions(model_asset_path=actual_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _calculate_ear(self, landmarks, eye_indices: List[int], width: int, height: int) -> float:
        """Calculates Eye Aspect Ratio (EAR) for blink dynamics."""
        coords = [
            np.array([landmarks[idx].x * width, landmarks[idx].y * height])
            for idx in eye_indices
        ]
        p1, p2, p3, p4, p5, p6 = coords
        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        # Horizontal distance
        h = np.linalg.norm(p1 - p4) + 1e-8
        return float((v1 + v2) / (2.0 * h))

    def _extract_roi_mean_rgb(self, frame_rgb: np.ndarray, landmarks, indices: List[int]) -> np.ndarray:
        """Extracts average RGB values within the convex hull of specified landmark points."""
        h, w, _ = frame_rgb.shape
        points = np.array([
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            for idx in indices
        ], dtype=np.int32)

        # Create mask for ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        roi_pixels = frame_rgb[mask == 255]
        if len(roi_pixels) == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(roi_pixels, axis=0)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        prev_nose_pos: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Dict[str, np.ndarray]], float, float, Optional[np.ndarray], np.ndarray]:
        """
        Processes a single frame.
        
        Returns:
            rois_rgb: Dict with 'forehead', 'left_cheek', 'right_cheek' means (RGB), or None
            ear: Mean Eye Aspect Ratio
            motion: Centroid displacement from previous frame
            current_nose_pos: Coordinates of nose bridge for next frame
            annotated_bgr: Frame with ROI overlays
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.landmarker.detect(mp_image)
        annotated = frame_bgr.copy()

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None, 0.0, 0.0, prev_nose_pos, annotated

        landmarks = result.face_landmarks[0]

        # Extract ROI RGB signals
        fh_rgb = self._extract_roi_mean_rgb(frame_rgb, landmarks, self.FOREHEAD_INDICES)
        lc_rgb = self._extract_roi_mean_rgb(frame_rgb, landmarks, self.LEFT_CHEEK_INDICES)
        rc_rgb = self._extract_roi_mean_rgb(frame_rgb, landmarks, self.RIGHT_CHEEK_INDICES)

        # Eye Aspect Ratio
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES, w, h)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Motion tracking using nose tip / bridge (landmark 1)
        nose_coord = np.array([landmarks[1].x * w, landmarks[1].y * h])
        if prev_nose_pos is not None:
            # Normalized displacement relative to face width
            face_w = abs(landmarks[454].x - landmarks[234].x) * w + 1e-6
            motion = float(np.linalg.norm(nose_coord - prev_nose_pos) / face_w)
        else:
            motion = 0.0

        # Draw ROI overlays on annotated frame for user visibility
        for name, indices, color in [
            ("Forehead", self.FOREHEAD_INDICES, (0, 255, 0)),     # Green
            ("L-Cheek", self.LEFT_CHEEK_INDICES, (255, 255, 0)),  # Cyan
            ("R-Cheek", self.RIGHT_CHEEK_INDICES, (255, 0, 255))  # Magenta
        ]:
            pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in indices], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.polylines(annotated, [hull], True, color, 2)

        rois = {
            "forehead": fh_rgb,
            "left_cheek": lc_rgb,
            "right_cheek": rc_rgb
        }
        return rois, avg_ear, motion, nose_coord, annotated


def process_video(
    video_path: str,
    max_duration_sec: float = 30.0,
    preview_sample_count: int = 6
) -> ExtractionResult:
    """
    Reads a video file and processes each frame through the FaceMesh tracker.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if (raw_fps and 10.0 <= raw_fps <= 120.0) else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Safe frame limit regardless of whether container metadata reports frame count
    target_frames_limit = int(max_duration_sec * fps)
    if total_frames > 0:
        max_frames_to_read = min(total_frames, target_frames_limit)
    else:
        max_frames_to_read = target_frames_limit

    tracker = FaceMeshTracker()

    fh_list = []
    lc_list = []
    rc_list = []
    ear_list = []
    motion_list = []
    preview_frames = []

    prev_nose = None
    detected_count = 0
    sample_interval = max(1, max_frames_to_read // preview_sample_count) if max_frames_to_read > 0 else 1

    for frame_idx in range(max_frames_to_read):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Auto-downscale very high resolution (e.g. 4K) videos to standard HD for fast MediaPipe tracking
        h_f, w_f = frame.shape[:2]
        if max(h_f, w_f) > 1080:
            scale = 1080.0 / max(h_f, w_f)
            frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)), interpolation=cv2.INTER_AREA)

        rois, ear, motion, prev_nose, annotated = tracker.process_frame(frame, prev_nose)

        if rois is not None:
            detected_count += 1
            fh_list.append(rois["forehead"])
            lc_list.append(rois["left_cheek"])
            rc_list.append(rois["right_cheek"])
        else:
            # Handle momentary detection miss by holding previous value or zeros
            last_fh = fh_list[-1] if fh_list else np.array([0.0, 0.0, 0.0])
            last_lc = lc_list[-1] if lc_list else np.array([0.0, 0.0, 0.0])
            last_rc = rc_list[-1] if rc_list else np.array([0.0, 0.0, 0.0])
            fh_list.append(last_fh)
            lc_list.append(last_lc)
            rc_list.append(last_rc)

        ear_list.append(ear)
        motion_list.append(motion)

        if frame_idx % sample_interval == 0 and len(preview_frames) < preview_sample_count:
            preview_frames.append(annotated)

    cap.release()

    n_read = len(fh_list)
    detection_ratio = detected_count / n_read if n_read > 0 else 0.0

    meta = VideoMetadata(
        fps=fps,
        frame_count=n_read,
        duration_sec=n_read / fps if fps > 0 else 0.0,
        width=width,
        height=height
    )

    return ExtractionResult(
        metadata=meta,
        forehead_rgb=np.array(fh_list) if fh_list else np.zeros((0, 3)),
        left_cheek_rgb=np.array(lc_list) if lc_list else np.zeros((0, 3)),
        right_cheek_rgb=np.array(rc_list) if rc_list else np.zeros((0, 3)),
        ear_series=np.array(ear_list) if ear_list else np.zeros((0,)),
        motion_series=np.array(motion_list) if motion_list else np.zeros((0,)),
        preview_frames=preview_frames,
        face_detected_ratio=detection_ratio
    )
