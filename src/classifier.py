"""
Machine Learning Classifier & Feature Extraction Module for PulseGuard AI Pro.
Slices rPPG signals into overlapping temporal windows, extracts 11-D feature vectors,
and applies a 3-State Calibrated Classifier (Likely Genuine, Likely Manipulated, Inconclusive).
"""

import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

from .rppg_engine import (
    plane_orthogonal_to_skin,
    bandpass_filter,
    estimate_heart_rate_welch,
    compute_coherence,
    compute_signal_quality_index
)
from .video_engine import ExtractionResult

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pulseguard_model.joblib")

FEATURE_NAMES = [
    "fh_peak_freq",
    "lcheek_peak_freq",
    "rcheek_peak_freq",
    "freq_delta_fh_lcheek",
    "freq_delta_fh_rcheek",
    "coherence_fh_lcheek",
    "coherence_fh_rcheek",
    "coherence_cheeks",
    "snr_forehead",
    "motion_instability",
    "ear_variance"
]


@dataclass
class WindowFeatures:
    window_index: int
    start_sec: float
    end_sec: float
    fh_peak_freq: float
    lcheek_peak_freq: float
    rcheek_peak_freq: float
    freq_delta_fh_lcheek: float
    freq_delta_fh_rcheek: float
    coherence_fh_lcheek: float
    coherence_fh_rcheek: float
    coherence_cheeks: float
    snr_forehead: float
    motion_instability: float
    ear_variance: float
    bpm_forehead: float


def extract_windowed_features(
    extraction: ExtractionResult,
    method: str = "pos",
    window_sec: float = 10.0,
    hop_sec: float = 5.0
) -> List[WindowFeatures]:
    """
    Slices the extracted multi-ROI RGB time-series into overlapping temporal windows
    and extracts feature vectors from each window.
    """
    fps = extraction.metadata.fps
    n_frames = extraction.forehead_rgb.shape[0]

    window_size = int(window_sec * fps)
    hop_size = int(hop_sec * fps)

    if n_frames < max(30, int(fps * 3.0)):
        # If video is shorter than window_sec, treat entire clip as single window
        window_size = n_frames
        hop_size = n_frames

    # Extract raw 1D pulse signals from the 3 ROIs
    if method == "green":
        # Green channel is column index 1
        raw_fh = extraction.forehead_rgb[:, 1]
        raw_lc = extraction.left_cheek_rgb[:, 1]
        raw_rc = extraction.right_cheek_rgb[:, 1]
    else:
        # Default: Plane-Orthogonal-to-Skin (POS)
        raw_fh = plane_orthogonal_to_skin(extraction.forehead_rgb)
        raw_lc = plane_orthogonal_to_skin(extraction.left_cheek_rgb)
        raw_rc = plane_orthogonal_to_skin(extraction.right_cheek_rgb)

    features_list: List[WindowFeatures] = []
    start_idx = 0
    window_idx = 0

    while start_idx + 15 <= n_frames:
        end_idx = min(start_idx + window_size, n_frames)

        win_fh = raw_fh[start_idx:end_idx]
        win_lc = raw_lc[start_idx:end_idx]
        win_rc = raw_rc[start_idx:end_idx]
        win_motion = extraction.motion_series[start_idx:end_idx]
        win_ear = extraction.ear_series[start_idx:end_idx]

        # Filter signals
        filt_fh = bandpass_filter(win_fh, fps)
        filt_lc = bandpass_filter(win_lc, fps)
        filt_rc = bandpass_filter(win_rc, fps)

        # Spectral analysis on Forehead
        fh_freq, fh_bpm, freqs, psd = estimate_heart_rate_welch(filt_fh, fps)
        lc_freq, _, _, _ = estimate_heart_rate_welch(filt_lc, fps)
        rc_freq, _, _, _ = estimate_heart_rate_welch(filt_rc, fps)

        # Coherence evaluated at dominant forehead cardiac frequency
        coh_fh_lc, _, _, _ = compute_coherence(filt_fh, filt_lc, fps, target_freq=fh_freq)
        coh_fh_rc, _, _, _ = compute_coherence(filt_fh, filt_rc, fps, target_freq=fh_freq)
        coh_cheeks, _, _, _ = compute_coherence(filt_lc, filt_rc, fps, target_freq=fh_freq)

        # Signal Quality Index (SNR) on forehead
        snr_fh = compute_signal_quality_index(freqs, psd, fh_freq)

        # Motion and EAR dynamics
        motion_score = float(np.mean(win_motion)) if len(win_motion) > 0 else 0.0
        ear_var = float(np.var(win_ear)) if len(win_ear) > 0 else 0.0

        feat = WindowFeatures(
            window_index=window_idx,
            start_sec=round(start_idx / fps, 2),
            end_sec=round(end_idx / fps, 2),
            fh_peak_freq=round(fh_freq, 3),
            lcheek_peak_freq=round(lc_freq, 3),
            rcheek_peak_freq=round(rc_freq, 3),
            freq_delta_fh_lcheek=round(abs(fh_freq - lc_freq), 3),
            freq_delta_fh_rcheek=round(abs(fh_freq - rc_freq), 3),
            coherence_fh_lcheek=round(coh_fh_lc, 3),
            coherence_fh_rcheek=round(coh_fh_rc, 3),
            coherence_cheeks=round(coh_cheeks, 3),
            snr_forehead=round(snr_fh, 3),
            motion_instability=round(motion_score, 4),
            ear_variance=round(ear_var, 5),
            bpm_forehead=round(fh_bpm, 1)
        )
        features_list.append(feat)

        if end_idx >= n_frames:
            break
        start_idx += hop_size
        window_idx += 1

    return features_list


def create_baseline_model() -> Tuple[StandardScaler, CalibratedClassifierCV]:
    """
    Creates and calibrates a robust baseline model based on biophysical rPPG distributions.
    Trained on synthetic physiological priors:
    - Genuine: High coherence (>0.6), low freq delta (<0.08 Hz), good SNR.
    - Manipulated: Low coherence (<0.35), high freq delta, or decorrelated cheek signals.
    """
    np.random.seed(42)
    n_samples = 400

    # 1. Genuine Distribution (Class 0)
    g_fh_freq = np.random.uniform(1.0, 1.8, n_samples)
    g_dfh_lc = np.abs(np.random.normal(0.02, 0.02, n_samples))
    g_dfh_rc = np.abs(np.random.normal(0.02, 0.02, n_samples))
    g_lc_freq = g_fh_freq + g_dfh_lc
    g_rc_freq = g_fh_freq + g_dfh_rc
    g_coh_fh_lc = np.clip(np.random.normal(0.82, 0.08, n_samples), 0.55, 0.98)
    g_coh_fh_rc = np.clip(np.random.normal(0.80, 0.09, n_samples), 0.52, 0.98)
    g_coh_cheeks = np.clip(np.random.normal(0.85, 0.07, n_samples), 0.58, 0.99)
    g_snr = np.clip(np.random.normal(0.65, 0.12, n_samples), 0.35, 0.95)
    g_motion = np.clip(np.random.exponential(0.05, n_samples), 0.0, 0.25)
    g_ear_var = np.clip(np.random.normal(0.003, 0.001, n_samples), 0.0005, 0.01)

    X_genuine = np.column_stack([
        g_fh_freq, g_lc_freq, g_rc_freq,
        g_dfh_lc, g_dfh_rc,
        g_coh_fh_lc, g_coh_fh_rc, g_coh_cheeks,
        g_snr, g_motion, g_ear_var
    ])
    y_genuine = np.zeros(n_samples)

    # 2. Manipulated / Deepfake Distribution (Class 1)
    f_fh_freq = np.random.uniform(0.9, 2.2, n_samples)
    f_dfh_lc = np.random.uniform(0.12, 0.9, n_samples)
    f_dfh_rc = np.random.uniform(0.12, 0.9, n_samples)
    f_lc_freq = f_fh_freq + f_dfh_lc
    f_rc_freq = f_fh_freq - f_dfh_rc
    f_coh_fh_lc = np.clip(np.random.normal(0.24, 0.12, n_samples), 0.02, 0.48)
    f_coh_fh_rc = np.clip(np.random.normal(0.22, 0.11, n_samples), 0.02, 0.45)
    f_coh_cheeks = np.clip(np.random.normal(0.28, 0.14, n_samples), 0.03, 0.52)
    f_snr = np.clip(np.random.normal(0.40, 0.18, n_samples), 0.1, 0.85)
    f_motion = np.clip(np.random.exponential(0.06, n_samples), 0.0, 0.3)
    f_ear_var = np.clip(np.random.normal(0.001, 0.0008, n_samples), 0.0001, 0.008)

    X_fake = np.column_stack([
        f_fh_freq, f_lc_freq, f_rc_freq,
        f_dfh_lc, f_dfh_rc,
        f_coh_fh_lc, f_coh_fh_rc, f_coh_cheeks,
        f_snr, f_motion, f_ear_var
    ])
    y_fake = np.ones(n_samples)

    X = np.vstack([X_genuine, X_fake])
    y = np.concatenate([y_genuine, y_fake])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    calibrated_clf = CalibratedClassifierCV(estimator=base_rf, cv=4)
    calibrated_clf.fit(X_scaled, y)

    return scaler, calibrated_clf


class PulseGuardClassifier:
    """Manages model loading, inference, and 3-State policy evaluation."""

    def __init__(self, model_path: str = MODEL_SAVE_PATH):
        self.model_path = os.path.abspath(model_path)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.scaler, self.model = self._load_or_create_model()

    def _load_or_create_model(self) -> Tuple[StandardScaler, CalibratedClassifierCV]:
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                return bundle["scaler"], bundle["model"]
            except Exception as e:
                print(f"Error loading {self.model_path}: {e}. Creating fresh baseline.")

        scaler, model = create_baseline_model()
        joblib.dump({"scaler": scaler, "model": model}, self.model_path)
        print(f"Baseline classifier saved to {self.model_path}")
        return scaler, model

    def evaluate_video(
        self,
        features_list: List[WindowFeatures],
        face_detected_ratio: float,
        sqi_threshold: float = 0.28,
        motion_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Applies the 3-State Decision Policy across all video windows.
        """
        if not features_list or face_detected_ratio < 0.40:
            return {
                "verdict": "Inconclusive",
                "verdict_code": "INCONCLUSIVE",
                "fake_probability": 0.5,
                "confidence_score": 0.0,
                "avg_heart_rate_bpm": 0.0,
                "avg_coherence": 0.0,
                "avg_sqi": 0.0,
                "reason": "Face detection was lost or duration was insufficient for reliable analysis.",
                "window_count": len(features_list)
            }

        avg_sqi = float(np.mean([f.snr_forehead for f in features_list]))
        avg_motion = float(np.mean([f.motion_instability for f in features_list]))
        avg_coherence = float(np.mean([
            (f.coherence_fh_lcheek + f.coherence_fh_rcheek + f.coherence_cheeks) / 3.0
            for f in features_list
        ]))
        avg_bpm = float(np.mean([f.bpm_forehead for f in features_list if f.bpm_forehead > 0]))

        # Safeguard: If signal is corrupted by illumination or violent motion, force Inconclusive
        if avg_sqi < sqi_threshold or avg_motion > motion_threshold:
            return {
                "verdict": "Inconclusive (Signal Degraded)",
                "verdict_code": "INCONCLUSIVE",
                "fake_probability": 0.5,
                "confidence_score": round(max(0.0, 1.0 - (avg_motion * 2.0)), 2),
                "avg_heart_rate_bpm": round(avg_bpm, 1),
                "avg_coherence": round(avg_coherence, 3),
                "avg_sqi": round(avg_sqi, 3),
                "reason": f"Signal Quality Index ({avg_sqi:.2f}) or Motion ({avg_motion:.2f}) exceeded reliable tolerance thresholds.",
                "window_count": len(features_list)
            }

        # Format feature matrix for ML model
        X_raw = np.array([[
            f.fh_peak_freq,
            f.lcheek_peak_freq,
            f.rcheek_peak_freq,
            f.freq_delta_fh_lcheek,
            f.freq_delta_fh_rcheek,
            f.coherence_fh_lcheek,
            f.coherence_fh_rcheek,
            f.coherence_cheeks,
            f.snr_forehead,
            f.motion_instability,
            f.ear_variance
        ] for f in features_list])

        X_scaled = self.scaler.transform(X_raw)
        window_probs = self.model.predict_proba(X_scaled)[:, 1] # Probability of Fake
        mean_fake_prob = float(np.mean(window_probs))

        # 3-State Verdict
        if mean_fake_prob >= 0.62:
            verdict = "Likely Manipulated (Deepfake)"
            verdict_code = "MANIPULATED"
            confidence = mean_fake_prob
            reason = (
                f"Cross-region coherence ({avg_coherence:.2f}) indicates desynchronized physiological "
                f"blood flow between forehead and cheek patches, characteristic of synthetic video synthesis."
            )
        elif mean_fake_prob <= 0.38:
            verdict = "Likely Genuine"
            verdict_code = "GENUINE"
            confidence = 1.0 - mean_fake_prob
            reason = (
                f"Strong physiological synchrony detected across multiple facial zones "
                f"(Coherence: {avg_coherence:.2f}, Est. Pulse: {avg_bpm:.1f} BPM)."
            )
        else:
            verdict = "Inconclusive (Borderline Confidence)"
            verdict_code = "INCONCLUSIVE"
            confidence = 0.5
            reason = (
                f"Classifier uncertainty is high (Manipulation probability: {mean_fake_prob*100:.1f}%). "
                f"The evidence is insufficient to decisively verify authenticity."
            )

        return {
            "verdict": verdict,
            "verdict_code": verdict_code,
            "fake_probability": round(mean_fake_prob, 3),
            "confidence_score": round(confidence, 3),
            "avg_heart_rate_bpm": round(avg_bpm, 1),
            "avg_coherence": round(avg_coherence, 3),
            "avg_sqi": round(avg_sqi, 3),
            "reason": reason,
            "window_count": len(features_list),
            "window_probabilities": [round(float(p), 3) for p in window_probs]
        }
