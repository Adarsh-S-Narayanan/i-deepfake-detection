"""
rPPG Signal Processing Engine for PulseGuard AI Pro.
Implements detrending, Butterworth SOS filtering, POS/CHROM color projection,
Welch PSD spectral analysis, and Magnitude-Squared Coherence.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import signal as sp_signal


def detrend_signal(sig: np.ndarray) -> np.ndarray:
    """Removes slow illumination drift via linear/polynomial detrending."""
    if len(sig) < 4:
        return sig
    return sp_signal.detrend(sig, type='linear')


def bandpass_filter(
    sig: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter using Second-Order Sections (SOS).
    Frequencies correspond to 42 to 180 BPM.
    """
    if len(sig) < 15 or fps <= 0:
        return np.zeros_like(sig)

    nyquist = 0.5 * fps
    # Ensure cutoff frequencies are within valid (0, nyquist) range
    low = max(0.01, min(low_hz / nyquist, 0.95))
    high = max(low + 0.05, min(high_hz / nyquist, 0.99))

    sos = sp_signal.butter(order, [low, high], btype='bandpass', output='sos')
    # Use zero-phase forward-backward filter so no phase distortion is introduced
    detrended = detrend_signal(sig)
    
    # Check if signal is long enough for default padlen
    padlen = 3 * (2 * order)
    if len(detrended) <= padlen:
        padlen = len(detrended) - 1

    filtered = sp_signal.sosfiltfilt(sos, detrended, padlen=padlen)
    return filtered


def plane_orthogonal_to_skin(rgb_signal: np.ndarray) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., 2017).
    Projects normalized RGB signals onto an orthogonal subspace to cancel specular reflection.
    
    Args:
        rgb_signal: Array of shape (N_frames, 3) where columns are [R, G, B].
    Returns:
        1D rPPG pulse signal of length N_frames.
    """
    n_frames = len(rgb_signal)
    if n_frames < 5:
        return np.zeros(n_frames)

    # Normalize each channel by its temporal mean
    mean_rgb = np.mean(rgb_signal, axis=0) + 1e-8
    c_n = rgb_signal / mean_rgb

    # POS projection matrices:
    # S1 = G - B
    # S2 = G + B - 2R
    s1 = c_n[:, 1] - c_n[:, 2]
    s2 = c_n[:, 1] + c_n[:, 2] - 2 * c_n[:, 0]

    std_s1 = np.std(s1)
    std_s2 = np.std(s2) + 1e-8

    alpha = std_s1 / std_s2
    h = s1 + alpha * s2
    return h


def extract_chrom_signal(rgb_signal: np.ndarray) -> np.ndarray:
    """
    Chrominance-based rPPG method (CHROM) (de Haan & Jeanne, 2013).
    """
    n_frames = len(rgb_signal)
    if n_frames < 5:
        return np.zeros(n_frames)

    mean_rgb = np.mean(rgb_signal, axis=0) + 1e-8
    c_n = rgb_signal / mean_rgb

    # X_s = 3*R - 2*G
    # Y_s = 1.5*R + G - 1.5*B
    xs = 3.0 * c_n[:, 0] - 2.0 * c_n[:, 1]
    ys = 1.5 * c_n[:, 0] + c_n[:, 1] - 1.5 * c_n[:, 2]

    std_x = np.std(xs)
    std_y = np.std(ys) + 1e-8
    alpha = std_x / std_y
    return xs - alpha * ys


def estimate_heart_rate_welch(
    filtered_signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Estimates heart rate using Welch's Power Spectral Density (PSD).
    
    Returns:
        (peak_freq_hz, bpm, freqs, psd_power)
    """
    n = len(filtered_signal)
    if n < 10 or fps <= 0:
        return 0.0, 0.0, np.array([]), np.array([])

    nperseg = min(n, max(16, int(fps * 6)))
    freqs, psd = sp_signal.welch(filtered_signal, fs=fps, window='hann', nperseg=nperseg, noverlap=nperseg // 2)

    # Restrict to physiological band
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return 0.0, 0.0, freqs, psd

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    peak_idx = np.argmax(band_psd)
    peak_freq = float(band_freqs[peak_idx])
    bpm = peak_freq * 60.0

    return peak_freq, bpm, freqs, psd


def compute_coherence(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fps: float,
    target_freq: Optional[float] = None,
    low_hz: float = 0.7,
    high_hz: float = 3.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculates Magnitude-Squared Coherence between two regional signals.
    C_xy(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))
    
    Returns:
        (peak_coherence, mean_band_coherence, freqs, coherence_array)
    """
    n = min(len(sig1), len(sig2))
    if n < 16 or fps <= 0:
        return 0.0, 0.0, np.array([]), np.array([])

    # Segment size: target 2 to 3 seconds, ensuring at least 3-4 Welch segments for unbiased coherence
    target_seg = int(fps * 2.5)
    nperseg = max(16, min(target_seg, n // 3))
    freqs, c_xy = sp_signal.coherence(sig1[:n], sig2[:n], fs=fps, window='hann', nperseg=nperseg, noverlap=nperseg // 2)

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return 0.0, 0.0, freqs, c_xy

    band_cxy = c_xy[band_mask]
    band_freqs = freqs[band_mask]

    # If target frequency is specified, evaluate coherence at that frequency bin
    if target_freq is not None and target_freq > 0:
        idx = np.argmin(np.abs(band_freqs - target_freq))
        peak_coh = float(np.clip(band_cxy[idx], 0.0, 1.0))
    else:
        peak_coh = float(np.clip(np.max(band_cxy), 0.0, 1.0))

    mean_coh = float(np.clip(np.mean(band_cxy), 0.0, 1.0))
    return peak_coh, mean_coh, freqs, c_xy


def compute_signal_quality_index(
    psd_freqs: np.ndarray,
    psd_power: np.ndarray,
    peak_freq: float,
    half_bandwidth: float = 0.15,
    low_hz: float = 0.7,
    high_hz: float = 3.0
) -> float:
    """
    Computes a Signal Quality Index (SQI) based on spectral peak-to-noise ratio.
    SQI is close to 1.0 for a clean periodic pulse, and close to 0.0 for white noise or flat signal.
    """
    if len(psd_freqs) == 0 or len(psd_power) == 0 or peak_freq <= 0:
        return 0.0

    band_mask = (psd_freqs >= low_hz) & (psd_freqs <= high_hz)
    total_band_power = np.sum(psd_power[band_mask])
    if total_band_power <= 1e-12:
        return 0.0

    peak_mask = (psd_freqs >= (peak_freq - half_bandwidth)) & (psd_freqs <= (peak_freq + half_bandwidth))
    peak_power = np.sum(psd_power[peak_mask])

    # Fraction of energy concentrated around the dominant harmonic
    ratio = peak_power / total_band_power
    return float(np.clip(ratio, 0.0, 1.0))
