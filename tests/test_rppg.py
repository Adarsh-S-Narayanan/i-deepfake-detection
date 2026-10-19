"""
Unit and integration tests for PulseGuard AI Pro.
Verifies Butterworth SOS filtering, Welch PSD heart-rate extraction,
magnitude-squared coherence, and classifier 3-state output.
"""

import unittest
import numpy as np
from src.rppg_engine import (
    bandpass_filter,
    estimate_heart_rate_welch,
    compute_coherence,
    compute_signal_quality_index,
    plane_orthogonal_to_skin
)
from src.classifier import (
    create_baseline_model,
    PulseGuardClassifier,
    WindowFeatures
)


class TestPulseGuardEngine(unittest.TestCase):

    def setUp(self):
        self.fps = 30.0
        self.duration_sec = 10.0
        self.n_samples = int(self.fps * self.duration_sec)
        self.t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)

    def test_butterworth_sos_filter(self):
        """Verifies bandpass filter attenuates DC and high-frequency components."""
        # 0.1 Hz (drift) + 1.2 Hz (pulse) + 8.0 Hz (high freq noise)
        drift = 10.0 * np.sin(2 * np.pi * 0.1 * self.t)
        pulse = 2.0 * np.sin(2 * np.pi * 1.2 * self.t)
        noise = 3.0 * np.sin(2 * np.pi * 8.0 * self.t)
        raw = drift + pulse + noise

        filtered = bandpass_filter(raw, self.fps, low_hz=0.7, high_hz=3.0)
        self.assertEqual(len(filtered), self.n_samples)
        # Power of filtered signal should be dominated by pulse
        peak_freq, bpm, _, _ = estimate_heart_rate_welch(filtered, self.fps)
        self.assertAlmostEqual(peak_freq, 1.2, delta=0.15)
        self.assertAlmostEqual(bpm, 72.0, delta=9.0)

    def test_welch_heart_rate_estimation(self):
        """Verifies heart rate extraction for standard physiological pulse (80 BPM = 1.333 Hz)."""
        target_hz = 1.333 # 80 BPM
        sig = np.sin(2 * np.pi * target_hz * self.t) + 0.1 * np.random.normal(size=self.n_samples)
        filtered = bandpass_filter(sig, self.fps)
        peak_freq, bpm, freqs, psd = estimate_heart_rate_welch(filtered, self.fps)

        self.assertAlmostEqual(bpm, 80.0, delta=6.0)
        sqi = compute_signal_quality_index(freqs, psd, peak_freq)
        self.assertGreater(sqi, 0.40)

    def test_magnitude_squared_coherence(self):
        """Verifies high coherence for synchronous signals and low coherence for desynchronized signals."""
        freq = 1.2
        # Two synchronous signals with slight phase difference
        sig1 = np.sin(2 * np.pi * freq * self.t) + 0.1 * np.random.normal(size=self.n_samples)
        sig2 = np.sin(2 * np.pi * freq * self.t - 0.1) + 0.1 * np.random.normal(size=self.n_samples)
        # Desynchronized signal (different frequency and noise)
        sig3 = np.sin(2 * np.pi * 2.2 * self.t) + 0.1 * np.random.normal(size=self.n_samples)

        peak_coh_sync, _, _, _ = compute_coherence(sig1, sig2, self.fps, target_freq=freq)
        peak_coh_async, _, _, _ = compute_coherence(sig1, sig3, self.fps, target_freq=freq)

        self.assertGreater(peak_coh_sync, 0.70)
        self.assertLess(peak_coh_async, 0.45)

    def test_plane_orthogonal_to_skin(self):
        """Verifies POS algorithm produces non-zero 1D pulse from RGB matrix."""
        rgb = np.full((self.n_samples, 3), 150.0)
        # Add cardiac pulse to green channel
        rgb[:, 1] += 2.0 * np.sin(2 * np.pi * 1.2 * self.t)
        pos_signal = plane_orthogonal_to_skin(rgb)

        self.assertEqual(len(pos_signal), self.n_samples)
        self.assertGreater(np.std(pos_signal), 1e-4)

    def test_classifier_3_state_policy(self):
        """Verifies Genuine, Manipulated, and Inconclusive verdict states."""
        clf = PulseGuardClassifier()

        # 1. Genuine Windows (Synchronized, high coherence, high SNR)
        genuine_feats = [
            WindowFeatures(
                window_index=0, start_sec=0, end_sec=10,
                fh_peak_freq=1.2, lcheek_peak_freq=1.2, rcheek_peak_freq=1.2,
                freq_delta_fh_lcheek=0.01, freq_delta_fh_rcheek=0.01,
                coherence_fh_lcheek=0.88, coherence_fh_rcheek=0.85, coherence_cheeks=0.90,
                snr_forehead=0.75, motion_instability=0.03, ear_variance=0.002, bpm_forehead=72.0
            )
        ]
        res_genuine = clf.evaluate_video(genuine_feats, face_detected_ratio=1.0)
        self.assertEqual(res_genuine["verdict_code"], "GENUINE")

        # 2. Manipulated Windows (Desynchronized cheek frequencies, low coherence)
        fake_feats = [
            WindowFeatures(
                window_index=0, start_sec=0, end_sec=10,
                fh_peak_freq=1.2, lcheek_peak_freq=1.9, rcheek_peak_freq=0.8,
                freq_delta_fh_lcheek=0.7, freq_delta_fh_rcheek=0.4,
                coherence_fh_lcheek=0.15, coherence_fh_rcheek=0.18, coherence_cheeks=0.20,
                snr_forehead=0.40, motion_instability=0.04, ear_variance=0.001, bpm_forehead=72.0
            )
        ]
        res_fake = clf.evaluate_video(fake_feats, face_detected_ratio=1.0)
        self.assertEqual(res_fake["verdict_code"], "MANIPULATED")

        # 3. Inconclusive Windows (Violent motion or bad SNR)
        inconclusive_feats = [
            WindowFeatures(
                window_index=0, start_sec=0, end_sec=10,
                fh_peak_freq=1.2, lcheek_peak_freq=1.2, rcheek_peak_freq=1.2,
                freq_delta_fh_lcheek=0.0, freq_delta_fh_rcheek=0.0,
                coherence_fh_lcheek=0.5, coherence_fh_rcheek=0.5, coherence_cheeks=0.5,
                snr_forehead=0.12, motion_instability=0.65, ear_variance=0.001, bpm_forehead=72.0
            )
        ]
        res_inconclusive = clf.evaluate_video(inconclusive_feats, face_detected_ratio=1.0)
        self.assertEqual(res_inconclusive["verdict_code"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
