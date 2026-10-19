"""
Synthetic Demo Video Generator for PulseGuard AI Pro.
Generates verified synthetic benchmark video clips for instant local testing:
1. demo_genuine.mp4: Synchronized 72 BPM pulse across forehead and cheeks.
2. demo_deepfake.mp4: Desynchronized regional frequencies (Forehead 72 BPM vs Cheeks 114 BPM).
3. demo_inconclusive.mp4: High motion and photometric flicker triggering the SQI safeguard.
"""

import os
import numpy as np
import cv2


DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo")


def generate_synthetic_face_frame(
    frame_idx: int,
    total_frames: int,
    fps: float,
    mode: str = "genuine"
) -> np.ndarray:
    """Generates a single synthetic face frame with embedded optical rPPG signals."""
    w, h = 360, 360
    t = frame_idx / fps

    # Base background (neutral gray-blue)
    frame = np.full((h, w, 3), (220, 215, 210), dtype=np.uint8)

    # Base skin tone in BGR (roughly Fitzpatrick Type II/III)
    base_bgr = np.array([160.0, 185.0, 225.0]) # B, G, R

    # Motion offset
    dx, dy = 0, 0
    if mode == "inconclusive":
        # Rapid erratic translation
        dx = int(18 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 4))
        dy = int(12 * np.cos(2 * np.pi * 2.1 * t) + np.random.normal(0, 3))
    else:
        # Subtle realistic micro-movement
        dx = int(2.0 * np.sin(2 * np.pi * 0.2 * t))
        dy = int(1.5 * np.cos(2 * np.pi * 0.25 * t))

    center_x = (w // 2) + dx
    center_y = (h // 2) + dy

    # 1. Compute biological pulse variations
    # Blood absorption primarily modulates Green channel (index 1)
    cardiac_freq_fh = 1.2 # 72 BPM
    if mode == "genuine":
        # Fully coupled blood perfusion
        cardiac_freq_lc = 1.2
        cardiac_freq_rc = 1.2
        pulse_fh = 3.5 * np.sin(2 * np.pi * cardiac_freq_fh * t)
        pulse_lc = 3.3 * np.sin(2 * np.pi * cardiac_freq_lc * t - 0.05) # slight phase delay
        pulse_rc = 3.4 * np.sin(2 * np.pi * cardiac_freq_rc * t - 0.04)
    elif mode == "deepfake":
        # Desynchronized frequencies & decorrelated phase
        cardiac_freq_lc = 1.9 # 114 BPM
        cardiac_freq_rc = 0.85 # 51 BPM
        pulse_fh = 3.5 * np.sin(2 * np.pi * cardiac_freq_fh * t)
        pulse_lc = 3.0 * np.sin(2 * np.pi * cardiac_freq_lc * t)
        pulse_rc = 2.8 * np.sin(2 * np.pi * cardiac_freq_rc * t + 1.2)
    else: # Inconclusive
        # Dominated by noise and lighting flicker
        flicker = 15.0 * np.sin(2 * np.pi * 0.4 * t)
        pulse_fh = flicker + np.random.normal(0, 8)
        pulse_lc = flicker + np.random.normal(0, 8)
        pulse_rc = flicker + np.random.normal(0, 8)

    # Draw Face Oval
    face_axes = (90, 120)
    cv2.ellipse(frame, (center_x, center_y), face_axes, 0, 0, 360, (150, 175, 215), -1)

    # 2. Draw Embedded ROI Skin Patches with modulated Green channel
    # Forehead patch
    fh_bgr = base_bgr.copy()
    fh_bgr[1] = np.clip(fh_bgr[1] + pulse_fh, 0, 255)
    color_fh = (int(fh_bgr[0]), int(fh_bgr[1]), int(fh_bgr[2]))
    cv2.ellipse(frame, (center_x, center_y - 55), (42, 22), 0, 0, 360, color_fh, -1)

    # Left Cheek patch
    lc_bgr = base_bgr.copy()
    lc_bgr[1] = np.clip(lc_bgr[1] + pulse_lc, 0, 255)
    color_lc = (int(lc_bgr[0]), int(lc_bgr[1]), int(lc_bgr[2]))
    cv2.ellipse(frame, (center_x - 42, center_y + 18), (24, 26), 0, 0, 360, color_lc, -1)

    # Right Cheek patch
    rc_bgr = base_bgr.copy()
    rc_bgr[1] = np.clip(rc_bgr[1] + pulse_rc, 0, 255)
    color_rc = (int(rc_bgr[0]), int(rc_bgr[1]), int(rc_bgr[2]))
    cv2.ellipse(frame, (center_x + 42, center_y + 18), (24, 26), 0, 0, 360, color_rc, -1)

    # 3. Draw Facial Geometry (Eyes, Eyebrows, Nose, Mouth)
    # Eyebrows
    cv2.line(frame, (center_x - 55, center_y - 30), (center_x - 18, center_y - 34), (80, 70, 60), 3)
    cv2.line(frame, (center_x + 18, center_y - 34), (center_x + 55, center_y - 30), (80, 70, 60), 3)

    # Eyes (with periodic blinking)
    blink = (frame_idx % int(fps * 3.5)) < (fps * 0.15) # Blink every 3.5 seconds
    if blink:
        cv2.line(frame, (center_x - 42, center_y - 18), (center_x - 20, center_y - 18), (50, 40, 30), 2)
        cv2.line(frame, (center_x + 20, center_y - 18), (center_x + 42, center_y - 18), (50, 40, 30), 2)
    else:
        # Eye white
        cv2.ellipse(frame, (center_x - 32, center_y - 18), (14, 7), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(frame, (center_x + 32, center_y - 18), (14, 7), 0, 0, 360, (245, 245, 245), -1)
        # Pupils
        cv2.circle(frame, (center_x - 32, center_y - 18), 5, (60, 45, 30), -1)
        cv2.circle(frame, (center_x + 32, center_y - 18), 5, (60, 45, 30), -1)

    # Nose
    cv2.line(frame, (center_x, center_y - 12), (center_x - 4, center_y + 12), (130, 150, 190), 2)
    cv2.line(frame, (center_x - 4, center_y + 12), (center_x + 6, center_y + 14), (130, 150, 190), 2)

    # Mouth
    cv2.ellipse(frame, (center_x, center_y + 52), (28, 10), 0, 0, 180, (90, 80, 160), -1)
    cv2.line(frame, (center_x - 28, center_y + 52), (center_x + 28, center_y + 52), (70, 60, 140), 2)

    # Subtle gaussian blur to blend edges naturally
    frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
    return frame


def create_demo_video(mode: str, output_path: str, duration_sec: float = 10.0, fps: float = 30.0) -> str:
    """Creates a synthetic demo MP4 video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total_frames = int(duration_sec * fps)
    w, h = 360, 360

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for idx in range(total_frames):
        frame = generate_synthetic_face_frame(idx, total_frames, fps, mode=mode)
        out.write(frame)

    out.release()
    print(f"Generated {mode} demo clip at: {output_path} ({total_frames} frames)")
    return output_path


def generate_all_demos():
    """Generates the three benchmark video files if they don't already exist."""
    demos = {
        "genuine": os.path.join(DEMO_DIR, "demo_genuine.mp4"),
        "deepfake": os.path.join(DEMO_DIR, "demo_deepfake.mp4"),
        "inconclusive": os.path.join(DEMO_DIR, "demo_inconclusive.mp4")
    }
    for mode, path in demos.items():
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            create_demo_video(mode, path)
    return demos


if __name__ == "__main__":
    generate_all_demos()
