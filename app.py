"""
PulseGuard AI Pro - Web Forensic Dashboard
Interactive biophysical deepfake detection system powered by rPPG,
cross-regional coherence, calibrated ML classification, and local Ollama AI reasoning.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.video_engine import process_video, ExtractionResult
from src.rppg_engine import (
    plane_orthogonal_to_skin,
    bandpass_filter,
    estimate_heart_rate_welch,
    compute_coherence
)
from src.classifier import (
    extract_windowed_features,
    PulseGuardClassifier,
    WindowFeatures
)
from src.ollama_agent import (
    check_ollama_status,
    generate_ollama_report,
    chat_with_agent
)
from src.demo_generator import generate_all_demos, DEMO_DIR

# Set page configuration
st.set_page_config(
    page_title="PulseGuard AI — Biophysical Deepfake Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for forensic interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .verdict-genuine {
        background-color: #DCFCE7;
        border: 2px solid #22C55E;
        color: #15803D;
        border-radius: 10px;
        padding: 18px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .verdict-fake {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        color: #B91C1C;
        border-radius: 10px;
        padding: 18px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .verdict-inconclusive {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        color: #B45309;
        border-radius: 10px;
        padding: 18px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    return PulseGuardClassifier()


@st.cache_resource
def get_demos():
    return generate_all_demos()


demos = get_demos()
classifier = load_classifier()
ollama_info = check_ollama_status()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("PulseGuard Pro")
    st.caption("Biophysical Deepfake Forensics")
    st.markdown("---")

    # Ollama Health Status
    st.subheader("🤖 Local AI Engine")
    if ollama_info["available"]:
        st.success(f"Ollama Active ({len(ollama_info['models'])} models)")
        selected_model = st.selectbox(
            "Select AI Model:",
            options=ollama_info["models"] if ollama_info["models"] else ["llama3:latest"],
            index=0
        )
    else:
        st.warning("Ollama Daemon Offline")
        st.caption("Using built-in Biophysical Forensic Rule Engine. Start Ollama with `ollama serve` to enable live LLM.")
        selected_model = None

    st.markdown("---")
    st.subheader("⚙️ Signal Parameters")
    rppg_method = st.radio(
        "Color Projection Algorithm:",
        options=["pos", "green"],
        format_func=lambda x: "POS (Plane-Orthogonal-to-Skin)" if x == "pos" else "Green Channel Baseline",
        help="POS mathematically removes specular lighting reflections and adapts to diverse skin tones."
    )
    window_sec = st.slider("Analysis Window (seconds):", min_value=5.0, max_value=20.0, value=10.0, step=1.0)

    st.markdown("---")
    st.subheader("🧪 Load Pre-Built Demo Clips")
    st.caption("Test immediately with synthetic benchmark recordings:")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("🟢 Real Clip", use_container_width=True):
            st.session_state["active_video_path"] = demos["genuine"]
            st.session_state["active_video_name"] = "demo_genuine.mp4"
            st.session_state["current_uploaded_file_name"] = "demo_genuine.mp4"
            st.session_state.pop("analysis_result", None)
            st.rerun()
    with col_d2:
        if st.button("🔴 Deepfake", use_container_width=True):
            st.session_state["active_video_path"] = demos["deepfake"]
            st.session_state["active_video_name"] = "demo_deepfake.mp4"
            st.session_state["current_uploaded_file_name"] = "demo_deepfake.mp4"
            st.session_state.pop("analysis_result", None)
            st.rerun()

    if st.button("🟡 Inconclusive (Motion/Flicker)", use_container_width=True):
        st.session_state["active_video_path"] = demos["inconclusive"]
        st.session_state["active_video_name"] = "demo_inconclusive.mp4"
        st.session_state["current_uploaded_file_name"] = "demo_inconclusive.mp4"
        st.session_state.pop("analysis_result", None)
        st.rerun()


# ----------------- MAIN PANEL -----------------
st.markdown('<div class="main-header">🛡️ PulseGuard AI Forensics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Detect synthetic video manipulation by analyzing subtle biological blood volume pulses (rPPG) and cross-regional facial synchrony.</div>',
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a face video clip (MP4, MOV, AVI) or use a Demo Clip from the sidebar:",
    type=["mp4", "mov", "avi"],
    help="Videos should be 8-30 seconds long, with the face reasonably stable and facing the camera."
)

if uploaded_file is not None:
    if st.session_state.get("current_uploaded_file_name") != uploaded_file.name:
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"pulseguard_upload_{uploaded_file.name}")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state["active_video_path"] = temp_video_path
        st.session_state["active_video_name"] = uploaded_file.name
        st.session_state["current_uploaded_file_name"] = uploaded_file.name
        st.session_state.pop("analysis_result", None)

active_video = st.session_state.get("active_video_path")
active_name = st.session_state.get("active_video_name", "No video loaded")

if active_video and os.path.exists(active_video):
    st.info(f"📁 Loaded Video: **{active_name}**")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.video(active_video)
    with col_v2:
        st.caption("Ready for biophysical pulse and cross-regional coherence analysis.")
        run_analysis = st.button("⚡ Run Forensic Analysis", type="primary", use_container_width=True)

    if run_analysis:
        progress_bar = st.progress(0, text="Initializing Computer Vision & MediaPipe FaceMesh...")
        
        try:
            # 1. Video Processing & Landmark Tracking
            progress_bar.progress(25, text="Tracking 478 3D facial landmarks and isolating Forehead & Cheeks...")
            extraction = process_video(active_video, max_duration_sec=30.0)

            if extraction.metadata.frame_count < 15:
                st.error("Error: Video clip has too few frames for reliable biophysical analysis (minimum 15 frames required).")
                st.stop()

            if extraction.face_detected_ratio < 0.20:
                st.warning("⚠️ Face Detection Low: MediaPipe was unable to track a face across most frames. Results may be inconclusive.")

            # 2. Feature Extraction & Windowing
            progress_bar.progress(55, text="Extracting rPPG pulses, Butterworth filtering, and computing Coherence...")
            features = extract_windowed_features(
                extraction,
                method=rppg_method,
                window_sec=window_sec,
                hop_sec=window_sec / 2.0
            )

            # 3. Machine Learning Classification
            progress_bar.progress(75, text="Evaluating calibrated ML classification & 3-State policy...")
            verdict_info = classifier.evaluate_video(features, extraction.face_detected_ratio)

            # 4. Ollama Forensic Report
            progress_bar.progress(90, text="Consulting Ollama Forensic Reasoning Agent...")
            telemetry_payload = {
                "video_name": active_name,
                "fps": extraction.metadata.fps,
                "duration_seconds": round(extraction.metadata.duration_sec, 2),
                "face_detected_ratio": round(extraction.face_detected_ratio, 2),
                **verdict_info
            }
            report = generate_ollama_report(telemetry_payload, model_name=selected_model)

            progress_bar.progress(100, text="Analysis Complete!")
            
            # Store in session state
            st.session_state["analysis_result"] = {
                "extraction": extraction,
                "features": features,
                "verdict_info": verdict_info,
                "telemetry": telemetry_payload,
                "report": report
            }
            # Directly re-render
            st.rerun()

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()


# ----------------- DISPLAY RESULTS -----------------
if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]
    verdict = res["verdict_info"]
    telemetry = res["telemetry"]
    extraction: ExtractionResult = res["extraction"]
    features: list[WindowFeatures] = res["features"]

    st.markdown("---")

    # 1. Top Verdict Banner
    v_code = verdict["verdict_code"]
    if v_code == "GENUINE":
        banner_cls = "verdict-genuine"
        icon = "✅"
    elif v_code == "MANIPULATED":
        banner_cls = "verdict-fake"
        icon = "🚨"
    else:
        banner_cls = "verdict-inconclusive"
        icon = "⚠️"

    st.markdown(
        f'<div class="{banner_cls}">{icon} VERDICT: {verdict["verdict"].upper()}<br>'
        f'<span style="font-size: 1rem; font-weight: 500;">Confidence: {verdict["confidence_score"]*100:.1f}% | '
        f'Manipulation Probability: {verdict["fake_probability"]*100:.1f}%</span></div>',
        unsafe_allow_html=True
    )
    st.write("")

    # 2. Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💓 Heart Rate</div>
            <div class="metric-value" style="color: #2563EB;">{verdict['avg_heart_rate_bpm']} <span style="font-size: 1rem;">BPM</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        coh_color = "#16A34A" if verdict['avg_coherence'] >= 0.60 else "#DC2626"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔗 Facial Coherence (Cxy)</div>
            <div class="metric-value" style="color: {coh_color};">{verdict['avg_coherence']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        sqi_color = "#16A34A" if verdict['avg_sqi'] >= 0.35 else "#D97706"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Signal Quality (SQI)</div>
            <div class="metric-value" style="color: {sqi_color};">{verdict['avg_sqi']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⏱️ Duration Analyzed</div>
            <div class="metric-value" style="color: #475569;">{extraction.metadata.duration_sec:.1f} <span style="font-size: 1rem;">sec</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 3. Visual Telemetry: ROI Preview & Time-Domain Waveforms
    col_vis1, col_vis2 = st.columns([1, 2])

    with col_vis1:
        st.subheader("📍 Facial Skin ROIs")
        st.caption("MediaPipe FaceMesh tracking active zones:")
        if extraction.preview_frames:
            # Display middle annotated frame
            mid_frame = extraction.preview_frames[len(extraction.preview_frames) // 2]
            # Convert BGR to RGB for Streamlit
            st.image(mid_frame[:, :, ::-1], caption="Forehead (Green) | L-Cheek (Cyan) | R-Cheek (Magenta)", use_container_width=True)
        else:
            st.info("No preview frames available.")

    with col_vis2:
        st.subheader("📈 Filtered Blood Volume Pulse (rPPG)")
        st.caption(f"Zero-phase Butterworth SOS filtered ({0.7}-{3.0} Hz) across facial zones:")
        
        fps = extraction.metadata.fps
        t = np.linspace(0, extraction.metadata.duration_sec, extraction.forehead_rgb.shape[0])

        if rppg_method == "pos":
            sig_fh = plane_orthogonal_to_skin(extraction.forehead_rgb)
            sig_lc = plane_orthogonal_to_skin(extraction.left_cheek_rgb)
            sig_rc = plane_orthogonal_to_skin(extraction.right_cheek_rgb)
        else:
            sig_fh = extraction.forehead_rgb[:, 1]
            sig_lc = extraction.left_cheek_rgb[:, 1]
            sig_rc = extraction.right_cheek_rgb[:, 1]

        filt_fh = bandpass_filter(sig_fh, fps)
        filt_lc = bandpass_filter(sig_lc, fps)
        filt_rc = bandpass_filter(sig_rc, fps)

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=filt_fh, mode='lines', name='Forehead Pulse', line=dict(color='#16A34A', width=2)))
        fig_wave.add_trace(go.Scatter(x=t, y=filt_lc, mode='lines', name='Left Cheek', line=dict(color='#06B6D4', width=1.5, dash='dot')))
        fig_wave.add_trace(go.Scatter(x=t, y=filt_rc, mode='lines', name='Right Cheek', line=dict(color='#D946EF', width=1.5, dash='dash')))
        
        fig_wave.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Time (seconds)",
            yaxis_title="Normalized Amplitude",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    # 4. Spectral Frequency Power & Regional Coherence Plot
    col_spec1, col_spec2 = st.columns(2)

    with col_spec1:
        st.subheader("⚡ Frequency Spectrum (Welch PSD)")
        freq_fh, bpm_fh, freqs, psd = estimate_heart_rate_welch(filt_fh, fps)
        band_mask = (freqs >= 0.5) & (freqs <= 3.5)
        
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs[band_mask], y=psd[band_mask], mode='lines', fill='tozeroy', line=dict(color='#2563EB', width=2), name="Power"))
        # Mark peak
        if freq_fh > 0:
            fig_psd.add_vline(x=freq_fh, line_dash="dash", line_color="#DC2626", annotation_text=f"Peak: {freq_fh:.2f} Hz ({bpm_fh:.1f} BPM)")
        
        fig_psd.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density"
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    with col_spec2:
        st.subheader("📊 Cross-Regional Coherence Matrix")
        # Extract average pairwise coherence
        if features:
            avg_fh_lc = np.mean([f.coherence_fh_lcheek for f in features])
            avg_fh_rc = np.mean([f.coherence_fh_rcheek for f in features])
            avg_lc_rc = np.mean([f.coherence_cheeks for f in features])
        else:
            avg_fh_lc, avg_fh_rc, avg_lc_rc = 0, 0, 0

        pairs = ["Forehead ↔ L-Cheek", "Forehead ↔ R-Cheek", "L-Cheek ↔ R-Cheek"]
        coherence_vals = [avg_fh_lc, avg_fh_rc, avg_lc_rc]
        colors = ['#16A34A' if c >= 0.60 else '#EF4444' for c in coherence_vals]

        fig_bar = go.Figure(go.Bar(
            x=pairs,
            y=coherence_vals,
            marker_color=colors,
            text=[f"{c:.2f}" for c in coherence_vals],
            textposition='auto'
        ))
        fig_bar.add_hline(y=0.60, line_dash="dash", line_color="#D97706", annotation_text="Nominal Threshold (0.60)")
        fig_bar.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis=dict(range=[0, 1.05], title="Magnitude-Squared Coherence"),
            xaxis_title="Facial Zone Pair"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 5. Ollama AI Forensic Diagnosis
    st.markdown("---")
    st.subheader("🤖 AI Forensic Expert Dossier")
    st.markdown(res["report"])

    # 6. Interactive Q&A Chat with the AI Agent
    st.markdown("---")
    st.subheader("💬 Cross-Examine the Forensic Agent")
    st.caption("Ask questions about this specific video's biophysical telemetry:")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question about the video's pulse, coherence, or verdict...")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing telemetry..."):
                reply = chat_with_agent(
                    user_query,
                    st.session_state["chat_history"],
                    telemetry,
                    model_name=selected_model
                )
                st.write(reply)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    # 7. Export Dossier
    st.write("")
    dossier_json = json.dumps(telemetry, indent=2)
    st.download_button(
        label="📥 Download Forensic Dossier (JSON)",
        data=dossier_json,
        file_name=f"pulseguard_dossier_{active_name}.json",
        mime="application/json"
    )

else:
    # Initial empty state instructions
    st.info("👆 Please upload a video file or click one of the **Demo Clips** in the sidebar to start.")
    st.markdown("""
    ### 🔬 How PulseGuard Detects Deepfakes:
    1. **Tracks Skin Patches:** Uses 478 MediaPipe facial landmarks to isolate the forehead and bilateral cheeks.
    2. **Measures rPPG Pulses:** Extracts subtle color shifts corresponding to capillary hemoglobin absorption.
    3. **Tests Coherence:** Compares the blood pulse between the forehead and cheeks. Authentic humans exhibit high spatial-temporal coupling ($C_{xy} > 0.60$).
    4. **3-State Policy:** Classifies as **Likely Genuine**, **Likely Manipulated**, or **Inconclusive** if video quality or motion obscures the signal.
    5. **Ollama Reasoning:** Generates a structured forensic report with local AI.
    """)
