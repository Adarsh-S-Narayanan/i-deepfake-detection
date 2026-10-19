# 🛡️ PulseGuard AI Pro

> **Biophysical Deepfake Detection & Local Forensics Web Application**  
> Powered by Remote Photoplethysmography (rPPG), Cross-Regional Coherence, Calibrated Machine Learning, and Local Ollama AI Reasoning.

---

## 🌟 Key Features

* **Biophysical Grounding (rPPG):** Measures micro-vascular blood volume pulsations non-invasively through standard RGB video cameras using **POS (Plane-Orthogonal-to-Skin)** and Green-channel baselines.
* **Spatial Multi-ROI Coherence:** Analyzes physiological coupling between the forehead, left cheek, and right cheek using **Magnitude-Squared Coherence** $C_{xy}(f)$.
* **Calibrated 3-State Classifier:** Rejects dangerous binary false alarms with an **Inconclusive** state for compressed or noisy clips, alongside **Likely Genuine** and **Likely Manipulated**.
* **Local Ollama Reasoning Agent:** Integrates directly with your local Ollama daemon (Llama 3, Mistral, Qwen 2.5, DeepSeek-R1) to generate forensic diagnostic briefs and answer interactive cross-examination questions.
* **Interactive Streamlit Web Dashboard:** Includes live facial ROI tracking previews, zoomable Plotly waveforms, Welch PSD frequency spectra, and one-click JSON forensic dossier export.
* **Pre-Built Synthetic Benchmark Clips:** Built-in generators for genuine, deepfake, and noisy test videos for immediate testing without downloading external 10 GB datasets.

---

## 🚀 Quick Start

### 1. Launch the Web Dashboard
Run the Streamlit application:
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. (Optional) Start Local Ollama for AI Reasoning & Chat
To enable real-time local LLM forensic reports and interactive chat:
```powershell
ollama serve
```
If Ollama is offline or paused, PulseGuard automatically uses its built-in biophysical expert rule engine without interruption.

---

## 🧪 Testing with Benchmark Demo Clips

The sidebar includes 3 instant demo clips:
1. **🟢 Real Clip (`demo_genuine.mp4`):** Realistic synchronized 72 BPM blood volume pulse ($C_{xy} \approx 0.97$). Verdict: `Likely Genuine`.
2. **🔴 Deepfake (`demo_deepfake.mp4`):** Forehead pulse retained at 70 BPM, but cheek frequencies desynchronized ($C_{xy} \approx 0.07$). Verdict: `Likely Manipulated (Deepfake)`.
3. **🟡 Inconclusive (`demo_inconclusive.mp4`):** Rapid simulated camera motion and lighting flicker triggering the Signal Quality Index (SQI) safeguard. Verdict: `Inconclusive (Signal Degraded)`.

To re-generate demo clips:
```powershell
python -m src.demo_generator
```

---

## 🔬 Automated Unit Tests

Run the test suite:
```powershell
python -m unittest tests/test_rppg.py
```

---

## 📁 Project Architecture

```
ai/
├── src/
│   ├── video_engine.py      # OpenCV + MediaPipe FaceMesh (478 pts) tracking & ROI cropping
│   ├── rppg_engine.py       # Butterworth SOS filter, POS/CHROM, Welch PSD, Coherence Cxy
│   ├── classifier.py        # 11-D feature windowing & calibrated 3-state ML model
│   ├── ollama_agent.py      # Local Ollama client & prompt-engineered forensic examiner
│   └── demo_generator.py    # Synthetic benchmark video generator
├── models/
│   ├── face_landmarker.task # Google MediaPipe Face Landmarker binary
│   └── pulseguard_model.joblib # Calibrated ML model bundle (StandardScaler + Classifier)
├── data/demo/               # Pre-generated benchmark video clips
├── tests/
│   └── test_rppg.py         # Automated unit test suite
├── app.py                   # Streamlit interactive forensic web application
├── requirements.txt         # Python dependencies
└── README.md
```
