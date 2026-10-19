"""
Local Ollama AI Forensic Reasoning Agent for PulseGuard AI Pro.
Connects to local Ollama instance (Llama 3, Mistral, Qwen, etc.) to analyze
biophysical telemetry JSON and generate structured forensic audit reports and Q&A.
Includes a graceful expert rule-based fallback if Ollama is paused or not running.
"""

from typing import List, Dict, Any, Optional
import requests
import json


OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def check_ollama_status() -> Dict[str, Any]:
    """Checks if the local Ollama daemon is reachable and lists installed models."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            return {
                "available": True,
                "models": models,
                "default_model": models[0] if models else "llama3:latest"
            }
    except Exception:
        pass
    return {
        "available": False,
        "models": [],
        "default_model": None
    }


def generate_fallback_report(telemetry: Dict[str, Any]) -> str:
    """
    Expert rule-based forensic report generator used when local Ollama is offline.
    Provides clear, professional biophysical analysis based on the extracted numbers.
    """
    verdict = telemetry.get("verdict", "Inconclusive")
    verdict_code = telemetry.get("verdict_code", "INCONCLUSIVE")
    fake_prob = telemetry.get("fake_probability", 0.5) * 100.0
    bpm = telemetry.get("avg_heart_rate_bpm", 0.0)
    coh = telemetry.get("avg_coherence", 0.0)
    sqi = telemetry.get("avg_sqi", 0.0)
    reason = telemetry.get("reason", "N/A")

    if verdict_code == "GENUINE":
        summary = (
            f"The video demonstrates clear, physiologically coupled blood volume pulsations. "
            f"An estimated heart rate of **{bpm:.1f} BPM** was isolated across the facial skin regions, "
            f"with a high cross-regional magnitude-squared coherence of **{coh:.2f}**. "
            f"Forehead and bilateral cheek micro-pulsations moved in unison with natural phase consistency."
        )
        recommendation = "Low probability of synthetic face manipulation. Consistent with authentic human recording."
    elif verdict_code == "MANIPULATED":
        summary = (
            f"The video exhibits significant biophysical anomalies characteristic of neural face-swapping or "
            f"generative synthesis. While localized frequency activity was recorded at **{bpm:.1f} BPM**, "
            f"the cross-regional magnitude-squared coherence between the forehead and cheeks collapsed to **{coh:.2f}** "
            f"(nominal biological threshold $\\ge 0.60$). This spatial decorrelation indicates artificial blending or lack of uniform subcutaneous blood perfusion."
        )
        recommendation = "High probability of synthetic manipulation or deepfake face insertion. Recommend secondary manual review."
    else:
        summary = (
            f"The biophysical analysis yielded an **Inconclusive** state. The extracted Signal Quality Index (**{sqi:.2f}**) "
            f"or subject motion exceeded allowable tolerance parameters. "
            f"Video compression or unstable lighting obscured the subtle optical pulse reflections."
        )
        recommendation = "Do not treat this result as evidence of manipulation. Test with a higher-resolution, well-lit recording."

    report = f"""### 🛡️ PulseGuard Forensic Biometric Dossier

**1. Executive Summary:**
{summary}

**2. Biophysical Telemetry Audit:**
* **Estimated Heart Rate:** `{bpm:.1f} BPM`
* **Cross-Regional Coherence ($C_{{xy}}$):** `{coh:.2f}` (Nominal: $>0.60$)
* **Signal Quality Index (SQI):** `{sqi:.2f}`
* **Calculated Manipulation Probability:** `{fake_prob:.1f}%`
* **Forensic Status:** **{verdict}**

**3. Forensic Explanation:**
{reason}

**4. Environmental & Methodological Cautions:**
* Video re-encoding (e.g. social media compression) diminishes high-frequency chrominance shifts.
* Advanced deepfakes that preserve driving video illumination may occasionally show residual pulse frequencies; cross-regional spatial coherence remains the primary discriminating feature.
* Local Ollama reasoning agent was offline; report synthesized via PulseGuard Biophysical Rule Engine.
"""
    return report


def generate_ollama_report(
    telemetry: Dict[str, Any],
    model_name: Optional[str] = None
) -> str:
    """
    Invokes the local Ollama LLM to synthesize a full forensic analysis.
    Falls back gracefully if the call fails.
    """
    status = check_ollama_status()
    if not status["available"]:
        return generate_fallback_report(telemetry)

    selected_model = model_name or status["default_model"] or "llama3:latest"

    system_prompt = (
        "You are PulseGuard AI, an expert digital forensics examiner specializing in "
        "Remote Photoplethysmography (rPPG) and synthetic media (deepfake) detection.\n"
        "You are presented with numerical biometric telemetry from a video clip.\n"
        "Your task is to produce a structured, impartial forensic report.\n"
        "Rules:\n"
        "1. Never guess numbers; strictly reference the provided telemetry JSON.\n"
        "2. Explain what the heart rate and Magnitude-Squared Coherence mean in plain, rigorous terms.\n"
        "3. Emphasize that finding a pulse alone does NOT prove authenticity (modern face-swaps copy source pulse), "
        "which is why cross-region coherence between forehead and cheeks is critical.\n"
        "4. If the verdict is Inconclusive, warn against false accusations due to compression or lighting.\n"
        "5. Output in clean GitHub-style Markdown with clear headings."
    )

    user_prompt = f"""Analyze this biophysical video telemetry:
```json
{json.dumps(telemetry, indent=2)}
```
Generate the official PulseGuard Forensic Diagnostic Report."""

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=30.0
        )
        if resp.status_code == 200:
            result = resp.json()
            content = result.get("message", {}).get("content", "")
            if content.strip():
                return content
    except Exception as e:
        print(f"Ollama call failed ({e}). Using rule-based generator.")

    return generate_fallback_report(telemetry)


def chat_with_agent(
    user_query: str,
    chat_history: List[Dict[str, str]],
    telemetry: Dict[str, Any],
    model_name: Optional[str] = None
) -> str:
    """Interactive follow-up chat with the Ollama agent about the current video analysis."""
    status = check_ollama_status()
    if not status["available"]:
        return (
            "Local Ollama is currently not running in the background. "
            "To enable interactive AI chat, launch Ollama via your terminal using `ollama serve`."
        )

    selected_model = model_name or status["default_model"] or "llama3:latest"

    messages = [
        {
            "role": "system",
            "content": (
                "You are PulseGuard AI Assistant. You answer questions about the video analysis "
                f"using this biophysical telemetry: {json.dumps(telemetry)}. Be concise, factual, and helpful."
            )
        }
    ]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": selected_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.4}
            },
            timeout=25.0
        )
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "No response generated.")
    except Exception as e:
        return f"Error contacting Ollama: {str(e)}"

    return "Unable to process query at this time."
