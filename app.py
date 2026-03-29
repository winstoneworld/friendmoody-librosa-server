"""
friendmoody Librosa Audio Analysis Server
==========================================
Flask + Librosa 기반 오디오 분석 및 채점 서버.
n8n에서 HTTP POST /analyze 요청을 받아 무드별 동적 채점 결과를 반환합니다.

Moods: CALM / MELO / ANXI
Endpoint: POST /analyze  { "url": "...", "mood": "CALM|MELO|ANXI|auto" }
"""

import os
import tempfile

import librosa
import numpy as np
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

MOOD_SPECS = {
    "CALM": {
        "tempo_range":        (62,  84),
        "energy_avg":          0.13,
        "energy_tolerance":    0.05,
        "acousticness_range": (0.80, 0.97),
        "brightness_target":   2000,
    },
    "MELO": {
        "tempo_range":        (72,  96),
        "energy_avg":          0.32,
        "energy_tolerance":    0.10,
        "acousticness_range": (0.55, 0.82),
        "brightness_target":   3000,
    },
    "ANXI": {
        "tempo_range":        (82, 108),
        "energy_avg":          0.52,
        "energy_tolerance":    0.15,
        "acousticness_range": (0.20, 0.55),
        "brightness_target":   4000,
    },
}

def _download_audio(url: str) -> str:
    resp = requests.get(url, timeout=90, stream=True)
    resp.raise_for_status()
    suffix = ".mp3"
    url_lower = url.lower().split("?")[0]
    for ext in (".wav", ".flac", ".ogg", ".m4a", ".aac"):
        if url_lower.endswith(ext):
            suffix = ext
            break
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in resp.iter_content(chunk_size=65536):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

def _bpm_stability(beats, sr: int) -> float:
    if len(beats) < 3:
        return 0.0
    beat_times = librosa.frames_to_time(beats, sr=sr)
    intervals  = np.diff(beat_times)
    cv         = np.std(intervals) / (np.mean(intervals) + 1e-9)
    return float(max(0.0, min(20.0, 20.0 * (1.0 - cv / 0.10))))

def _acousticness(y, sr: int) -> float:
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_hz = float(np.mean(rolloff))
    return float(max(0.0, min(1.0, 1.0 - (mean_hz - 2000.0) / 8000.0)))

def _speechiness(y) -> float:
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    return float(min(1.0, float(np.mean(zcr)) / 0.30))

def _zone_clean(segment_y, full_rms_mean: float, threshold: float = 0.60) -> bool:
    seg_rms = float(np.mean(librosa.feature.rms(y=segment_y)[0]))
    return seg_rms < full_rms_mean * threshold

def analyze_audio(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=22050, duration=180)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    bpm_stab   = _bpm_stability(beats, sr)
    rms        = librosa.feature.rms(y=y)[0]
    energy_avg = float(np.mean(rms))
    sc         = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(sc))
    acousticness = _acousticness(y, sr)
    speechiness  = _speechiness(y)
    zone_frames  = 10 * sr
    intro_y = y[:zone_frames]  if len(y) > zone_frames else y
    outro_y = y[-zone_frames:] if len(y) > zone_frames else y
    return {
        "tempo": tempo, "energy": energy_avg, "brightness": brightness,
        "acousticness": acousticness, "speechiness": speechiness,
        "bpm_stability": bpm_stab,
        "intro_clean": _zone_clean(intro_y, energy_avg),
        "outro_clean": _zone_clean(outro_y, energy_avg),
        "duration": duration,
    }

def detect_mood(features: dict) -> str:
    best, best_score = "MELO", -1.0
    for mood, spec in MOOD_SPECS.items():
        bmin, bmax = spec["tempo_range"]
        amin, amax = spec["acousticness_range"]
        t_match = 1.0 if bmin <= features["tempo"]        <= bmax else 0.0
        a_match = 1.0 if amin <= features["acousticness"] <= amax else 0.0
        e_diff  = abs(features["energy"] - spec["energy_avg"])
        e_match = max(0.0, 1.0 - e_diff / (spec["energy_avg"] + 0.01))
        score   = t_match * 0.40 + e_match * 0.40 + a_match * 0.20
        if score > best_score:
            best_score, best = score, mood
    return best

def score_track(features: dict, mood: str) -> dict:
    spec = MOOD_SPECS[mood]
    scores, details = {}, {}
    s1 = min(20.0, features["bpm_stability"])
    scores["bpm_stability"]      = round(s1, 1)
    details["bpm_stability"]     = f"{s1:.1f}/20"
    s2 = (10.0 if features["intro_clean"] else 0.0) + (10.0 if features["outro_clean"] else 0.0)
    scores["intro_outro_clean"]  = round(s2, 1)
    details["intro_outro_clean"] = f"{s2:.1f}/20"
    e_target = spec["energy_avg"]; e_tol = spec["energy_tolerance"]
    e_diff   = abs(features["energy"] - e_target)
    s3 = 15.0 if e_diff <= e_tol else max(0.0, 15.0 * (1.0 - (e_diff - e_tol) / (e_target + 0.01)))
    scores["energy_curve"]  = round(s3, 1); details["energy_curve"] = f"{s3:.1f}/15"
    b_diff = abs(features["brightness"] - spec["brightness_target"])
    s4 = max(0.0, 10.0 * (1.0 - b_diff / 6000.0))
    scores["freq_balance"]  = round(s4, 1); details["freq_balance"] = f"{s4:.1f}/10"
    bmin, bmax = spec["tempo_range"]; t = features["tempo"]
    s5 = 15.0 if bmin <= t <= bmax else max(0.0, 15.0 * (1.0 - min(abs(t-bmin),abs(t-bmax)) / ((bmax-bmin)/2.0)))
    scores["genre_intent"]  = round(s5, 1); details["genre_intent"] = f"{s5:.1f}/15"
    amin, amax = spec["acousticness_range"]; ac = features["acousticness"]
    s6 = 10.0 if amin <= ac <= amax else max(0.0, 10.0 * (1.0 - min(abs(ac-amin),abs(ac-amax)) / 0.5))
    scores["mix_position"]  = round(s6, 1); details["mix_position"] = f"{s6:.1f}/10"
    dur = features.get("duration", 180)
    dur_s = 5.0 if 150 <= dur <= 240 else (3.0 if 120 <= dur <= 270 else 1.0)
    sp = features["speechiness"]
    s7 = dur_s + 5.0 * max(0.0, 1.0 - sp * 2.0)
    scores["structure"]  = round(s7, 1); details["structure"] = f"{s7:.1f}/10"
    total  = sum(scores.values())
    passed = (total >= 70.0) and (features["bpm_stability"] >= 15.0)
    return {"mood": mood, "total_score": round(total, 1), "scores": scores, "details": details, "passed": passed}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "friendmoody-librosa-server", "version": "1.0.0"})

@app.route("/mood_specs", methods=["GET"])
def mood_specs_route():
    return jsonify(MOOD_SPECS)

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True)
    if not body or "url" not in body:
        return jsonify({"error": "Request body must include 'url'"}), 400
    audio_url  = body["url"]
    mood_input = body.get("mood", "auto").strip().upper()
    tmp_path = None
    try:
        tmp_path = _download_audio(audio_url)
        features = analyze_audio(tmp_path)
        mood = detect_mood(features) if mood_input in ("AUTO", "", "NONE") else mood_input
        if mood not in MOOD_SPECS:
            return jsonify({"error": f"Unknown mood '{mood}'. Valid: CALM, MELO, ANXI, auto"}), 400
        scoring = score_track(features, mood)
        return jsonify({
            "status": "success", "url": audio_url, "detected_mood": mood,
            "features": {
                "tempo":         round(features["tempo"],         1),
                "energy":        round(features["energy"],        4),
                "brightness":    round(features["brightness"],    1),
                "acousticness":  round(features["acousticness"],  3),
                "speechiness":   round(features["speechiness"],   3),
                "bpm_stability": round(features["bpm_stability"], 2),
                "intro_clean":   features["intro_clean"],
                "outro_clean":   features["outro_clean"],
                "duration_sec":  round(features["duration"],      1),
            },
            "scoring": scoring, "passed": scoring["passed"], "total_score": scoring["total_score"],
        })
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": f"Failed to download audio: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
