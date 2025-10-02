# app.py - EchoSoul (robust import + graceful audio fallbacks)
# Purpose: fixes ModuleNotFoundError from pydub / pyaudioop by avoiding hard imports at top-level.
# Keep the rest of the features (chat, timeline, vault, export, brain mimic, optional live call).
#
# Deployment notes:
# - If you want full audio STT/TTS/webrtc support on Streamlit Cloud, add packages.txt:
#     ffmpeg
#     libportaudio2
#     libsndfile1
# - Use the corrected requirements.txt (use cryptography >= 44.0.0 for streamlit-webrtc compatibility).
#
# The app tries to enable audio features when available, otherwise shows helpful messages.

import os
import io
import json
import time
import queue
import tempfile
import threading
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet

# Optional heavy dependencies (lazy / safe imports)
gtts_available = False
gtts_error = ""
try:
    from gtts import gTTS
    gtts_available = True
except Exception as e:
    gtts_error = str(e)

sr_available = False
sr_error = ""
try:
    import speech_recognition as sr
    sr_available = True
except Exception as e:
    sr_error = str(e)

webrtc_available = False
webrtc_error = ""
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, ClientSettings, VideoProcessorBase
    webrtc_available = True
except Exception as e:
    webrtc_error = str(e)

# SciPy WAV writer used in the call processor if available
scipy_wav_available = False
scipy_error = ""
try:
    import scipy.io.wavfile as wavfile
    scipy_wav_available = True
except Exception as e:
    scipy_error = str(e)

# OpenAI is optional (for generation); show friendly error if missing / key not set
openai_available = False
openai_error = ""
try:
    import openai
    openai_available = True
except Exception as e:
    openai_error = str(e)

# -------------------------
# Config & DB
# -------------------------
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
DB_FILE = "echosoul.db"
VAULT_KEY_FILE = "vault_key.key"
AUDIO_CACHE = "audio_cache"
os.makedirs(AUDIO_CACHE, exist_ok=True)

# initialize DB
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            when_date TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vault (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            encrypted_blob TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persona (
            id INTEGER PRIMARY KEY,
            persona_json TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# -------------------------
# Vault encryption helpers
# -------------------------
def ensure_vault_key():
    if os.path.exists(VAULT_KEY_FILE):
        with open(VAULT_KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(VAULT_KEY_FILE, "wb") as f:
        f.write(key)
    return key

FERNET_KEY = ensure_vault_key()
fernet = Fernet(FERNET_KEY)

def encrypt_blob(text: str) -> str:
    return fernet.encrypt(text.encode()).decode()

def decrypt_blob(token: str) -> str:
    try:
        return fernet.decrypt(token.encode()).decode()
    except Exception as e:
        return f"[decryption error: {e}]"

# -------------------------
# Persistence helpers
# -------------------------
def save_message(role: str, content: str):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)", (role, content, ts))
    conn.commit()

def get_messages(limit: int = 500):
    cur = conn.cursor()
    cur.execute("SELECT role, content, created_at FROM messages ORDER BY id ASC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

def add_timeline_event(title: str, description: str, when_date: str):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO timeline (title, description, when_date, created_at) VALUES (?, ?, ?, ?)", (title, description, when_date, ts))
    conn.commit()

def get_timeline(limit: int = 500):
    cur = conn.cursor()
    cur.execute("SELECT title, description, when_date, created_at FROM timeline ORDER BY when_date DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"title": r[0], "description": r[1], "when_date": r[2], "created_at": r[3]} for r in rows]

def vault_store(title: str, plaintext: str):
    token = encrypt_blob(plaintext)
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO vault (title, encrypted_blob, created_at) VALUES (?, ?, ?)", (title, token, ts))
    conn.commit()

def vault_list():
    cur = conn.cursor()
    cur.execute("SELECT id, title, created_at FROM vault ORDER BY id DESC")
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in cur.fetchall()]

def vault_get(v_id: int):
    cur = conn.cursor()
    cur.execute("SELECT encrypted_blob FROM vault WHERE id=?", (v_id,))
    row = cur.fetchone()
    if not row:
        return None
    return decrypt_blob(row[0])

def get_persona():
    cur = conn.cursor()
    cur.execute("SELECT persona_json FROM persona WHERE id=1")
    row = cur.fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except:
            return {}
    return {}

def save_persona(obj: Dict):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO persona (id, persona_json) VALUES (1, ?)", (json.dumps(obj),))
    conn.commit()

# -------------------------
# Lightweight emotion detection (TextBlob fallback)
# -------------------------
def detect_emotion(text: str):
    if not text:
        return {}
    try:
        tb = __import__("textblob").TextBlob(text)
        p = tb.sentiment.polarity
        if p > 0.3:
            return {"joy": float(0.6 + 0.4 * p)}
        elif p < -0.3:
            return {"sadness": float(0.6 - 0.4 * p)}
        else:
            return {"neutral": 0.9}
    except Exception:
        return {"neutral": 1.0}

# -------------------------
# TTS helper (gTTS -> MP3 bytes). No pydub import required.
# Returns mp3 bytes (Streamlit can play mp3 directly).
# -------------------------
def synthesize_tts_mp3_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    if not gtts_available:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        # gTTS has write_to_fp method; if not available in version, fallback to save temp file
        try:
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception:
            tmp_mp3 = os.path.join(AUDIO_CACHE, f"tts_{int(time.time()*1000)}.mp3")
            tts.save(tmp_mp3)
            with open(tmp_mp3, "rb") as f:
                data = f.read()
            try:
                os.remove(tmp_mp3)
            except:
                pass
            return data
    except Exception:
        return None

# -------------------------
# STT helper (if SpeechRecognition available)
# -------------------------
def stt_from_wav_bytes(wav_bytes: bytes) -> Optional[str]:
    if not sr_available:
        return None
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            path = tmp.name
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        try:
            os.remove(path)
        except:
            pass
        return text
    except Exception:
        return None

# -------------------------
# OpenAI wrapper (safe)
# -------------------------
def openai_chat(messages: list, model: str = "gpt-4o-mini", max_tokens: int = 600, temperature: float = 0.8):
    if not openai_available:
        return {"error": "openai_missing", "content": "OpenAI python package not installed."}
    # api key can be provided via Streamlit secrets or sidebar
    if not getattr(openai, "api_key", None) and os.getenv("OPENAI_API_KEY") is None:
        return {"error": "api_key_missing", "content": "OpenAI API key missing. Set OPENAI_API_KEY in secrets or enter in sidebar."}
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        txt = resp.choices[0].message["content"]
        return {"content": txt, "raw": resp}
    except Exception as e:
        return {"error": str(e), "content": f"OpenAI error: {e}"}

# -------------------------
# Optional WebRTC audio processor (only used if streamlit-webrtc is available)
# This collects audio frames in small chunks, writes a WAV via scipy (if available), runs STT, calls model, and enqueues TTS MP3 bytes.
# -------------------------
if webrtc_available:
    SAMPLE_RATE = 48000

    class EchoAudioProcessor(VideoProcessorBase):
        def __init__(self):
            self.in_q = queue.Queue()
            self.out_mp3_q = queue.Queue()
            self.running = True
            self.worker = threading.Thread(target=self._worker, daemon=True)
            self.worker.start()

        def _worker(self):
            # assemble ~1.4s chunks
            CHUNK_SECONDS = 1.4
            SAMPLES_NEEDED = int(CHUNK_SECONDS * SAMPLE_RATE)
            while self.running:
                try:
                    frames = []
                    total = 0
                    f = self.in_q.get(timeout=1.0)
                    frames.append(f)
                    total += f.shape[0]
                    while total < SAMPLES_NEEDED:
                        try:
                            f = self.in_q.get_nowait()
                            frames.append(f)
                            total += f.shape[0]
                        except queue.Empty:
                            break
                    if total == 0:
                        continue
                    arr = np.concatenate(frames, axis=0)
                    # convert float32 [-1,1] -> int16
                    if arr.dtype != np.int16:
                        int16 = (arr * 32767).astype(np.int16)
                    else:
                        int16 = arr
                    # write wav via scipy if available; else skip STT
                    wav_bytes = None
                    if scipy_wav_available:
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                wavfile.write(tmp.name, SAMPLE_RATE, int16)
                                tmp.flush()
                                with open(tmp.name, "rb") as f:
                                    wav_bytes = f.read()
                            try:
                                os.remove(tmp.name)
                            except:
                                pass
                        except Exception:
                            wav_bytes = None

                    # STT (if available)
                    stt_text = None
                    if wav_bytes and sr_available:
                        stt_text = stt_from_wav_bytes(wav_bytes)
                    if stt_text:
                        save_message("user", stt_text)
                        # generate reply via OpenAI if available
                        persona = get_persona()
                        system = {"role": "system", "content": "You are EchoSoul — short, empathetic voice replies."}
                        if persona:
                            system = {"role": "system", "content": f"Persona: {json.dumps(persona)}. " + system["content"]}
                        recent = get_messages(30)
                        msgs = [system]
                        for m in recent[-15:]:
                            msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})
                        msgs.append({"role": "user", "content": stt_text})
                        resp = openai_chat(msgs, max_tokens=250)
                        reply_text = resp.get("content", "(no reply)")
                        save_message("assistant", reply_text)
                        # synthesize TTS to MP3 (if available)
                        mp3 = synthesize_tts_mp3_bytes(reply_text, lang=st.session_state.get("current_voice", "en"))
                        if mp3:
                            self.out_mp3_q.put(mp3)
                    # small throttle
                    time.sleep(0.05)
                except queue.Empty:
                    continue
                except Exception:
                    time.sleep(0.2)
                    continue

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # Convert to mono float32
            arr = frame.to_ndarray()  # shape (channels, samples)
            if arr.ndim == 2:
                mono = arr.mean(axis=0)
            else:
                mono = arr
            # Convert to float32 in [-1,1]
            if mono.dtype == np.int16:
                float32 = mono.astype(np.float32) / 32767.0
            else:
                float32 = mono.astype(np.float32)
            try:
                self.in_q.put(float32, block=False)
            except queue.Full:
                pass
            return frame

        def stop(self):
            self.running = False
            try:
                self.worker.join(timeout=1.0)
            except:
                pass

# -------------------------
# Session defaults
# -------------------------
if "api_pin" not in st.session_state:
    st.session_state.api_pin = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_voice" not in st.session_state:
    st.session_state.current_voice = "en"
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Chat"

# -------------------------
# Sidebar & Navigation
# -------------------------
st.title("EchoSoul — Personal AI Companion")
with st.sidebar:
    st.image("https://placehold.co/200x200?text=EchoSoul", use_column_width=True)
    st.markdown("### Session / API")
    if not st.session_state.logged_in:
        api_pin = st.text_input("API PIN or OpenAI key (sk-...)", type="password")
        if api_pin:
            st.session_state.api_pin = api_pin
            # if user pasted a full OpenAI key, set environment for openai
            if api_pin.startswith("sk-") and openai_available:
                os.environ["OPENAI_API_KEY"] = api_pin
                openai.api_key = api_pin
                st.success("OpenAI key set for session.")
            else:
                st.info("API PIN stored in session for quick use.")
            st.session_state.logged_in = True
            st.experimental_rerun()
    else:
        st.markdown("✅ Session active")
        if st.button("Log out (clear session)"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()

    st.markdown("---")
    st.session_state.selected_page = st.radio("Navigation", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call (Live)", "About"], index=["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call (Live)", "About"].index(st.session_state.selected_page))
    st.markdown("---")
    voice_choice = st.selectbox("AI voice language", ["en (default)", "hi (Hindi)", "es (Spanish)"])
    if voice_choice.startswith("hi"):
        st.session_state.current_voice = "hi"
    elif voice_choice.startswith("es"):
        st.session_state.current_voice = "es"
    else:
        st.session_state.current_voice = "en"

# -------------------------
# Right column (UI refs & quick)
# -------------------------
left_col, right_col = st.columns([3,1])
with right_col:
    st.markdown("### UI reference images (optional)")
    uploaded = st.file_uploader("Upload up to 3 images", accept_multiple_files=True, type=["png","jpg","jpeg"])
    if uploaded:
        for i, f in enumerate(uploaded[:3]):
            st.image(f, caption=f"UI ref {i+1}", use_column_width=True)
    st.markdown("---")
    st.markdown("System & audio status")
    st.write("gTTS available:", gtts_available, " (error: " + (gtts_error or "none") + ")")
    st.write("SpeechRecognition available:", sr_available, " (error: " + (sr_error or "none") + ")")
    st.write("streamlit-webrtc available:", webrtc_available, " (error: " + (webrtc_error or "none") + ")")
    st.write("scipy.io.wavfile available:", scipy_wav_available, " (error: " + (scipy_error or "none") + ")")
    st.markdown("---")
    st.markdown("Quick exports")
    if st.button("Export timeline JSON"):
        st.download_button("Download timeline JSON", data=json.dumps(get_timeline(1000), indent=2), file_name="timeline.json", mime="application/json")

# -------------------------
# Pages
# -------------------------
with left_col:
    page = st.session_state.selected_page

    if page == "Chat":
        st.header("Chat with EchoSoul")
        st.markdown("Type and press Enter (input clears automatically).")

        # display messages
        for m in get_messages(500):
            who = "You" if m["role"] == "user" else "EchoSoul"
            t = m["created_at"][:19].replace("T", " ")
            st.markdown(f"**{who}** ({t}): {m['content']}")

        # Use st.chat_input when available to auto-clear
        try:
            user_input = st.chat_input("Say something to EchoSoul...")
        except Exception:
            # fallback for older streamlit: use text_input and manual clear
            if "chat_submitted" not in st.session_state:
                st.session_state.chat_submitted = False
            if st.session_state.chat_submitted:
                st.session_state.chat_value = ""
                st.session_state.chat_submitted = False
            user_input = st.text_input("Say something to EchoSoul...", key="chat_value")

        tts_play = st.checkbox("Play reply as voice (TTS)", value=False)
        life_sim = st.checkbox("Use Life-Path Simulation (what-if)", value=False)

        if user_input:
            save_message("user", user_input)
            # simple prompt to model
            persona = get_persona()
            system_msg = {"role": "system", "content": "You are EchoSoul — kind, concise, and helpful."}
            msgs = [system_msg]
            if persona:
                msgs.append({"role": "system", "content": f"Persona: {json.dumps(persona)}"})
            recent = get_messages(30)
            for m in recent[-20:]:
                msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})
            msgs.append({"role": "user", "content": user_input})
            if life_sim:
                msgs.append({"role": "user", "content": "Please present 3 plausible future outcomes with pros/cons (brief)."})
            resp = openai_chat(msgs)
            reply = resp.get("content", "Sorry, could not generate a reply.")
            save_message("assistant", reply)
            st.markdown("**EchoSoul:**")
            st.write(reply)
            if tts_play:
                mp3 = synthesize_tts_mp3_bytes(reply, lang=st.session_state.get("current_voice", "en"))
                if mp3:
                   
