# app.py - EchoSoul (full)
# Live voice call (in-app) using streamlit-webrtc, Chat + Memory + Vault + Brain Mimic
# Date: 2025-10-02
# Requirements (example requirements.txt entries):
# streamlit>=1.25
# streamlit-webrtc>=0.54.0
# openai>=0.27.0
# sqlalchemy
# cryptography
# gtts
# pydub
# speechrecognition
# numpy
# pandas
# python-dotenv
# textblob
# transformers  # optional
# torch         # optional (for transformers)
#
# NOTE: pydub requires ffmpeg available in the environment for mp3->wav conversions.
# On Streamlit Cloud you might need to provide ffmpeg via package or use TTS API that returns wav.
# Also ensure streamlit-webrtc dependencies install correctly in your environment.

import os
import io
import sys
import json
import time
import queue
import base64
import tempfile
import threading
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
from cryptography.fernet import Fernet
import openai
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd

# Optional transformers-based emotion detection
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# --------------------
# Config / constants
# --------------------
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
DB_FILE = "echosoul.db"
VAULT_KEY_FILE = "vault_key.key"
AUDIO_CACHE_DIR = "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Read OpenAI key from secrets or env
OPENAI_ENV_NAME = "OPENAI_API_KEY"
if OPENAI_ENV_NAME in st.secrets:
    os.environ.setdefault("OPENAI_API_KEY", st.secrets[OPENAI_ENV_NAME])
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

# For WebRTC, we may provide STUN servers for better connectivity:
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --------------------
# DB init
# --------------------
def init_db():
    c = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = c.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_profile (
        id INTEGER PRIMARY KEY,
        display_name TEXT,
        meta_json TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        created_at TEXT,
        metadata_json TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        when_date TEXT,
        tags TEXT,
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
    c.commit()
    return c

conn = init_db()

# --------------------
# Encryption (Vault)
# --------------------
def ensure_vault_key():
    if os.path.exists(VAULT_KEY_FILE):
        with open(VAULT_KEY_FILE, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(VAULT_KEY_FILE, "wb") as f:
            f.write(key)
    return key

FERNET_KEY = ensure_vault_key()
fernet = Fernet(FERNET_KEY)

def encrypt_text(plain: str) -> str:
    return fernet.encrypt(plain.encode()).decode()

def decrypt_text(token: str) -> str:
    try:
        return fernet.decrypt(token.encode()).decode()
    except Exception as e:
        return f"[decryption error: {e}]"

# --------------------
# Persistence helpers
# --------------------
def save_message(role:str, content:str, metadata:Optional[Dict]=None):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (role, content, created_at, metadata_json) VALUES (?, ?, ?, ?)",
                (role, content, ts, json.dumps(metadata or {})))
    conn.commit()
    return cur.lastrowid

def get_messages(limit:int=1000):
    cur = conn.cursor()
    cur.execute("SELECT id, role, content, created_at, metadata_json FROM messages ORDER BY id ASC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"id":r[0],"role":r[1],"content":r[2],"created_at":r[3],"metadata":json.loads(r[4] or "{}")} for r in rows]

def add_timeline_event(title, description, when_date, tags=""):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO timeline (title, description, when_date, tags, created_at) VALUES (?, ?, ?, ?, ?)",
                (title, description, when_date, tags, ts))
    conn.commit()
    return cur.lastrowid

def get_timeline(limit=500):
    cur = conn.cursor()
    cur.execute("SELECT id, title, description, when_date, tags, created_at FROM timeline ORDER BY when_date DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"id":r[0],"title":r[1],"description":r[2],"when_date":r[3],"tags":r[4],"created_at":r[5]} for r in rows]

def vault_store(title, plaintext):
    token = encrypt_text(plaintext)
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO vault (title, encrypted_blob, created_at) VALUES (?, ?, ?)", (title, token, ts))
    conn.commit()
    return cur.lastrowid

def vault_list():
    cur = conn.cursor()
    cur.execute("SELECT id, title, created_at FROM vault ORDER BY id DESC")
    return [{"id":r[0],"title":r[1],"created_at":r[2]} for r in cur.fetchall()]

def vault_get(id):
    cur = conn.cursor()
    cur.execute("SELECT encrypted_blob FROM vault WHERE id=?", (id,))
    row = cur.fetchone()
    if not row: return None
    return decrypt_text(row[0])

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

def save_persona(obj):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO persona (id, persona_json) VALUES (1, ?)", (json.dumps(obj),))
    conn.commit()

# --------------------
# NLP helpers
# --------------------
if TRANSFORMERS_AVAILABLE:
    try:
        emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    except Exception:
        emotion_pipe = None
else:
    emotion_pipe = None

def detect_emotion_text(text: str):
    if not text: return {}
    if emotion_pipe:
        try:
            out = emotion_pipe(text[:512])
            # transform output to label:score map
            if isinstance(out, list) and len(out) and isinstance(out[0], list):
                res = {x['label']:float(x['score']) for x in out[0]}
                return res
        except Exception:
            pass
    # fallback: TextBlob polarity heuristic
    tb = TextBlob(text)
    pol = tb.sentiment.polarity
    if pol > 0.3:
        return {"joy": 0.6 + 0.4*pol, "neutral": 0.4 - 0.4*pol}
    if pol < -0.3:
        return {"sadness": 0.6 - 0.4*pol, "anger":0.3}
    return {"neutral": 0.8}

# --------------------
# OpenAI helper
# --------------------
def openai_chat(messages:list, model="gpt-4o-mini", max_tokens=900, temperature=0.8):
    if not getattr(openai, "api_key", None):
        st.error("OpenAI API key not found. Set it in Streamlit secrets or environment variable OPENAI_API_KEY.")
        return {"error":"no_api","content":"Missing API key"}
    try:
        # Use ChatCompletion
        resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        text = resp.choices[0].message["content"]
        return {"content":text, "raw":resp}
    except Exception as e:
        return {"error": str(e), "content": f"OpenAI error: {e}"}

# --------------------
# TTS helper (gTTS -> wav)
# --------------------
def synthesize_tts_wav(text:str, lang="en") -> bytes:
    """
    Return WAV bytes synthesized from text using gTTS and pydub. Use temporary files.
    """
    try:
        mp3_path = os.path.join(AUDIO_CACHE_DIR, f"tts_{int(time.time()*1000)}.mp3")
        wav_path = os.path.join(AUDIO_CACHE_DIR, f"tts_{int(time.time()*1000)}.wav")
        tts = gTTS(text=text, lang=lang)
        tts.save(mp3_path)
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio.export(wav_path, format="wav")
        with open(wav_path, "rb") as f:
            b = f.read()
        # cleanup
        try:
            os.remove(mp3_path)
            os.remove(wav_path)
        except:
            pass
        return b
    except Exception as e:
        st.error(f"TTS error: {e}")
        return b""

# --------------------
# STT helper (SpeechRecognition wav file -> text)
# --------------------
def stt_from_wav_bytes(wav_bytes: bytes) -> str:
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            tmp_name = tmp.name
        with sr.AudioFile(tmp_name) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        try:
            os.remove(tmp_name)
        except:
            pass
        return text
    except Exception as e:
        return f"[STT error: {e}]"

# --------------------
# WebRTC Processor - handles live audio stream
# --------------------
class EchoSoulAudioProcessor(VideoProcessorBase):
    """
    This processor receives audio frames from browser, collects them (small frames),
    converts to WAV bytes, sends to STT and generative model, and queues TTS audio to play back.
    We'll maintain thread-safe queues:
    - in_audio_queue: incoming raw audio frames (numpy float32)
    - out_wav_queue: WAV bytes produced by TTS to be played back to the client via webrtc audio
    """

    def __init__(self):
        self.in_audio_q = queue.Queue()
        self.out_wav_q = queue.Queue()
        self.buffer_lock = threading.Lock()
        self.running = True
        # Start background worker thread for processing small chunks to speech->text->reply->tts
        self.worker = threading.Thread(target=self._background_worker, daemon=True)
        self.worker.start()

    def _background_worker(self):
        """
        Consume audio frames from in_audio_q, assemble into short clips (~1.5s), run STT and generate replies.
        """
        # We'll assemble frames into 1.5s chunks based on sample rate
        CHUNK_SECONDS = 1.5
        FRAME_RATE = 48000  # webrtc default for audio
        SAMPLES_NEEDED = int(CHUNK_SECONDS * FRAME_RATE)
        while self.running:
            try:
                # collect frames until we have enough samples or timeout
                samples = []
                total = 0
                # try to gather at least one frame (block)
                frame = self.in_audio_q.get(timeout=1.0)
                samples.append(frame)
                total += frame.shape[0]
                # nonblocking gather until we have enough or brief pause
                while total < SAMPLES_NEEDED:
                    try:
                        frame = self.in_audio_q.get_nowait()
                        samples.append(frame)
                        total += frame.shape[0]
                    except queue.Empty:
                        break
                if total == 0:
                    continue
                # concatenate samples
                audio_np = np.concatenate(samples, axis=0)
                # convert float32 [-1,1] to int16 PCM
                int16 = (audio_np * 32767).astype(np.int16)
                # write to wav bytes
                out = io.BytesIO()
                # av.AudioFrame to write? We'll use pydub to convert raw data
                from scipy.io import wavfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wavfile.write(tmp.name, FRAME_RATE, int16)
                    tmp.flush()
                    with open(tmp.name, "rb") as f:
                        wav_bytes = f.read()
                # STT
                stt_text = stt_from_wav_bytes(wav_bytes)
                # If stt produced a meaningful text, pass to model
                if stt_text and not stt_text.startswith("[STT error"):
                    # Save message
                    save_message("user", stt_text, metadata={"via":"call","ts":datetime.utcnow().isoformat()})
                    # Emotion detection (text)
                    emo = detect_emotion_text(stt_text)
                    # Build messages for model
                    persona = get_persona()
                    system_msg = {"role":"system","content":"You are EchoSoul, an empathetic assistant. Use short replies suitable for a real-time call. Refer to user's memories when helpful."}
                    if persona:
                        system_msg = {"role":"system","content":f"Persona: {json.dumps(persona)}. {system_msg['content']}"}
                    recent = get_messages(30)
                    msgs = [system_msg]
                    for m in recent[-20:]:
                        msgs.append({"role": "user" if m["role"]=="user" else "assistant", "content": m["content"]})
                    msgs.append({"role":"user","content": stt_text})
                    # call OpenAI
                    resp = openai_chat(msgs, max_tokens=300, temperature=0.7)
                    reply_text = resp.get("content", "(no reply)")
                    save_message("assistant", reply_text, metadata={"via":"call","ts":datetime.utcnow().isoformat(), "emotion":emo})
                    # Synthesize TTS
                    wav_reply = synthesize_tts_wav(reply_text, lang=st.session_state.get("current_voice", "en"))
                    # Enqueue wav bytes to out queue
                    if wav_reply:
                        self.out_wav_q.put(wav_reply)
                # small sleep to yield
                time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                # avoid crashing the worker
                print("Worker error:", e, file=sys.stderr)
                time.sleep(0.5)
                continue

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """
        Called each time we receive an audio frame from the browser (user microphone).
        We'll convert to numpy and push into in_audio_q.
        Also, if out_wav_q has audio, we will schedule playback by returning a frame created from playback data.
        The webrtc_streamer manages mixing audio sources; here we just mirror input to output but also may inject TTS frames.
        """
        # Convert frame to np array
        frame_ndarray = frame.to_ndarray()  # shape: (channels, samples)
        # Convert to mono by averaging channels
        if frame_ndarray.ndim == 2:
            mono = frame_ndarray.mean(axis=0)
        else:
            mono = frame_ndarray
        # Convert int16 to float32 in [-1,1]
        if mono.dtype == np.int16:
            float32 = (mono.astype(np.float32) / 32767.0)
        elif mono.dtype == np.float32:
            float32 = mono
        else:
            float32 = mono.astype(np.float32)
        # push into queue
        try:
            self.in_audio_q.put(float32, block=False)
        except queue.Full:
            pass

        # If we have TTS bytes to play, we could inject audio. However, VideoProcessorBase.recv returns a frame to send back to the client.
        # The recommended approach: let WebRTC handle outgoing audio with a separate track. streamlit-webrtc does not allow us to push raw wav bytes directly here easily.
        # So we'll simply pass through audio (echo) — the TTS playback will be handled via st.audio or saved file playback on the client side.
        return frame

    def stop(self):
        self.running = False
        try:
            self.worker.join(timeout=1.0)
        except:
            pass

# --------------------
# UI: Sidebar and navigation
# --------------------
if "api_pin" not in st.session_state:
    st.session_state.api_pin = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_voice" not in st.session_state:
    st.session_state.current_voice = "en"

if "call_active" not in st.session_state:
    st.session_state.call_active = False

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Chat"

st.title("EchoSoul — live, adaptive personal AI")
st.caption("Persistent memory, adaptive persona, live calls, timeline, vault, and brain mimic.")

with st.sidebar:
    st.image("https://placehold.co/200x200?text=EchoSoul", use_column_width=True)
    st.markdown("### Session & API")
    if not st.session_state.logged_in:
        api_pin = st.text_input("API PIN or OpenAI key (store in session)", type="password")
        if api_pin:
            st.session_state.api_pin = api_pin
            # if user supplied full OpenAI key starting with sk-, set it
            if api_pin.startswith("sk-"):
                openai.api_key = api_pin
                st.success("OpenAI API key set for session.")
            else:
                st.info("API PIN stored in session to protect vault (session only).")
            st.session_state.logged_in = True
            st.experimental_rerun()
    else:
        st.markdown("✅ Session active")
        if st.button("Log out (clear session)"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()

    st.markdown("---")
    nav = st.radio("Navigation", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call (Live)", "About"], index=["Chat","Chat history","Life timeline","Vault","Export","Brain mimic","Call (Live)","About"].index(st.session_state.selected_page))
    st.session_state.selected_page = nav
    st.markdown("---")
    st.markdown("**Voice selection**")
    voice_choice = st.selectbox("Voice / language", ["en (default)", "hi (Hindi)", "es (Spanish)"], index=0)
    if voice_choice.startswith("hi"):
        st.session_state.current_voice = "hi"
    elif voice_choice.startswith("es"):
        st.session_state.current_voice = "es"
    else:
        st.session_state.current_voice = "en"

# --------------------
# Main area layout
# --------------------
left_col, right_col = st.columns([3,1])

# Right column: always visible quick controls & uploaded UI refs
with right_col:
    st.markdown("### UI reference (optional)")
    ui_files = st.file_uploader("Upload reference images (3 max)", accept_multiple_files=True, type=["png","jpg","jpeg"])
    if ui_files:
        for i, f in enumerate(ui_files[:3]):
            st.image(f, use_column_width=True, caption=f"ref {i+1}")

    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Export timeline JSON"):
        data = get_timeline(1000)
        st.download_button("Download", data=json.dumps(data, indent=2), file_name="timeline.json", mime="application/json")

# Left column: pages
with left_col:
    page = st.session_state.selected_page

    if page == "Chat":
        st.header("Chat with EchoSoul")
        st.markdown("Type a message and press Send. Input clears automatically after sending.")
        # show recent messages
        msgs = get_messages(500)
        chat_box = st.container()
        with chat_box:
            for m in msgs:
                ts = m["created_at"][:19].replace("T"," ")
                if m["r
