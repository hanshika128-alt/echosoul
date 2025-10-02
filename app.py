# app.py - EchoSoul (corrected, streamlit-webrtc live-call included)
# Requirements snippet (put in requirements.txt):
# streamlit==1.50.0
# streamlit-webrtc>=0.54.0
# openai
# cryptography
# gtts
# pydub
# speechrecognition
# numpy
# pandas
# python-dotenv
# textblob
# scipy

import os
import io
import sys
import json
import time
import queue
import tempfile
import threading
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, ClientSettings, VideoProcessorBase
import av
import numpy as np
from cryptography.fernet import Fernet
import openai
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd

# Optional transformer-based emotion detection (not required)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -------------------------
# Basic config & init
# -------------------------
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
DB_FILE = "echosoul.db"
VAULT_KEY_FILE = "vault_key.key"
AUDIO_CACHE = "audio_cache"
os.makedirs(AUDIO_CACHE, exist_ok=True)

# Load OpenAI key if in secrets or env
if "OPENAI_API_KEY" in st.secrets:
    os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

# RTC config (STUN)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# -------------------------
# Database (SQLite)
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
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
    conn.commit()
    return conn

conn = init_db()

# -------------------------
# Vault encryption helpers
# -------------------------
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
def save_message(role: str, content: str, metadata: Optional[Dict]=None):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (role, content, created_at, metadata_json) VALUES (?, ?, ?, ?)",
                (role, content, ts, json.dumps(metadata or {})))
    conn.commit()
    return cur.lastrowid

def get_messages(limit=500):
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

def vault_store(title: str, plaintext: str):
    token = encrypt_blob(plaintext)
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
# NLP helpers
# -------------------------
if TRANSFORMERS_AVAILABLE:
    try:
        emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    except Exception:
        emotion_pipe = None
else:
    emotion_pipe = None

def detect_emotion(text: str):
    if not text:
        return {}
    if emotion_pipe:
        try:
            out = emotion_pipe(text[:512])
            if isinstance(out, list) and out and isinstance(out[0], list):
                return {item['label']: float(item['score']) for item in out[0]}
        except Exception:
            pass
    # fallback simple
    tb = TextBlob(text)
    p = tb.sentiment.polarity
    if p > 0.3:
        return {"joy": 0.6 + 0.4 * p}
    elif p < -0.3:
        return {"sadness": 0.6 - 0.4 * p}
    else:
        return {"neutral": 0.9}

# -------------------------
# OpenAI helper
# -------------------------
def openai_chat_completion(messages, model="gpt-4o-mini", max_tokens=600, temperature=0.8):
    if not getattr(openai, "api_key", None):
        # Inform but return fallback
        return {"error":"no_api_key","content":"OpenAI API key missing."}
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        text = resp.choices[0].message["content"]
        return {"content": text, "raw": resp}
    except Exception as e:
        return {"error": str(e), "content": f"OpenAI error: {e}"}

# -------------------------
# TTS & STT helpers (gTTS + pydub)
# -------------------------
def synthesize_tts_wav_bytes(text: str, lang="en") -> bytes:
    try:
        tmp_mp3 = os.path.join(AUDIO_CACHE, f"tts_{int(time.time()*1000)}.mp3")
        tmp_wav = os.path.join(AUDIO_CACHE, f"tts_{int(time.time()*1000)}.wav")
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_mp3)
        audio = AudioSegment.from_file(tmp_mp3, format="mp3")
        audio.export(tmp_wav, format="wav")
        with open(tmp_wav, "rb") as f:
            b = f.read()
        try:
            os.remove(tmp_mp3)
            os.remove(tmp_wav)
        except:
            pass
        return b
    except Exception as e:
        st.error(f"TTS error: {e}")
        return b""

def stt_from_wav_bytes(wav_bytes: bytes) -> str:
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            tmppath = tmp.name
        with sr.AudioFile(tmppath) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        try:
            os.remove(tmppath)
        except:
            pass
        return text
    except Exception as e:
        return f"[STT error: {e}]"

# -------------------------
# WebRTC audio processor
# -------------------------
class EchoAudioProcessor(VideoProcessorBase):
    def __init__(self):
        self.in_q = queue.Queue()
        self.out_wav_q = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        SAMPLE_RATE = 48000
        CHUNK_SECONDS = 1.4
        SAMPLES_NEEDED = int(SAMPLE_RATE * CHUNK_SECONDS)
        while self.running:
            try:
                frames = []
                total = 0
                # block for first frame
                frame = self.in_q.get(timeout=1.0)
                frames.append(frame)
                total += frame.shape[0]
                # gather more if available
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
                # convert float32 [-1,1] to int16
                int16 = (arr * 32767).astype(np.int16)
                # write wav file and STT
                import scipy.io.wavfile as wavfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wavfile.write(tmp.name, SAMPLE_RATE, int16)
                    tmp.flush()
                    with open(tmp.name, "rb") as f:
                        wav_bytes = f.read()
                stt_text = stt_from_wav_bytes(wav_bytes)
                if stt_text and not stt_text.startswith("[STT error"):
                    save_message("user", stt_text, metadata={"via":"call","ts":datetime.utcnow().isoformat()})
                    emo = detect_emotion(stt_text)
                    persona = get_persona()
                    system = {"role":"system","content":"You are EchoSoul — quick, empathetic, helpful. Keep replies short for a live call."}
                    if persona:
                        system = {"role":"system","content": f"Persona: {json.dumps(persona)}. " + system["content"]}
                    recent = get_messages(30)
                    msgs = [system]
                    for m in recent[-15:]:
                        msgs.append({"role": "user" if m["role"]=="user" else "assistant", "content": m["content"]})
                    msgs.append({"role":"user","content": stt_text})
                    res = openai_chat_completion(msgs, max_tokens=250)
                    reply = res.get("content", "(no reply)")
                    save_message("assistant", reply, metadata={"via":"call","ts":datetime.utcnow().isoformat(), "emotion":emo})
                    wav = synthesize_tts_wav_bytes(reply, lang=st.session_state.get("current_voice", "en"))
                    if wav:
                        self.out_wav_q.put(wav)
                try:
                    os.remove(tmp.name)
                except:
                    pass
            except queue.Empty:
                continue
            except Exception as e:
                print("Processor worker error:", e, file=sys.stderr)
                time.sleep(0.5)
                continue

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        if arr.ndim == 2:
            mono = arr.mean(axis=0)
        else:
            mono = arr
        # convert to float32 [-1,1]
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
# Session state defaults
# -------------------------
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

# -------------------------
# Sidebar UI
# -------------------------
st.title("EchoSoul — Personal AI Companion (Live-call enabled)")
with st.sidebar:
    st.image("https://placehold.co/200x200?text=EchoSoul", use_column_width=True)
    st.markdown("### API / Session")
    if not st.session_state.logged_in:
        api_input = st.text_input("API PIN or OpenAI key (sk-...)", type="password")
        if api_input:
            st.session_state.api_pin = api_input
            if api_input.startswith("sk-"):
                openai.api_key = api_input
                st.success("OpenAI key set for session.")
            else:
                st.info("API PIN saved for session (vault protector).")
            st.session_state.logged_in = True
            st.experimental_rerun()
    else:
        st.write("✅ Session active")
        if st.button("Log out (clear session)"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()
    st.markdown("---")
    nav = st.radio("Navigation", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call (Live)", "About"], index=["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call (Live)", "About"].index(st.session_state.selected_page))
    st.session_state.selected_page = nav
    st.markdown("---")
    voice_choice = st.selectbox("AI voice language", ["en (default)", "hi (Hindi)", "es (Spanish)"])
    if voice_choice.startswith("hi"):
        st.session_state.current_voice = "hi"
    elif voice_choice.startswith("es"):
        st.session_state.current_voice = "es"
    else:
        st.session_state.current_voice = "en"

# -------------------------
# Layout: main + right column
# -------------------------
left, right = st.columns([3,1])

with right:
    st.markdown("### UI references (optional)")
    refs = st.file_uploader("Upload up to 3 images to guide UI", accept_multiple_files=True, type=["png","jpg","jpeg"])
    if refs:
        for i, r in enumerate(refs[:3]):
            st.image(r, caption=f"ref {i+1}", use_column_width=True)
    st.markdown("---")
    if st.button("Export timeline JSON"):
        data = get_timeline(1000)
        st.download_button("Download timeline JSON", data=json.dumps(data, indent=2), file_name="timeline.json", mime="application/json")

with left:
    page = st.session_state.selected_page

    # -------------------------
    # Chat page (uses st.chat_input)
    # -------------------------
    if page == "Chat":
        st.header("Chat with EchoSoul")
        st.markdown("Type in the box below and press Enter. Input auto-clears after sending.")
        messages = get_messages(500)
        # Display messages
        for m in messages:
            t = m["created_at"][:19].replace("T", " ")
            if m["role"] == "user":
                st.markdown(f"**You** ({t}): {m['content']}")
            else:
                st.markdown(f"**EchoSoul** ({t}): {m['content']}")

        # Use st.chat_input (clears automatically)
        user_input = st.chat_input("Say something to EchoSoul...")
        tts_opt = st.checkbox("Play reply as voice (TTS)", value=False)
        life_sim = st.checkbox("Life-Path Simulation (what-if)", value=False)

        if user_input:
            # Save user message
            save_message("user", user_input, metadata={"via":"chat", "ts": datetime.utcnow().isoformat()})
            # Build prompt
            persona = get_persona()
            system_msg = {"role":"system", "content":"You are EchoSoul — kind, reflective, and briefly helpful."}
            msgs_for_api = [system_msg]
            if persona:
                msgs_for_api.append({"role":"system", "content": f"Persona: {json.dumps(persona)}"})
            recent = get_messages(30)
            for m in recent[-20:]:
                msgs_for_api.append({"role":"user" if m["role"]=="user" else "assistant", "content": m["content"]})
            msgs_for_api.append({"role":"user", "content": user_input})
            if life_sim:
                msgs_for_api.append({"role":"user", "content":"Please give 3 plausible future outcomes and pros/cons briefly."})
            with st.spinner("EchoSoul is thinking..."):
                resp = openai_chat_completion(msgs_for_api)
            reply = resp.get("content", "Sorry, I couldn't generate a reply.")
            save_message("assistant", reply, metadata={"from":"openai", "ts": datetime.utcnow().isoformat()})
            st.markdown("**EchoSoul:**")
            st.write(reply)
            if tts_opt:
                wavb = synthesize_tts_wav_bytes(reply, lang=st.session_state.get("current_voice","en"))
                if wavb:
                    st.audio(wavb, format="audio/wav")
            # update persona excerpts
            per = get_persona()
            per.setdefault("excerpts", [])
            per["excerpts"].append({"text": user_input, "ts": datetime.utcnow().isoformat()})
            per["excerpts"] = per["excerpts"][-300:]
            save_persona(per)
            # do NOT set st.session_state.chat_input (we are using st.chat_input which clears automatically)

    # -------------------------
    # Chat history
    # -------------------------
    elif page == "Chat history":
        st.header("Chat history & search")
        msgs = get_messages(5000)
        df = pd.DataFrame([{"id":m["id"], "role":m["role"], "content":m["content"], "created_at":m["created_at"]} for m in msgs])
        st.dataframe(df)
        q = st.text_input("Search messages (contains)")
        if q:
            res = df[df['content'].str.contains(q, case=False, na=False)]
            st.dataframe(res)

    # -------------------------
    # Life timeline
    # -------------------------
    elif page == "Life timeline":
        st.header("Life Timeline")
        with st.form("add_event"):
            title = st.text_input("Title")
            when = st.date_input("When")
            desc = st.text_area("Description")
            tags = st.text_input("Tags (comma separated)")
            submit = st.form_submit_button("Add event")
            if submit and title.strip():
                add_timeline_event(title, desc, when.isoformat(), tags=tags)
                st.success("Added event")
                st.experimental_rerun()
        timeline = get_timeline(500)
        if not timeline:
            st.info("No timeline events yet.")
        else:
            for ev in timeline:
                st.markdown(f"**{ev['title']}** — *{ev['when_date'][:10]}*")
                st.write(ev['description'])
                if ev['tags']:
                    st.caption("Tags: " + ev['tags'])
                st.markdown("---")

    # -------------------------
    # V
