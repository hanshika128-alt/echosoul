# EchoSoul - Full Corrected app.py
# Fixes:
#  1. Chat input clearing bug
#  2. st.download_button parentheses error

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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
from cryptography.fernet import Fernet
import openai
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd

# -------------------------
# Basic config
# -------------------------
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
DB_FILE = "echosoul.db"
VAULT_KEY_FILE = "vault_key.key"
AUDIO_CACHE = "audio_cache"
os.makedirs(AUDIO_CACHE, exist_ok=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------------
# Database setup
# -------------------------
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
    conn.commit()
    return conn

conn = init_db()

# -------------------------
# Vault encryption
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
# Persistence
# -------------------------
def save_message(role: str, content: str):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
        (role, content, ts)
    )
    conn.commit()

def get_messages(limit=200):
    cur = conn.cursor()
    cur.execute("SELECT role, content, created_at FROM messages ORDER BY id ASC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

def add_timeline_event(title, description, when_date):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO timeline (title, description, when_date, created_at) VALUES (?, ?, ?, ?)",
        (title, description, when_date, ts)
    )
    conn.commit()

def get_timeline(limit=200):
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

def vault_get(id):
    cur = conn.cursor()
    cur.execute("SELECT encrypted_blob FROM vault WHERE id=?", (id,))
    row = cur.fetchone()
    if not row:
        return None
    return decrypt_blob(row[0])

# -------------------------
# TTS / STT
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
        os.remove(tmp_mp3)
        os.remove(tmp_wav)
        return b
    except Exception as e:
        st.error(f"TTS error: {e}")
        return b""

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("EchoSoul Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"])

# -------------------------
# Pages
# -------------------------
if page == "Chat":
    st.header("Chat with EchoSoul")

    # Chat display
    messages = get_messages(200)
    for m in messages:
        who = "You" if m["role"] == "user" else "EchoSoul"
        t = m["created_at"][:19].replace("T", " ")
        st.markdown(f"**{who}** ({t}): {m['content']}")

    # FIX: safe chat input
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "chat_submitted" not in st.session_state:
        st.session_state.chat_submitted = False

    if st.session_state.chat_submitted:
        st.session_state.chat_input = ""
        st.session_state.chat_submitted = False

    user_input = st.text_input("Type your message...", key="chat_input")

    if st.button("Send"):
        if user_input.strip():
            save_message("user", user_input)
            # Placeholder AI reply
            save_message("assistant", f"(EchoSoul replies to: {user_input})")
            st.session_state.chat_submitted = True
            st.rerun()

elif page == "Chat history":
    st.header("Chat history")
    msgs = get_messages(500)
    df = pd.DataFrame(msgs)
    st.dataframe(df)

elif page == "Life timeline":
    st.header("Life Timeline")
    with st.form("add_event"):
        title = st.text_input("Title")
        when = st.date_input("When")
        desc = st.text_area("Description")
        submit = st.form_submit_button("Add")
        if submit and title.strip():
            add_timeline_event(title, desc, when.isoformat())
            st.success("Added event")
            st.rerun()
    for ev in get_timeline(200):
        st.markdown(f"**{ev['title']}** — *{ev['when_date']}*")
        st.write(ev['description'])
        st.markdown("---")

elif page == "Vault":
    st.header("Vault (encrypted)")
    with st.form("vault_form"):
        vt = st.text_input("Title")
        vb = st.text_area("Secret / memory")
        add = st.form_submit_button("Save to Vault")
        if add and vt and vb:
            vault_store(vt, vb)
            st.success("Saved to Vault")
            st.rerun()
    entries = vault_list()
    for e in entries:
        cols = st.columns([3,1])
        cols[0].write(f"{e['id']}: {e['title']} ({e['created_at'][:19]})")
        if cols[1].button("Reveal", key=f"rev_{e['id']}"):
            content = vault_get(e['id'])
            st.text_area("Decrypted content", value=content, height=150)

elif page == "Export":
    st.header("Export data")
    data = {"messages": get_messages(500), "timeline": get_timeline(200)}
    st.download_button("Download full JSON", data=json.dumps(data, indent=2), file_name="echosoul_export.json", mime="application/json")

    # FIXED download button
    timeline = get_timeline(200)
    if timeline:
        df = pd.DataFrame(timeline)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "Download timeline.csv",
            data=buf.getvalue(),
            file_name="timeline.csv",
            mime="text/csv"
        )

elif page == "Brain mimic":
    st.header("Brain Mimic")
    st.info("Placeholder: mimic user’s style using past chats.")

elif page == "Call":
    st.header("Call (Live via WebRTC)")
    st.info("Live call feature placeholder here (streamlit-webrtc).")

elif page == "About":
    st.header("About EchoSoul")
    st.write("EchoSoul is your AI companion with memory, vault, timeline, and live call features.")
