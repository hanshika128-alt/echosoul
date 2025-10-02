# app.py - EchoSoul (Streamlit)
# A single-file Streamlit app implementing chat, memory, timeline, vault (encrypted), call simulation, brain mimic, export, etc.
# Replace / adapt TTS and LLM provider calls with your preferred services & keys.

import streamlit as st
from datetime import datetime
import sqlite3
import json
import os
import hashlib
from cryptography.fernet import Fernet, InvalidToken
import base64
import pandas as pd
from dateutil import parser as dateparser
import requests
import time
import uuid

# Optional imports handled gracefully
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# Optional TTS libs
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# --- Constants & Paths ---
DB_PATH = "echosoul.db"
VAULT_PATH = "vault.enc"
MEMORY_TABLE = "memories"
CHAT_TABLE = "chats"
TIMELINE_TABLE = "timeline"

st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")

# --- Utility functions ---

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {CHAT_TABLE} (
        id TEXT PRIMARY KEY,
        role TEXT,
        content TEXT,
        timestamp TEXT
    )""")
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
        id TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        tags TEXT,
        timestamp TEXT
    )""")
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {TIMELINE_TABLE} (
        id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        datetime TEXT,
        tags TEXT
    )""")
    conn.commit()
    return conn

conn = init_db()

def sql_insert(table, data: dict):
    keys = ", ".join(data.keys())
    qs = ", ".join(["?"]*len(data))
    sql = f"INSERT INTO {table} ({keys}) VALUES ({qs})"
    conn.execute(sql, tuple(data.values()))
    conn.commit()

def sql_query_all(table, order_by="timestamp", desc=True):
    order = "DESC" if desc else "ASC"
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY {order_by} {order}")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

def sql_delete_all(table):
    conn.execute(f"DELETE FROM {table}")
    conn.commit()

# --- Vault encryption helpers ---
def derive_key_from_password(password: str) -> bytes:
    # Create a key for Fernet using SHA256 -> 32 bytes -> base64
    h = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(h)

def vault_exists():
    return os.path.exists(VAULT_PATH)

def save_vault(obj: dict, password: str):
    key = derive_key_from_password(password)
    f = Fernet(key)
    data = json.dumps(obj).encode()
    token = f.encrypt(data)
    with open(VAULT_PATH, "wb") as fh:
        fh.write(token)

def load_vault(password: str):
    if not vault_exists():
        return {}
    key = derive_key_from_password(password)
    f = Fernet(key)
    with open(VAULT_PATH, "rb") as fh:
        token = fh.read()
    try:
        data = f.decrypt(token)
    except InvalidToken:
        raise ValueError("Invalid vault password")
    return json.loads(data.decode())

# --- Emotion recognition (simple) ---
def detect_emotion_from_text(text: str):
    if VADER_AVAILABLE:
        sid = SentimentIntensityAnalyzer()
        s = sid.polarity_scores(text)
        # Use compound for rough emotion polarity; map to simple categories
        comp = s['compound']
        if comp >= 0.5:
            return "happy", s
        elif comp <= -0.5:
            return "sad/angry", s
        elif -0.5 < comp < 0.1:
            return "neutral", s
        else:
            return "mixed", s
    else:
        # fallback simple heuristic
        low = ['sad','angry','upset','hate','hurt','annoyed','depressed']
        high = ['happy','excited','love','yay','joy','awesome','great']
        t = text.lower()
        if any(w in t for w in high):
            return "happy", {"compound":0.6}
        if any(w in t for w in low):
            return "sad/angry", {"compound":-0.6}
        return "neutral", {"compound":0.0}

# --- OpenAI helpers (optional) ---
def set_openai_key(key: str):
    if not OPENAI_AVAILABLE:
        return
    openai.api_key = key

def chat_with_llm(prompt: str, system_prompt: str = None, model="gpt-4o-mini", max_tokens=512):
    """
    Uses OpenAI Chat Completion if available and key provided. Falls back to simple echo.
    """
    try:
        if OPENAI_AVAILABLE and hasattr(openai, "ChatCompletion") or hasattr(openai, "chat"):
            # Newer openai clients use openai.chat.completions.create or openai.ChatCompletion.create
            # Try ChatCompletion API
            messages = []
            if system_prompt:
                messages.append({"role":"system","content":system_prompt})
            messages.append({"role":"user","content":prompt})
            # Try both methods depending on openai version
            try:
                resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.8)
                text = resp.choices[0].message.content
                return text
            except Exception:
                # fallback to chat.completions
                resp = openai.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.8)
                text = resp.choices[0].message.content
                return text
        else:
            raise RuntimeError("OpenAI not configured")
    except Exception as e:
        # fallback basic echoing behaviour
        return f"(offline fallback) EchoSoul received: {prompt[:200]}"

# --- TTS helpers (simple) ---
def generate_tts(text: str, voice: str = "default", api_key: str = None):
    """
    Returns a tuple (success_bool, bytes_audio, mime) where bytes_audio is raw audio bytes (mp3/wav).
    This function attempts (in decreasing priority):
      - OpenAI TTS (if configured)
      - ElevenLabs (if configured & you add code)
      - gTTS fallback (if installed)
      - Otherwise returns None
    NOTE: You must adapt / provide credentials for production TTS.
    """
    # Attempt gTTS fallback (mp3)
    if GTTS_AVAILABLE:
        try:
            tts = gTTS(text=text)
            out_path = f"/tmp/tts_{uuid.uuid4().hex}.mp3"
            tts.save(out_path)
            with open(out_path, "rb") as fh:
                b = fh.read()
            os.remove(out_path)
            return True, b, "audio/mp3"
        except Exception:
            pass

    # If OpenAI is available and supports TTS in this environment, you could implement it here.
    # For now fallback to failure.
    return False, None, None

# --- Business logic: memory, chats, timeline, brain-mimic prompts ---

def add_chat(role: str, content: str):
    item = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    sql_insert(CHAT_TABLE, item)

def get_chats(limit=200):
    return sql_query_all(CHAT_TABLE, order_by="timestamp", desc=False)  # oldest -> newest

def add_memory(title: str, content: str, tags: list=None):
    item = {
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "tags": ",".join(tags or []),
        "timestamp": datetime.utcnow().isoformat()
    }
    sql_insert(MEMORY_TABLE, item)

def get_memories():
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {MEMORY_TABLE} ORDER BY timestamp DESC")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

def add_timeline_event(title: str, description: str, dtstr: str):
    # dtstr should be ISO or parseable form
    try:
        dt = dateparser.parse(dtstr)
        dt_iso = dt.isoformat()
    except Exception:
        dt_iso = datetime.utcnow().isoformat()
    item = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "datetime": dt_iso,
        "tags": ""
    }
    sql_insert(TIMELINE_TABLE, item)

def get_timeline():
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TIMELINE_TABLE} ORDER BY datetime ASC")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

# --- Brain mimic helper (build prompt from user chats/memories) ---
def build_brain_mimic_prompt(user_messages=20):
    chats = get_chats()
    # collect latest user messages to mimic
    user_texts = [c['content'] for c in chats if c['role'] == 'user']
    sample = "\n".join(user_texts[-user_messages:])
    memories = get_memories()
    mem_texts = "\n".join([f"{m['title']}: {m['content']}" for m in memories[:20]])
    prompt = (
        "You are EchoSoul. In 'Brain Mimic' mode, emulate the user's style, tone, typical phrases and preferences "
        "based on the user's previous messages and memories. Use the following examples and memories:\n\n"
        f"EXAMPLES:\n{sample}\n\nMEMORIES:\n{mem_texts}\n\n"
        "When you respond, reply as the user would, including the kinds of emotive cues and shorthand the user tends to use. "
        "Be concise and genuine."
    )
    return prompt

# --- Streamlit UI components ---
def render_chat_ui():
    st.header("EchoSoul — Chat")
    # Chat display (left main column)
    cols = st.columns([3,1])
    with cols[0]:
        chats = get_chats()
        for c in chats:
            role = c['role']
            t = dateparser.parse(c['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            if role == "user":
                st.markdown(f"<div style='text-align:right; padding:6px; margin:6px; background:#2b2b2b; color:white; border-radius:10px; display:inline-block;'>{c['content']}<div style='font-size:10px;color:#999'>{t}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; padding:6px; margin:6px; background:#383e56; color:white; border-radius:10px; display:inline-block;'>{c['content']}<div style='font-size:10px;color:#ddd'>{t}</div></div>", unsafe_allow_html=True)

    # Controls (right column)
    with cols[1]:
        st.subheader("Controls")
        voice_choice = st.selectbox("Voice (for calls & audio)", ["default", "soft-male", "soft-female", "deep", "bright"], key="voice_choice")
        st.write("Emotion detection, Brain Mimic, Memory saving available.")
        st.checkbox("Auto-save memory from chat (detects statements like 'I love...' or 'I live in...')", key="autosave_memory")

    # Input area at bottom
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    def send_message():
        txt = st.session_state.user_input.strip()
        if not txt:
            return
        # Save user message
        add_chat("user", txt)

        # Emotion detection
        emo, details = detect_emotion_from_text(txt)

        # Possibly autosave memory (very naive heuristics)
        if st.session_state.get("autosave_memory"):
            if any(kw in txt.lower() for kw in ["i am", "i'm", "my name", "i live", "i love", "i like", "i worked at", "i have"]):
                try:
                    add_memory(title=f"Auto: {txt[:40]}", content=txt, tags=[emo])
                except Exception:
                    pass

        # Use LLM to reply
        system = "You are EchoSoul — a caring personal mirror. Keep responses empathetic and personal."
        resp_text = chat_with_llm(prompt=txt, system_prompt=system)
        add_chat("assistant", resp_text)

        # Clear input box automatically
        st.session_state.user_input = ""

    st.text_input("Message...", key="user_input", on_change=send_message, placeholder="Type your message then press Enter (box will clear automatically).")

def render_history_ui():
    st.header("Chat History")
    chats = get_chats()
    df = pd.DataFrame(chats)
    if not df.empty:
        df_display = df[['timestamp','role','content']].copy()
        st.dataframe(df_display)
    if st.button("Delete all history"):
        sql_delete_all(CHAT_TABLE)
        st.success("Chat history cleared.")

def render_timeline_ui():
    st.header("Life Timeline")
    timeline = get_timeline()
    if timeline:
        for e in timeline:
            dt = dateparser.parse(e['datetime']).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**{e['title']}**  —  *{dt}*\n\n{e['description']}\n---")
    else:
        st.info("Timeline empty. Add events below.")

    with st.form("add_timeline"):
        title = st.text_input("Event Title")
        dt = st.text_input("Date/time (any format, e.g. '2025-10-02 15:00')")
        desc = st.text_area("Description")
        submitted = st.form_submit_button("Add to timeline")
        if submitted and title:
            add_timeline_event(title, desc, dt or datetime.utcnow().isoformat())
            st.success("Added to timeline.")

def render_vault_ui():
    st.header("Memory Vault (Encrypted)")

    vault_password = st.text_input("Vault password", type="password", key="vault_password")
    if vault_password:
        try:
            vault_data = load_vault(vault_password)
            st.success("Vault unlocked.")
            st.write("Stored entries:")
            st.write(vault_data.keys())
            # Display and manage entries
            for k in list(vault_data.keys()):
                st.markdown(f"**{k}**")
                st.write(vault_data[k])
                if st.button(f"Delete {k}", key=f"del_{k}"):
                    vault_data.pop(k, None)
                    save_vault(vault_data, vault_password)
                    st.experimental_rerun()
        except ValueError as e:
            st.error("Invalid vault password.")
    else:
        st.info("Enter vault password to unlock. To create a new vault, enter a password and save data below.")

    st.markdown("---")
    with st.form("vault_add"):
        key_name = st.text_input("Entry title")
        content = st.text_area("Sensitive content")
        add_btn = st.form_submit_button("Save to vault (encrypted)")
        if add_btn:
            if not vault_password:
                st.error("Enter vault password first.")
            else:
                try:
                    data = {}
                    if vault_exists():
                        try:
                            data = load_vault(vault_password)
                        except Exception:
                            data = {}
                    data[key_name or f"entry_{uuid.uuid4().hex[:6]}"] = {
                        "content": content,
                        "saved_at": datetime.utcnow().isoformat()
                    }
                    save_vault(data, vault_password)
                    st.success("Saved to encrypted vault.")
                except Exception as e:
                    st.error(f"Error saving vault: {e}")

def render_export_ui():
    st.header("Export / Backup")
    st.write("Download your chats, memories and timeline as JSON.")
    data = {
        "chats": get_chats(),
        "memories": get_memories(),
        "timeline": get_timeline()
    }
    st.download_button("Download JSON", json.dumps(data, indent=2), file_name="echosoul_export.json", mime="application/json")
    if st.button("Clear all data (chats, memories, timeline)"):
        sql_delete_all(CHAT_TABLE)
        sql_delete_all(MEMORY_TABLE)
        sql_delete_all(TIMELINE_TABLE)
        st.success("Cleared.")

def render_brain_mimic_ui():
    st.header("Brain Mimic")
    st.write("EchoSoul will attempt to adopt your voice and thinking style using your past chats and memories.")
    mimic_prompt = build_brain_mimic_prompt()
    st.write("Preview of the prompt given to the LLM (editable):")
    user_prompt = st.text_area("Prompt / question for Brain Mimic", value="Say something supportive in my style.", height=150, key="bm_prompt")
    mimic_strength = st.slider("Mimic strength (how closely to imitate)", 0.0, 1.0, 0.8)
    if st.button("Ask Brain Mimic"):
        system = "You are EchoSoul in Brain Mimic mode. Use the user's past content to imitate their voice."
        # Compose a final prompt combining mimic prompt and user prompt
        final_prompt = f"{mimic_prompt}\n\nUser question: {user_prompt}\nMimic strength: {mimic_strength}"
        answer = chat_with_llm(final_prompt, system_prompt=system)
        add_chat("assistant", answer)
        st.success("Brain Mimic response saved to chat.")
        st.write(answer)

def render_call_ui():
    st.header("Call (Voice) Simulation")
    # Emulate the look: calling... active call with avatar, end call button
    call_state = st.session_state.get("call_state", "idle")  # idle, calling, active
    if call_state == "idle":
        if st.button("Start Call (simulate)"):
            st.session_state.call_state = "calling"
            st.experimental_rerun()
    elif call_state == "calling":
        st.markdown("<div style='text-align:center; font-size:22px; padding:60px;'>📞 Calling... (EchoSoul)</div>", unsafe_allow_html=True)
        cols = st.columns([1,1,1])
        if cols[1].button("Answer"):
            st.session_state.call_state = "active"
            st.experimental_rerun()
        if cols[1].button("End call"):
            st.session_state.call_state = "idle"
            st.experimental_rerun()
    elif call_state == "active":
        st.markdown("<div style='text-align:center; font-size:18px; padding:6px;'>🔊 Active Call — Use the text box below to speak (EchoSoul will voice its replies)</div>", unsafe_allow_html=True)
        if st.button("End call"):
            st.session_state.call_state = "idle"
            st.experimental_rerun()

        # During active call, emulate mic input and TTS reply:
        if 'call_input' not in st.session_state:
            st.session_state.call_input = ""
        def call_send():
            txt = st.session_state.call_input.strip()
            if not txt:
                return
            # Save user utterance as chat
            add_chat("user", txt)
            # Response via LLM
            resp = chat_with_llm(txt, system_prompt="You are EchoSoul speaking with the user on a call, be conversational.")
            add_chat("assistant", resp)
            # Create TTS
            success, audio_bytes, mime = generate_tts(resp, voice=st.session_state.get("voice_choice", "default"))
            if success and audio_bytes:
                st.audio(audio_bytes, format=mime)
            else:
                st.warning("TTS generation unavailable (no TTS engine configured). Showing text response instead.")
                st.write(resp)
            st.session_state.call_input = ""
        st.text_input("Speak (type) — press Enter to send", key="call_input", on_change=call_send)

# --- Sidebar with navigation ---
st.sidebar.title("EchoSoul")
st.sidebar.markdown("Personal AI companion — persistent memory, timeline, vault, and voice calls.")
page = st.sidebar.radio("Navigate", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"], index=0)

st.sidebar.markdown("---")
# API key input
openai_key = None
if "OPENAI_API_KEY" in st.secrets:
    openai_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_key = st.sidebar.text_input("API Key (optional)", type="password", placeholder="Paste your OpenAI API key (recommended)", key="api_key_input")
if openai_key:
    try:
        set_openai_key(openai_key)
        st.sidebar.success("API key set.")
    except Exception:
        st.sidebar.error("Could not set API key.")

# Quick "call" indicator (like the icon in your screenshot)
if "messages" in st.session_state:
