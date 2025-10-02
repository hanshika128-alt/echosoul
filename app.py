# app.py - EchoSoul (basic working version)
# Streamlit app implementing: Chat, Chat history, Life timeline, Vault (encrypted), Export, Brain mimic, Call (TTS), About
# Defensive: works even if OpenAI / NLTK / gTTS are not available.

import streamlit as st
from datetime import datetime
import sqlite3
import os
import uuid
import json
import hashlib
import base64
from dateutil import parser as dateparser

# Optional: cryptography for secure vault
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

# Optional: openai
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional: gTTS for TTS fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Optional: nltk VADER for emotion detection
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# ---------------------------
# Configuration & DB setup
# ---------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")

DB_FILE = "echosoul.db"
VAULT_FILE = "vault.enc"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            tags TEXT,
            timestamp TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS timeline (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            datetime TEXT,
            tags TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def insert(table, obj: dict):
    cols = ", ".join(obj.keys())
    placeholders = ", ".join(["?"] * len(obj))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    conn.execute(sql, tuple(obj.values()))
    conn.commit()

def query_all(table, order_by="timestamp", asc=False):
    order = "ASC" if asc else "DESC"
    cur = conn.cursor()
    # fallback for timeline which uses 'datetime'
    try:
        cur.execute(f"SELECT * FROM {table} ORDER BY {order_by} {order}")
    except Exception:
        cur.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

def delete_all(table):
    conn.execute(f"DELETE FROM {table}")
    conn.commit()

# ---------------------------
# Vault helpers (optional cryptography)
# ---------------------------
def derive_key(password: str) -> bytes:
    h = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(h)

def vault_exists():
    return os.path.exists(VAULT_FILE)

def save_vault(data: dict, password: str):
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography not available")
    key = derive_key(password)
    f = Fernet(key)
    token = f.encrypt(json.dumps(data).encode())
    with open(VAULT_FILE, "wb") as fh:
        fh.write(token)

def load_vault(password: str) -> dict:
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography not available")
    if not vault_exists():
        return {}
    key = derive_key(password)
    f = Fernet(key)
    with open(VAULT_FILE, "rb") as fh:
        token = fh.read()
    try:
        data = f.decrypt(token)
    except InvalidToken:
        raise ValueError("Invalid vault password")
    return json.loads(data.decode())

# ---------------------------
# LLM & Emotion & TTS helpers
# ---------------------------
def set_openai_key(key: str):
    if OPENAI_AVAILABLE:
        openai.api_key = key

def chat_with_llm(prompt: str, system: str = None, model: str = "gpt-4o-mini"):
    # defensive: if OpenAI not configured, return fallback
    try:
        if OPENAI_AVAILABLE and getattr(openai, "ChatCompletion", None):
            messages = []
            if system:
                messages.append({"role":"system","content":system})
            messages.append({"role":"user","content":prompt})
            resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=512, temperature=0.8)
            return resp.choices[0].message.content.strip()
        elif OPENAI_AVAILABLE and getattr(openai, "chat", None):
            messages = []
            if system:
                messages.append({"role":"system","content":system})
            messages.append({"role":"user","content":prompt})
            resp = openai.chat.completions.create(model=model, messages=messages, max_tokens=512, temperature=0.8)
            return resp.choices[0].message.content.strip()
    except Exception:
        pass
    # Fallback: simple echo / transformation
    return f"(EchoSoul offline response) I heard: {prompt[:400]}"

def detect_emotion(text: str):
    # returns (label, score_dict)
    if VADER_AVAILABLE:
        sid = SentimentIntensityAnalyzer()
        s = sid.polarity_scores(text)
        comp = s.get("compound", 0.0)
        if comp >= 0.5:
            return "happy", s
        elif comp <= -0.5:
            return "sad/angry", s
        elif comp > 0:
            return "positive", s
        elif comp < 0:
            return "negative", s
        else:
            return "neutral", s
    else:
        t = text.lower()
        pos = any(w in t for w in ["happy","love","great","yay","awesome","good","nice"])
        neg = any(w in t for w in ["sad","angry","hate","upset","bad","terrible","depressed"])
        if pos and not neg:
            return "happy", {"compound":0.6}
        if neg and not pos:
            return "sad/angry", {"compound":-0.6}
        return "neutral", {"compound":0.0}

def generate_tts_bytes(text: str, lang="en"):
    # Try gTTS fallback if available. Returns (success, bytes, mime)
    if GTTS_AVAILABLE:
        try:
            t = gTTS(text=text, lang=lang)
            tmp = f"/tmp/echosoul_tts_{uuid.uuid4().hex}.mp3"
            t.save(tmp)
            with open(tmp, "rb") as fh:
                b = fh.read()
            try:
                os.remove(tmp)
            except Exception:
                pass
            return True, b, "audio/mp3"
        except Exception:
            pass
    return False, None, None

# ---------------------------
# App state initialization
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts {id, role, content, ts}
if "call_state" not in st.session_state:
    st.session_state["call_state"] = "idle"  # idle, calling, active
if "voice_choice" not in st.session_state:
    st.session_state["voice_choice"] = "default"
if "vault_unlocked" not in st.session_state:
    st.session_state["vault_unlocked"] = False
if "vault_password" not in st.session_state:
    st.session_state["vault_password"] = ""

# ---------------------------
# Small persistence wrappers (db-backed as well)
# ---------------------------
def save_chat(role: str, content: str):
    item = {"id": str(uuid.uuid4()), "role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
    insert("chats", item)
    # also keep in session_state for UI quick access
    st.session_state["messages"].append(item)

def load_chats_to_session():
    rows = query_all("chats", order_by="timestamp", asc=True)
    st.session_state["messages"] = rows[:1000]  # cap to recent 1000

def save_memory(title: str, content: str, tags=None):
    item = {"id": str(uuid.uuid4()), "title": title, "content": content, "tags": ",".join(tags or []), "timestamp": datetime.utcnow().isoformat()}
    insert("memories", item)

def load_memories():
    return query_all("memories", order_by="timestamp", asc=False)

def save_timeline(title: str, description: str, dtstr: str = None):
    dt_iso = datetime.utcnow().isoformat()
    try:
        if dtstr:
            dt = dateparser.parse(dtstr)
            dt_iso = dt.isoformat()
    except Exception:
        dt_iso = datetime.utcnow().isoformat()
    item = {"id": str(uuid.uuid4()), "title": title, "description": description, "datetime": dt_iso, "tags": ""}
    insert("timeline", item)

def load_timeline():
    return query_all("timeline", order_by="datetime", asc=True)

# preload DB chats into session on start
if not st.session_state["messages"]:
    try:
        load_chats_to_session()
    except Exception:
        st.session_state["messages"] = []

# ---------------------------
# UI RENDERers
# ---------------------------
def chat_view():
    st.header("Chat — EchoSoul")
    # left: chat display, right: controls
    col1, col2 = st.columns([3,1])
    with col1:
        # display messages
        for m in st.session_state["messages"]:
            ts = dateparser.parse(m["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            if m["role"] == "user":
                st.markdown(f"<div style='text-align:right; padding:8px; margin:6px; display:block;'><div style='display:inline-block; background:#1f2937; color:white; padding:8px 12px; border-radius:12px; max-width:80%'>{m['content']}<div style='font-size:10px;color:#9CA3AF'>{ts}</div></div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; padding:8px; margin:6px; display:block;'><div style='display:inline-block; background:#0b1220; color:white; padding:8px 12px; border-radius:12px; max-width:80%'>{m['content']}<div style='font-size:10px;color:#9CA3AF'>{ts}</div></div></div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Controls")
        st.selectbox("Voice", ["default","soft-male","soft-female","deep","bright"], key="voice_choice")
        st.checkbox("Auto-save memory (detect profile-like sentences)", key="autosave_mem")
        st.markdown("---")
        st.write("API key (optional, for better responses):")
        if "OPENAI_API_KEY" in st.secrets:
            set_openai_key(st.secrets["OPENAI_API_KEY"])
            st.success("API key set from secrets.")
        else:
            k = st.text_input("Paste OpenAI key (optional)", type="password", key="ui_openai_key")
            if k:
                try:
                    set_openai_key(k)
                    st.success("API key set.")
                except Exception:
                    st.error("Invalid key or OpenAI library not available.")

    # input area (bottom)
    user_input = st.text_input("Message (press Enter to send)", key="chat_input")
    if st.button("Send"):
        if user_input and user_input.strip():
            txt = user_input.strip()
            save_chat("user", txt)
            # auto-save memory naive heuristics
            if st.session_state.get("autosave_mem", False):
                low = txt.lower()
                triggers = ["i am ", "i'm ", "my name ", "i live ", "i work", "i love", "i like", "i have "]
                if any(t in low for t in triggers):
                    try:
                        save_memory(title=f"Auto: {txt[:40]}", content=txt)
                    except Exception:
                        pass
            # generate assistant reply
            sys = "You are EchoSoul, a kind, reflective companion. Keep replies short, empathetic, and personalized."
            resp = chat_with_llm(txt, system=sys)
            save_chat("assistant", resp)
            # emotion detection
            emo, details = detect_emotion(txt)
            st.experimental_rerun()

def history_view():
    st.header("Chat History")
    chats = query_all("chats", order_by="timestamp", asc=True)
    if not chats:
        st.info("No chat history saved yet.")
    else:
        import pandas as pd
        df = pd.DataFrame(chats)
        st.dataframe(df[["timestamp","role","content"]])
    if st.button("Delete all chat history"):
        delete_all("chats")
        st.success("Chat history deleted.")
        st.experimental_rerun()

def timeline_view():
    st.header("Life Timeline")
    tl = load_timeline()
    if not tl:
        st.info("Timeline is empty — add events below.")
    else:
        for e in tl:
            dt = dateparser.parse(e["datetime"]).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**{e['title']}**  —  *{dt}*\n\n{e['description']}\n---")
    with st.form("add_timeline"):
        title = st.text_input("Event title")
        dt = st.text_input("Date/time (optional)")
        desc = st.text_area("Description")
        add = st.form_submit_button("Add event")
        if add and title:
            save_timeline(title, desc, dt)
            st.success("Added to timeline.")
            st.experimental_rerun()

def vault_view():
    st.header("Memory Vault (Encrypted)")
    if not CRYPTO_AVAILABLE:
        st.error("Vault requires 'cryptography' package. Install it and redeploy to use the vault feature.")
        return
    pw = st.text_input("Vault password", type="password", key="vault_pw")
    if st.button("Unlock vault"):
        if not pw:
            st.warning("Enter a password.")
        else:
            try:
                data = load_vault(pw)
                st.session_state["vault_unlocked"] = True
                st.session_state["vault_password"] = pw
                st.success("Vault unlocked.")
                st.experimental_rerun()
            except ValueError:
                st.error("Invalid password.")
    if st.session_state["vault_unlocked"]:
        try:
            vault_data = load_vault(st.session_state["vault_password"])
        except Exception as e:
            st.error("Could not read vault.")
            vault_data = {}
        st.write("Entries:")
        for k, v in vault_data.items():
            st.markdown(f"**{k}**")
            st.write(v)
            if st.button(f"Delete {k}", key=f"del_{k}"):
                vault_data.pop(k, None)
                save_vault(vault_data, st.session_state["vault_password"])
                st.success("Deleted entry.")
                st.experimental_rerun()
        st.markdown("---")
        with st.form("add_vault"):
            name = st.text_input("Entry title")
            content = st.text_area("Sensitive content")
            add = st.form_submit_button("Save to vault (encrypted)")
            if add:
                if not name:
                    st.error("Give an entry title.")
                else:
                    vault_data[name] = {"content": content, "saved_at": datetime.utcnow().isoformat()}
                    save_vault(vault_data, st.session_state["vault_password"])
                    st.success("Saved to vault.")
                    st.experimental_rerun()
    else:
        st.info("Unlock the vault to see/edit sensitive memories. Vault uses strong encryption (Fernet).")

def export_view():
    st.header("Export / Backup")
    data = {
        "chats": query_all("chats", order_by="timestamp", asc=True),
        "memories": load_memories(),
        "timeline": load_timeline()
    }
    st.download_button("Download export (JSON)", json.dumps(data, indent=2), "echosoul_export.json", "application/json")
    if st.button("Clear all data (chats/memories/timeline)"):
        delete_all("chats")
        delete_all("memories")
        delete_all("timeline")
        st.success("Cleared all local data.")
        st.experimental_rerun()

def brain_mimic_view():
    st.header("Brain Mimic")
    st.write("EchoSoul will attempt to reply in your voice using your recent chat messages and memories.")
    mimic_strength = st.slider("Mimic strength", 0.0, 1.0, 0.8)
    example = st.text_area("Question / prompt for Brain Mimic", "Say something comforting in my style...")
    if st.button("Ask Brain Mimic"):
        # Build a prompt using last user messages + memories
        chats = query_all("chats", order_by="timestamp", asc=True)
        user_examples = [c["content"] for c in chats if c["role"] == "user"][-30:]
        memories = load_memories()[:20]
        mem_text = "\n".join([f"{m['title']}: {m['content']}" for m in memories])
        prompt = (
            f"You are EchoSoul in 'Brain Mimic' mode. Imitate the user's writing voice and typical phrasing. "
            f"Use these user examples:\n{json.dumps(user_examples[-10:], ensure_ascii=False)}\n\n"
            f"User memories:\n{mem_text}\n\n"
            f"Now answer the user's prompt: {example}\nMimic strength: {mimic_strength}"
        )
        sys = "You are EchoSoul - behave like the user would, concise and authentic."
        resp = chat_with_llm(prompt, system=sys)
        save_chat("assistant", resp)
        st.success("Saved Brain Mimic response to chat.")
        st.write(resp)

def call_view():
    st.header("Call (Voice) Simulation")
    st.write("Simulates a voice call: type messages (as if speaking) and EchoSoul will reply and try to speak via TTS.")
    state = st.session_state["call_state"]
    if state == "idle":
        if st.button("Start call"):
            st.session_state["call_state"] = "calling"
            st.experimental_rerun()
    elif state == "calling":
        st.info("Calling... Choose answer or end.")
        if st.button("Answer"):
            st.session_state["call_state"] = "active"
            st.experimental_rerun()
        if st.button("End call"):
            st.session_state["call_state"] = "idle"
            st.experimental_rerun()
    elif state == "active":
        if st.button("End call"):
            st.session_state["call_state"] = "idle"
            st.experimental_rerun()
        # Input area as "speech"
        speak = st.text_input("Speak (type text and press Send)", key="call_input")
        if st.button("Send (call)"):
            if speak and speak.strip():
                save_chat("user", speak.strip())
                sys = "You are EchoSoul on a friendly call. Keep voice responses natural and conversational."
                resp = chat_with_llm(speak.strip(), system=sys)
                save_chat("assistant", resp)
                # attempt to generate TTS and play
                ok, audio_bytes, mime = generate_tts_bytes(resp)
                if ok:
                    st.audio(audio_bytes, format=mime)
                else:
                    st.warning("TTS not available; showing text response instead.")
                    st.write(resp)

def about_view():
    st.header("About EchoSoul")
    st.markdown("""
    EchoSoul is a personal AI companion prototype that demonstrates:
    - Persistent chat & memory (local sqlite)
    - Life timeline
    - Encrypted private vault (Fernet) for sensitive memories
    - Brain mimic (uses chat + memories as prompt context)
    - Call simulation (text + TTS fallback)
    - Export / backup
    """)
    st.markdown("**Notes & next steps**")
    st.write("""
    - To enable stronger LLM responses and real TTS voices, connect an LLM and a TTS provider (e.g., OpenAI, ElevenLabs).
    - Vault requires `cryptography` to be installed.
    - This app stores data locally in the app instance. For multi-device persistence, move to a cloud DB.
    """)

# ---------------------------
# Sidebar & navigation
# ---------------------------
st.sidebar.title("EchoSoul")
st.sidebar.caption("Personal AI companion — persistent memory, timeline, vault, and voice calls.")
page = st.sidebar.radio("Navigate", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"])
st.sidebar.markdown("---")

# Show a simple call indicator
if st.session_state["call_state"] == "active":
    st.sidebar.markdown("🔵 Active call")
elif st.session_state["call_state"] == "calling":
    st.sidebar.markdown("⚪ Calling...")

# quick debug actions
if st.sidebar.button("Load DB chats into session"):
    load_chats
