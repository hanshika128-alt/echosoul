# app.py - EchoSoul
# EchoSoul single-file Streamlit app prototype
# 2025-10-02
# Drop into a Streamlit Cloud repo. Add required packages to requirements.txt if you want advanced features.

import streamlit as st
from datetime import datetime
from pathlib import Path
import json
import io
import os
import uuid
import base64
import typing

# Optional libraries (graceful fallback)
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

# App config
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
st.title("EchoSoul — Personal Reflective AI")

# Data directory and files
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MEMORY_FILE = DATA_DIR / "memories.json"
CHAT_FILE = DATA_DIR / "chats.json"
TIMELINE_FILE = DATA_DIR / "timeline.json"
VAULT_FILE = DATA_DIR / "vault.bin"
VAULT_META = DATA_DIR / "vault_meta.json"
VOICE_DIR = DATA_DIR / "voices"
VOICE_DIR.mkdir(exist_ok=True)

# -- helpers for persistence
def load_json(p: Path, default):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

memories = load_json(MEMORY_FILE, [])
chats = load_json(CHAT_FILE, [])
timeline = load_json(TIMELINE_FILE, [])

# -- session defaults
if "messages" not in st.session_state:
    st.session_state.messages = []   # live chat messages: list of {"role","text","ts"}
if "vault_unlocked" not in st.session_state:
    st.session_state.vault_unlocked = False
if "vault_contents" not in st.session_state:
    st.session_state.vault_contents = {}
if "vault_key" not in st.session_state:
    st.session_state.vault_key = None
if "call_active" not in st.session_state:
    st.session_state.call_active = False
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "EchoSoul (Default)"
if "custom_voices" not in st.session_state:
    # items are {"name":..., "path":...}
    st.session_state.custom_voices = []
if "api_pin" not in st.session_state:
    st.session_state.api_pin = ""
if "adaptive_style" not in st.session_state:
    st.session_state.adaptive_style = {}

# ------------------ Vault encryption helpers ------------------
def _derive_key(password: str, salt: bytes):
    """Derive a Fernet key from password using PBKDF2 (returns base64 urlsafe 32-bytes)"""
    if not HAS_CRYPTO:
        raise RuntimeError("cryptography not installed")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    return key

def create_vault(password: str):
    if not HAS_CRYPTO:
        raise RuntimeError("cryptography not installed")
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    initial = {"notes": [], "created": datetime.utcnow().isoformat()}
    token = f.encrypt(json.dumps(initial).encode("utf-8"))
    VAULT_FILE.write_bytes(token)
    save_json(VAULT_META, {"salt": base64.b64encode(salt).decode()})
    st.session_state.vault_unlocked = True
    st.session_state.vault_contents = initial
    st.session_state.vault_key = key
    return True

def unlock_vault(password: str) -> bool:
    if not HAS_CRYPTO:
        st.error("cryptography not installed, cannot unlock vault.")
        return False
    if not VAULT_FILE.exists() or not VAULT_META.exists():
        st.warning("No vault exists yet.")
        return False
    meta = load_json(VAULT_META, {})
    salt_b64 = meta.get("salt")
    if not salt_b64:
        st.error("Vault metadata corrupted.")
        return False
    salt = base64.b64decode(salt_b64)
    key = _derive_key(password, salt)
    f = Fernet(key)
    try:
        data = f.decrypt(VAULT_FILE.read_bytes())
        st.session_state.vault_unlocked = True
        st.session_state.vault_contents = json.loads(data.decode("utf-8"))
        st.session_state.vault_key = key
        return True
    except InvalidToken:
        return False

def save_vault():
    if not (HAS_CRYPTO and st.session_state.vault_unlocked and st.session_state.vault_key):
        st.error("Vault not ready to save.")
        return False
    f = Fernet(st.session_state.vault_key)
    token = f.encrypt(json.dumps(st.session_state.vault_contents).encode("utf-8"))
    VAULT_FILE.write_bytes(token)
    return True

# ------------------ Emotion recognition (text heuristics) ------------------
EMO_KEYS = {
    "sad": ["sad","unhappy","down","depressed","miserable","tears","lonely"],
    "happy": ["happy","joy","glad","excited","great","yay","amazing"],
    "angry": ["angry","mad","furious","annoyed","hate"],
    "anxious": ["anxious","nervous","worried","panic","anxiety","stressed"],
    "tired": ["tired","exhausted","sleepy","drained"]
}
def detect_emotion_text(text: str) -> str:
    t = text.lower()
    scores = {k:0 for k in EMO_KEYS}
    for k, lst in EMO_KEYS.items():
        for kw in lst:
            if kw in t:
                scores[k] += 1
    winner = max(scores.items(), key=lambda x: x[1])
    if winner[1] == 0:
        return "neutral"
    return winner[0]

# ------------------ Chat / model helper ------------------
def build_system_prompt():
    # combine some memories into a prompt
    snippet = ""
    if memories:
        top = memories[:8]
        snippet = "Known details: " + "; ".join(f"{m.get('k')}: {m.get('v')}" for m in top)
    base = (
        "You are EchoSoul, a compassionate, adaptive reflective AI companion. "
        "Be empathic, concise, and adapt your tone based on detected mood and user preferences. "
        + snippet
    )
    return base

def local_fallback_response(user_text: str, mimic: bool=False, style_examples: typing.List[str]=None) -> str:
    # Simple fallback reply
    emo = detect_emotion_text(user_text)
    if mimic and style_examples:
        # try to mimic: simple tactic — borrow punctuation and short-sentence vibe
        sample = style_examples[-1] if style_examples else ""
        if len(sample.split()) < 5:
            return f"{sample}... {user_text}"
        return f"{sample.split()[0].capitalize()}, {user_text}"
    if emo == "happy":
        return "That's wonderful to hear — tell me more about it!"
    if emo == "sad":
        return "I'm sorry you're feeling that way. Do you want to talk about what's weighing on you?"
    if emo == "tired":
        return "You sound tired. Would you like a short relaxation exercise?"
    return f"I hear you: \"{user_text}\" — how would you like me to support you?"

def chat_with_ai(user_text: str, mimic=False, style_examples: typing.List[str]=None, api_key: str=None) -> str:
    """
    If OpenAI key available and HAS_OPENAI True, attempt an API call. Otherwise fallback to rules.
    """
    # Prefer explicit api_key (api_pin) or st.secrets
    key = api_key or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
    if HAS_OPENAI and key:
        try:
            openai.api_key = key
            system_prompt = build_system_prompt()
            messages = [{"role":"system","content":system_prompt}]
            if mimic and style_examples:
                ex_block = "\n".join(style_examples[-10:])
                messages.append({"role":"system","content":"Mimic the user's voice using these examples:\n" + ex_block})
            messages.append({"role":"user","content":user_text})
            # Use ChatCompletion if available; fallback to ChatCompletion.create
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if "gpt-4o-mini" in [m.id for m in openai.Model.list().data] else "gpt-4o",
                messages=messages,
                temperature=0.8,
                max_tokens=400
            )
            return resp.choices[0].message.content
        except Exception as e:
            # fall back
            return f"[AI API failed: {e}] " + local_fallback_response(user_text, mimic, style_examples)
    else:
        return local_fallback_response(user_text, mimic, style_examples)

# ------------------ TTS helper (gTTS fallback) ------------------
def generate_tts_audio(text: str, voice_name: str="default"):
    """
    Returns BytesIO with audio (mp3) or None.
    Uses OpenAI TTS if available & configured, else gTTS fallback if installed.
    """
    # Attempt OpenAI TTS (if available)
    key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
    if HAS_OPENAI and key:
        try:
            openai.api_key = key
            # Note: OpenAI TTS API surface can differ; this is a guarded try.
            # If you have a specific endpoint or SDK version, swap here.
            audio_resp = openai.Audio.speech.create(model="gpt-4o-mini-tts", voice=voice_name, input=text)
            bio = io.BytesIO(audio_resp.read())
            bio.seek(0)
            return bio
        except Exception:
            pass
    # Fallback to gTTS
    if HAS_GTTS:
        try:
            tts = gTTS(text)
            bio = io.BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            return bio
        except Exception:
            pass
    return None

# ------------------ Voice upload handling ------------------
def save_uploaded_voice(uploaded_file, name: str=None):
    if not uploaded_file:
        return None
    name = name or uploaded_file.name
    safe_name = f"{uuid.uuid4().hex}_{name}"
    dest = VOICE_DIR / safe_name
    bytes_data = uploaded_file.read()
    dest.write_bytes(bytes_data)
    entry = {"name": name, "path": str(dest)}
    st.session_state.custom_voices.append(entry)
    return entry

# ------------------ UI: Sidebar ------------------
with st.sidebar:
    st.header("EchoSoul — Menu")
    page = st.radio("", ["Chat", "Chat History", "Life Timeline", "Vault", "Export", "Brain Mimic", "About"], index=0)
    st.markdown("---")
    st.subheader("Call & Voice")
    if st.button("Start Call (simulate)"):
        st.session_state.call_active = True
    if st.button("End Call"):
        st.session_state.call_active = False

    # built-in voices + custom ones
    builtin_voices = ["EchoSoul (Default)", "Warm", "Calm", "Bright"]
    voice_options = builtin_voices + [v["name"] for v in st.session_state.custom_voices]
    st.selectbox("Choose voice", options=voice_options, key="selected_voice")

    uploaded_voice = st.file_uploader("Upload voice (mp3/wav) to use in calls", type=["mp3","wav"])
    if uploaded_voice is not None:
        try:
            ent = save_uploaded_voice(uploaded_voice, uploaded_voice.name)
            if ent:
                st.success(f"Saved custom voice: {ent['name']}")
                # update selectbox choices by re-render (selectbox uses session_state so value will persist)
        except Exception as e:
            st.error(f"Failed to save voice: {e}")

    st.text_input("API PIN (optional)", key="api_pin", placeholder="Optional: OpenAI key or pin")
    st.markdown("---")
    st.write("Quick actions")
    if st.button("New Conversation"):
        st.session_state.messages = []
    st.caption("Left: chat, history, timeline, vault, brain mimic, export & about.")

# ------------------ Call screen simulation ------------------
def call_screen_ui():
    st.markdown("## Call — EchoSoul (simulated)")
    # big avatar
    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        st.image("https://via.placeholder.com/180.png?text=EchoSoul", width=140)
        st.markdown("<div style='text-align:center'><b>Calling...</b></div>", unsafe_allow_html=True)
        # call controls
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            if st.button("Mute"):
                st.info("Muted (simulated)")
        with cc2:
            if st.button("End Call (screen)"):
                st.session_state.call_active = False
        with cc3:
            if st.button("Speaker"):
                st.info("Speaker toggled (simulated)")

    # if a live chat message exists, play it over TTS as the "call voice"
    last_ai = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "ai":
            last_ai = m
            break
    if last_ai:
        st.markdown(f"**EchoSoul says:** {last_ai['text']}")
        if st.button("Play as call audio"):
            audio = generate_tts_audio(last_ai["text"], voice_name=st.session_state.selected_voice)
            if audio:
                try:
                    st.audio(audio.read(), format="audio/mp3")
                except Exception:
                    audio.seek(0)
                    st.audio(audio)
            else:
                st.warning("TTS not available (no OpenAI key or gTTS).")

# ------------------ Pages ------------------
def render_chat_page():
    st.header("Chat — Text & Voice")
    # call button in header (appears as small icon)
    call_col = st.columns([9,1])[1]
    if call_col.button("📞"):
        st.session_state.call_active = True

    # chat area
    chat_box = st.container()
    with chat_box:
        if not st.session_state.messages and chats:
            # load from persisted chats if available
            # chats are persisted global chat history (chats list)
            for c in chats[-200:]:
                # map persisted roles to 'user' / 'ai'
                role = "user" if c.get("role","") == "user" else "ai"
                st.session_state.messages.append({"role": role, "text": c.get("text",""), "ts": c.get("ts","")})
        # render messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align:right;background:#b9fbc0;padding:10px;border-radius:12px;margin:8px'>{msg['text']}<div style='font-size:10px;color:#444'>{msg.get('ts','')}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left;background:#1f2937;color:white;padding:10px;border-radius:12px;margin:8px'>{msg['text']}<div style='font-size:10px;color:#ddd'>{msg.get('ts','')}</div></div>", unsafe_allow_html=True)

    # bottom input bar like messaging apps (st.chat_input)
    user_input = st.chat_input("Message...")
    if user_input:
        ts = datetime.utcnow().isoformat()
        st.session_state.messages.append({"role":"user","text":user_input,"ts":ts})
        chats.append({"role":"user","text":user_input,"ts":ts})
        # heuristic memory capture
        lower = user_input.lower()
        if lower.startswith("i am ") or "my name is " in lower or lower.startswith("i'm "):
            memories.insert(0, {"k":"self_statement", "v": user_input, "ts": ts})
            save_json(MEMORY_FILE, memories)
        # generate reply (by default mimic False)
        style_examples = [m['text'] for m in chats if m.get("role")=="user"]
        ai_reply = chat_with_ai(user_input, mimic=False, style_examples=style_examples, api_key=(st.session_state.api_pin or None))
        rts = datetime.utcnow().isoformat()
        st.session_state.messages.append({"role":"ai","text":ai_reply,"ts":rts})
        chats.append({"role":"ai","text":ai_reply,"ts":rts})
        # timeline event
        timeline.append({"type":"chat","text":user_input,"response":ai_reply,"ts":ts})
        save_json(CHAT_FILE, chats)
        save_json(TIMELINE_FILE, timeline)
        # after sending, chat_input auto-clears; we re-render to show messages present
        st.experimental_rerun()

    # quick TTS playback / send voice
    col1, col2, col3 = st.columns([1,1,6])
    with col1:
        if st.button("Play last AI TTS"):
            # find last AI message
            last_ai = None
            for m in reversed(st.session_state.messages):
                if m["role"] == "ai":
                    last_ai = m
                    break
            if last_ai:
                audio = generate_tts_audio(last_ai["text"], voice_name=st.session_state.selected_voice)
                if audio:
                    try:
                        st.audio(audio.read(), format="audio/mp3")
                    except Exception:
                        audio.seek(0)
                        st.audio(audio)
                else:
                    st.warning("TTS not available.")
            else:
                st.info("No AI message yet.")

    with col2:
        if st.button("Save conversation"):
            # persist all in-memory messages to chats list & file
            # chats global already appended above, but ensure complete persistence
            save_json(CHAT_FILE, chats)
            st.success("Saved chat history.")

    with col3:
        st.write("")  # spacer

def render_history_page():
    st.header("Chat History")
    st.write("Full persisted chat history (most recent first).")
    for m in reversed(chats[-500:]):
        who = "You" if m.get("role")=="user" else "EchoSoul"
        st.markdown(f"**{who}** ({m.get('ts','')}): {m.get('text','')}")

def render_timeline_page():
    st.header("Life Timeline")
    st.write("A chronological record of events, chats, and important moments.")
    with st.expander("Add timeline event"):
        e_text = st.text_input("Event description", key="ev_text")
        e_tags = st.text_input("Tags (comma separated)", key="ev_tags")
        if st.button("Add event"):
            ev = {"id": str(uuid.uuid4()), "text": e_text, "tags": [t.strip() for t in e_tags.split(",") if t.strip()], "ts": datetime.utcnow().isoformat(), "type":"manual"}
            timeline.append(ev)
            save_json(TIMELINE_FILE, timeline)
            st.success("Event added.")
            st.experimental_rerun()
    for ev in sorted(timeline, key=lambda x: x.get("ts",""), reverse=True):
        tags = ", ".join(ev.get("tags",[]))
        st.markdown(f"**{ev.get('ts')}** — {ev.get('text')}  \nTags: {tags}")

def render_vault_page():
    st.header("Memory Vault — Secure storage")
    if not HAS_CRYPTO:
        st.warning("Vault requires `cryptography` library. Install it to enable secure vault.")
    if not VAULT_FILE.exists():
        st.info("No vault found. Create a secure vault to store sensitive memories.")
        pwd = st.text_input("Create vault password", type="password", key="vault_create_pwd")
        if st.button("Create Vault"):
            try:
                create_vault(pwd)
                st.success("Vault created and unlocked.")
            except Exception as e:
                st.error(f"Unable to create vault: {e}")
    else:
        if not st.session_state.vault_unlocked:
            pwd = st.text_input("Vault password", type="password", key="vault_unlock_pwd")
            if st.button("Unlock Vault"):
                ok = unlock_vault(pwd)
                if ok:
                    st.success("Vault unlocked.")
                else:
                    st.error("Wrong password.")
        else:
            st.success("Vault unlocked.")
            st.json(st.session_state.vault_contents)
            note = st.text_area("Add secure note", key="vault_note")
            if st.button("Save secure note"):
                if "notes" not in st.session_state.vault_contents:
                    st.session_state.vault_contents["notes"] = []
                st.session_state.vault_contents["notes"].append({"text": note, "ts": datetime.utcnow().isoformat()})
                save_vault()
                st.success("Saved to vault.")
            if st.button("Lock Vault"):
                st.session_state.vault_unlocked = False
                st.session_state.vault_contents = {}
                st.s
