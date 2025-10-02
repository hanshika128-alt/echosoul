# app.py - EchoSoul (Streamlit)
# 2025-10-02
#
# Notes:
# - Place recommended packages into requirements.txt on Streamlit Cloud.
# - Add your OpenAI API key into Streamlit secrets or provide it in-app.
# - This implementation focuses on being robust & safe, and provides fallbacks
#   if optional services are not available.

import streamlit as st
from datetime import datetime
import json
import os
import base64
import hashlib
import io
import uuid
from pathlib import Path

# Optional imports with graceful fallback
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

# --- Basic app config ---
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
st.title("EchoSoul — Personal Reflective AI")

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# files
MEMORY_FILE = DATA_DIR / "memories.json"
TIMELINE_FILE = DATA_DIR / "timeline.json"
CHAT_HISTORY_FILE = DATA_DIR / "chats.json"
VAULT_FILE = DATA_DIR / "vault.bin"
VAULT_META = DATA_DIR / "vault_meta.json"

# Initialize persistent stores
def load_json_file(p: Path, default):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json_file(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

memories = load_json_file(MEMORY_FILE, [])
timeline = load_json_file(TIMELINE_FILE, [])
chats = load_json_file(CHAT_HISTORY_FILE, [])

# Session state defaults
if "voice_choice" not in st.session_state:
    st.session_state.voice_choice = "EchoSoul (default)"
if "api_pin" not in st.session_state:
    st.session_state.api_pin = ""
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []
if "vault_unlocked" not in st.session_state:
    st.session_state.vault_unlocked = False

# --- Helpers: vault encryption ---
def generate_key_from_password(password: str, salt: bytes):
    # Use PBKDF2 to derive a Fernet key (urlsafe base64-encoded 32 bytes)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def vault_create(password: str):
    if not HAS_CRYPTO:
        st.error("Cryptography library missing. Vault can't be created.")
        return False
    salt = os.urandom(16)
    key = generate_key_from_password(password, salt)
    f = Fernet(key)
    initial = json.dumps({"memories": [], "notes": [], "created": datetime.utcnow().isoformat()}).encode()
    token = f.encrypt(initial)
    VAULT_FILE.write_bytes(token)
    save_json_file(VAULT_META, {"salt": base64.b64encode(salt).decode()})
    st.session_state.vault_unlocked = True
    return True

def vault_unlock(password: str):
    if not HAS_CRYPTO:
        st.error("Cryptography library missing. Cannot unlock vault.")
        return False
    if not VAULT_FILE.exists() or not VAULT_META.exists():
        st.warning("No vault found. Create one first.")
        return False
    meta = load_json_file(VAULT_META, {})
    salt = base64.b64decode(meta["salt"])
    key = generate_key_from_password(password, salt)
    f = Fernet(key)
    try:
        data = f.decrypt(VAULT_FILE.read_bytes())
        st.session_state.vault_unlocked = True
        st.session_state._vault_contents = json.loads(data.decode())
        st.session_state._vault_key = key
        return True
    except InvalidToken:
        st.session_state.vault_unlocked = False
        return False

def vault_save_contents():
    if not HAS_CRYPTO:
        st.error("Cryptography library missing. Vault can't be saved.")
        return False
    if not st.session_state.vault_unlocked:
        st.warning("Vault not unlocked.")
        return False
    key = st.session_state._vault_key
    f = Fernet(key)
    token = f.encrypt(json.dumps(st.session_state._vault_contents).encode())
    VAULT_FILE.write_bytes(token)
    return True

# --- Simple emotion recognition (text) ---
EMOTION_KEYWORDS = {
    "sad": ["sad", "unhappy", "depressed", "down", "lonely", "miserable"],
    "happy": ["happy", "joy", "glad", "excited", "elated", "cheerful"],
    "angry": ["angry", "mad", "furious", "irritated", "annoyed"],
    "anxious": ["anxious", "anxiety", "nervous", "worried", "stressed"],
    "tired": ["tired", "sleepy", "exhausted", "drained"],
    "neutral": []
}

def detect_emotion_from_text(text: str):
    text_lower = text.lower()
    scores = {k: 0 for k in EMOTION_KEYWORDS}
    for emo, keys in EMOTION_KEYWORDS.items():
        for kw in keys:
            if kw in text_lower:
                scores[emo] += 1
    # choose best match or neutral
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        # fallback heuristics
        if any(p in text_lower for p in ["!", "yay", "awesome", "great"]):
            return "happy"
        return "neutral"
    return best[0]

# --- Simple voice-to-text using uploaded audio (optional) ---
def transcribe_audio_file(uploaded_file):
    # If openai + whisper available, use it. Otherwise, return a placeholder.
    if HAS_OPENAI and st.secrets.get("OPENAI_API_KEY"):
        try:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            # Use OpenAI Whisper API if available
            resp = openai.Audio.transcriptions.create(
                file=uploaded_file,
                model="whisper-1"
            )
            return resp["text"]
        except Exception:
            pass
    # Fallback: we can't reliably transcribe in this environment
    return None

# --- Chat / memory / brain mimic engine (uses OpenAI if available)
def build_system_prompt_for_user(user_profile):
    # Personalized system prompt using stored memories
    profile_snippet = ""
    if user_profile:
        profile_snippet = " Known details: " + "; ".join(f"{m.get('k')}: {m.get('v')}" for m in user_profile[:10])
    sys_prompt = (
        "You are EchoSoul, a gentle, reflective AI companion whose purpose is to help, mirror, and guide. "
        "You store memories gently and respect privacy. Use empathic tone and adapt to the user's style."
        + profile_snippet
    )
    return sys_prompt

def chat_with_model(user_input, mimic=False, style_examples=None, api_pin=None):
    """
    chat_with_model supports:
    - mimic=True uses style_examples (list of user's messages) to instruct the model to speak like the user
    - If OpenAI is available (and key present), call it; otherwise reply with a simple rule-based fallback
    """
    # quick local fallback (without OpenAI)
    if not (HAS_OPENAI and (st.secrets.get("OPENAI_API_KEY") or api_pin)):
        # Local heuristic "echo + reflect" plus a "brain mimic" style option
        emo = detect_emotion_from_text(user_input)
        if mimic and style_examples:
            # build a simplistic mimic by echoing style punctuation & short phrases
            sample = style_examples[-1] if len(style_examples) else ""
            if len(sample.split()) < 6:
                resp = f"{sample} — hmm. {user_input}"
            else:
                resp = f"{sample.split()[0].capitalize()}, {user_input}"
        else:
            if emo == "happy":
                resp = f"I'm glad to hear that. Tell me more — what's making you feel good?"
            elif emo == "sad":
                resp = "I hear you. That's heavy. Do you want to say more about what's going on?"
            elif emo == "tired":
                resp = "You sound tired. Rest is important. Would you like a soothing breathing exercise?"
            else:
                resp = f"I hear you: \"{user_input}\". How would you like me to support you?"
        return resp

    # If here, use OpenAI chat
    try:
        key = st.secrets.get("OPENAI_API_KEY") or api_pin
        openai.api_key = key
        system_prompt = build_system_prompt_for_user(memories)
        messages = [{"role": "system", "content": system_prompt}]
        if mimic and style_examples:
            # give examples and ask to "mimic"
            mimic_prompt = (
                "Mimic the user's voice and style based on these examples. Keep content aligned with safety and empathy."
                + "\n\nExamples:\n" + "\n".join(style_examples[-10:])
            )
            messages.append({"role": "system", "content": mimic_prompt})
        messages.append({"role": "user", "content": user_input})
        # Chat completion
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list()["data"] else "gpt-4o",
            messages=messages,
            temperature=0.8,
            max_tokens=600
        )
        text = resp.choices[0].message.content
        return text
    except Exception as e:
        # fallback
        return f"[fallback response due to API error: {e}] I heard: {user_input}"

# --- TTS generation (gTTS fallback) ---
def tts_generate_and_play(text: str, voice_choice="default"):
    """
    Returns bytes of an mp3 audio. Uses OpenAI if available & supports TTS, otherwise gTTS fallback.
    """
    # Attempt OpenAI TTS (if available)
    if HAS_OPENAI and st.secrets.get("OPENAI_API_KEY"):
        try:
            # NOTE: API usage may differ depending on OpenAI SDK version.
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            # Example pseudo-call (Streamlit environment may not have TTS configured)
            audio_resp = openai.Audio.speech.create(model="gpt-4o-mini-tts", voice=voice_choice, input=text)
            return io.BytesIO(audio_resp.read())
        except Exception:
            pass
    # gTTS fallback
    if HAS_GTTS:
        try:
            tts = gTTS(text)
            bio = io.BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            return bio
        except Exception:
            pass
    # Last fallback: generate simple beep-speech as silence (no TTS available)
    return None

# --- UI Layout: Sidebar ---
with st.sidebar:
    st.header("EchoSoul — Menu")
    page = st.radio("", ["Chat", "Chat History", "Life Timeline", "Vault", "Export", "Brain Mimic", "About"], index=0)
    st.markdown("---")
    st.subheader("Call Controls")
    if st.button("Start Call (simulate)"):
        st.session_state._in_call = True
        st.session_state.call_id = str(uuid.uuid4())
    if st.button("End Call"):
        st.session_state._in_call = False
        st.session_state.call_id = None

    st.selectbox("Voice (choose)", ["EchoSoul (default)", "Warm", "Neutral", "Bright"], key="voice_choice")
    st.text_input("API PIN (optional)", key="api_pin", placeholder="Enter API pin or openai key (or set secrets)")
    st.markdown("---")
    st.write("Quick Links")
    st.button("New Conversation", on_click=lambda: st.session_state.current_chat.clear())
    st.caption("Left: chat, history, timeline, vault, brain mimic, export & about.")

# --- Top bar: Simulated call screens (two variants from your screenshots) ---
if st.session_state.get("_in_call"):
    # display call UI
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.image("https://via.placeholder.com/180.png?text=EchoSoul", width=150)  # replace with real avatar if available
        st.markdown("<h3 style='text-align:center'>Calling... (EchoSoul)</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # call actions
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Mute"):
                st.info("Muted (simulated).")
        with c2:
            if st.button("End Call"):
                st.session_state._in_call = False
        with c3:
            if st.button("Speaker"):
                st.info("Speaker (simulated).")
    st.write("---")

# --- Pages Implementation ---
def page_chat():
    st.header("Chat with EchoSoul")
    # top right call button
    right_col = st.columns([3,1])[1]
    if right_col.button("Call"):
        st.session_state._in_call = True

    # display conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.current_chat:
            role = msg.get("role")
            timestamp = msg.get("ts")
            if role == "user":
                st.markdown(f"<div style='text-align:right;background:#c6f6d5;padding:8px;border-radius:12px;margin:6px'>{msg['text']}<div style='font-size:10px;color:#666'>{timestamp}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left;background:#2d3748;color:white;padding:8px;border-radius:12px;margin:6px'>{msg['text']}<div style='font-size:10px;color:#ddd'>{timestamp}</div></div>", unsafe_allow_html=True)

    # message input
    st.text_area("Message...", key="input_text", height=120, placeholder="Write to EchoSoul... (press Send)")
    col_send, col_tts, col_clear = st.columns([1,1,1])
    with col_send:
        if st.button("Send"):
            user_msg = st.session_state.input_text.strip()
            if user_msg:
                ts = datetime.utcnow().isoformat()
                st.session_state.current_chat.append({"role":"user","text":user_msg,"ts":ts})
                chats.append({"role":"user","text":user_msg,"ts":ts})
                # store potential memory if user states facts like "I am a violinist"
                # naive heuristic: if user said 'I am' or 'My name is' store as memory
                lowered = user_msg.lower()
                if lowered.startswith("i am ") or "my name is " in lowered or "i'm " in lowered:
                    memories.insert(0, {"k":"self_statement","v":user_msg,"ts":ts})
                    save_json_file(MEMORY_FILE, memories)
                # generate reply (mimic off by default)
                reply = chat_with_model(user_msg, mimic=False, style_examples=[m['text'] for m in chats if m['role']=='user'], api_pin=st.session_state.api_pin)
                rts = datetime.utcnow().isoformat()
                st.session_state.current_chat.append({"role":"echo","text":reply,"ts":rts})
                chats.append({"role":"echo","text":reply,"ts":rts})
                save_json_file(CHAT_HISTORY_FILE, chats)
                # auto-clear input
                st.session_state.input_text = ""
                # Log timeline event
                timeline.append({"type":"chat","text":user_msg,"response":reply,"ts":ts})
                save_json_file(TIMELINE_FILE, timeline)
                st.experimental_rerun()
    with col_tts:
        if st.button("Play last TTS"):
            if st.session_state.current_chat:
                last = st.session_state.current_chat[-1]
                if last["role"] == "echo":
                    bio = tts_generate_and_play(last["text"], voice_choice=st.session_state.voice_choice)
                    if bio:
                        st.audio(bio.read(), format="audio/mp3")
                    else:
                        st.warning("TTS not available. Install OpenAI or gTTS for voice output.")
    with col_clear:
        if st.button("Clear Conversation"):
            st.session_state.current_chat.clear()
            st.experimental_rerun()

def page_history():
    st.header("Chat History")
    st.write("All past messages (persisted).")
    for m in chats[-200:]:
        ts = m.get("ts")
        who = m.get("role")
        txt = m.get("text")
        if who == "user":
            st.markdown(f"*You* — {ts}: {txt}")
        else:
            st.markdown(f"*EchoSoul* — {ts}: {txt}")

def page_timeline():
    st.header("Life Timeline")
    st.write("Chronological record of significant moments and interactions.")
    # add manual event
    with st.expander("Add timeline event"):
        ev_text = st.text_input("Event text", key="ev_text")
        ev_tags = st.text_input("Tags (comma separated)", key="ev_tags")
        if st.button("Add event"):
            ev = {"id": str(uuid.uuid4()), "text": ev_text, "tags": [t.strip() for t in ev_tags.split(",") if t.strip()], "ts": datetime.utcnow().isoformat()}
            timeline.append(ev)
            save_json_file(TIMELINE_FILE, timeline)
            st.success("Event added.")
            st.experimental_rerun()
    # display
    for ev in sorted(timeline, key=lambda x: x.get("ts"), reverse=True):
        st.markdown(f"**{ev.get('ts')}** — {ev.get('text')}  \nTags: {', '.join(ev.get('tags',[]))}")

def page_vault():
    st.header("Memory Vault (encrypted storage)")
    if not HAS_CRYPTO:
        st.warning("Cryptography library not installed; Vault functions disabled. Install `cryptography` to enable.")
    # vault creation / unlock
    if not VAULT_FILE.exists():
        st.info("No Vault exists. Create one now (this will generate encryption metadata).")
        pwd = st.text_input("Create vault password", type="password", key="vault_create_pwd")
        if st.button("Create Vault"):
            ok = vault_create(pwd)
            if ok:
                st.success("Vault created and unlocked.")
    else:
        if not st.session_state.vault_unlocked:
            pwd = st.text_input("Unlock vault password", type="password", key="vault_unlock_pwd")
            if st.button("Unlock Vault"):
                ok = vault_unlock(pwd)
                if ok:
                    st.success("Vault unlocked.")
                else:
                    st.error("Wrong password or vault corrupted.")
        else:
            st.success("Vault unlocked.")
            contents = st.session_state.get("_vault_contents", {})
            st.markdown("**Vault contents (secure):**")
            st.json(contents)
            new_note = st.text_area("Add secure note", key="vault_note")
            if st.button("Save secure note"):
                if "notes" not in st.session_state._vault_contents:
                    st.session_state._vault_contents["notes"] = []
                st.session_state._vault_contents["notes"].append({"text": new_note, "ts": datetime.utcnow().isoformat()})
                vault_save_contents()
                st.success("Note saved and encrypted.")
                st.session_state.vault_unlocked = True
                st.experimental_rerun()
            if st.button("Lock Vault"):
                st.session_state.vault_unlocked = False
                st.session_state._vault_key = None
                st.session_state._vault_contents = None
                st.success("Locked.")

def page_export():
    st.header("Export / Backup")
    st.write("You can export memories, timeline, and chat history.")
    if st.button("Export all as JSON"):
        payload = {
            "memories": memories,
            "timeline": timeline,
            "chats": chats,
            "exported_at": datetime.utcnow().isoformat()
        }
        b = json.dumps(payload, indent=2, ensure_ascii=False).encode()
        st.download_button("Download archive.json", data=b, file_name="echosoul_export.json", mime="application/json")
    if st.button("Export timeline CSV"):
        import csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ts", "type", "text", "tags"])
        for ev in timeline:
            writer.writerow([ev.get("ts",""), ev.get("type",""), ev.get("text",""), ",".join(ev.get("tags",[]))])
        st.download_button("Download timeline.csv", data=buf.getvalue(), file_name="time"
