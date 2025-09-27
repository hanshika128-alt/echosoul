# app.py - EchoSoul (single-file)
# Full-featured Streamlit app with GPT chat, persistent memory, vault, timeline, export, and TTS.

import streamlit as st
import json
import os
import base64
import hashlib
import re
import io
import datetime
import textwrap

# OpenAI Python client
import openai

# Optional strong encryption (Fernet) if installed
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

# ---------------------------
# Configuration & setup
# ---------------------------
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")

# Read OpenAI key from Streamlit Secrets
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else None
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
else:
    openai.api_key = None  # We'll check later and show helpful messages

DATA_FILE = "echosoul_data.json"
DEFAULT_TTS_VOICES = ["alloy", "verse", "shimmer", "default"]

# ---------------------------
# Utilities
# ---------------------------
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def safe_load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------
# Default data and load/save
# ---------------------------
def default_data():
    return {
        "profile": {
            "name": None,
            "age": None,
            "hobbies": None,
            "free_time": None,
            "created": ts_now(),
            "intro_completed": False,
            "persona": {"tone": "friendly", "style": "casual"}
        },
        "timeline": [],
        "vault": [],
        "conversations": [],
        "settings": {"theme": "dark", "bg_image_b64": None, "tts_voice": "default"},
        "voice_samples": {}  # store uploaded voice sample metadata (not actual cloning)
    }

def load_data():
    d = safe_load_json(DATA_FILE)
    if not d:
        d = default_data()
        safe_save_json(DATA_FILE, d)
        return d
    # auto-heal keys
    base = default_data()
    for k, v in base.items():
        if k not in d:
            d[k] = v
    return d

def save_data(data):
    safe_save_json(DATA_FILE, data)

data = load_data()

# ---------------------------
# Encryption helpers (Fernet optional; XOR fallback)
# ---------------------------
def gen_fernet_key_from_password(password: str) -> bytes:
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h)

def encrypt_with_fernet(password: str, plaintext: str) -> str:
    key = gen_fernet_key_from_password(password)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

def decrypt_with_fernet(password: str, token: str) -> str:
    try:
        key = gen_fernet_key_from_password(password)
        f = Fernet(key)
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception:
        return None

# XOR demo
def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def encrypt_xor(password, plaintext):
    pb = plaintext.encode("utf-8")
    key = _derive_key(password, len(pb))
    x = bytes([pb[i] ^ key[i] for i in range(len(pb))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_xor(password, cipher_b64):
    try:
        data_b = base64.b64decode(cipher_b64.encode("utf-8"))
        key = _derive_key(password, len(data_b))
        x = bytes([data_b[i] ^ key[i] for i in range(len(data_b))])
        return x.decode("utf-8")
    except Exception:
        return None

def encrypt_text(password, plaintext):
    if USE_FERNET:
        return encrypt_with_fernet(password, plaintext)
    return encrypt_xor(password, plaintext)

def decrypt_text(password, cipher):
    if USE_FERNET:
        return decrypt_with_fernet(password, cipher)
    return decrypt_xor(password, cipher)

# ---------------------------
# Memory helpers
# ---------------------------
def add_memory(data_obj, title, content, tags=None):
    item = {
        "id": hashlib.sha1((title + content + ts_now()).encode("utf-8")).hexdigest(),
        "title": title,
        "content": content,
        "tags": tags or [],
        "timestamp": ts_now()
    }
    data_obj["timeline"].append(item)
    save_data(data_obj)
    return item

def find_relevant_memories(data_obj, text, limit=3):
    found = []
    txt = text.lower()
    for item in reversed(data_obj["timeline"]):
        # naive relevance: overlap of words
        if any(w in txt for w in re.findall(r"\w+", item["content"].lower())) or any(w in txt for w in re.findall(r"\w+", item["title"].lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# ---------------------------
# Sentiment / persona
# ---------------------------
POS_WORDS = {"good","great","happy","love","excellent","amazing","wonderful","nice","fun","delighted","calm","optimistic","excited"}
NEG_WORDS = {"bad","sad","angry","depressed","unhappy","terrible","awful","hate","lonely","anxious","stressed","worried","frustrated"}

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    return score / max(1, len(toks))

def update_persona_based_on_sentiment(data_obj, score):
    if score < -0.06:
        data_obj["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data_obj["profile"]["persona"]["tone"] = "energetic"
    else:
        data_obj["profile"]["persona"]["tone"] = "friendly"
    save_data(data_obj)

# ---------------------------
# OpenAI wrappers (GPT + transcription + TTS)
# ---------------------------
def call_gpt(system_prompt: str, conversation, user_message: str, max_tokens=600, temperature=0.8):
    if openai.api_key is None:
        return ("[No OpenAI key configured. Put OPENAI_API_KEY in Streamlit Secrets.]", None)
    try:
        # Build messages: system + trimmed history + user message
        messages = [{"role":"system","content":system_prompt}]
        # use last 6 conversation turns for context
        for conv in conversation[-6:]:
            messages.append({"role":"user","content":conv.get("user","")})
            messages.append({"role":"assistant","content":conv.get("bot","")})
        messages.append({"role":"user","content":user_message})

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # SDK shape variations handled
        try:
            content = resp.choices[0].message.content
        except Exception:
            # fallback
            content = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else str(resp)
        # heuristic confidence
        conf = min(0.99, max(0.2, 1 - len(user_message)/1000))
        return content, conf
    except Exception as e:
        return f"[GPT error: {e}]", None

def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    if openai.api_key is None:
        return None
    try:
        # openai SDKs vary; common pattern:
        # file-like object
        file_obj = io.BytesIO(file_bytes)
        # Attempt transcription call (may vary by SDK availability)
        resp = openai.Audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=file_obj
        )
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        if hasattr(resp, "text"):
            return resp.text
        return str(resp)
    except Exception:
        return None

def generate_tts_bytes(text: str, voice: str = "alloy") -> bytes:
    """
    Generate TTS using OpenAI audio.speech (gpt-4o-mini-tts).
    Returns raw audio bytes (mp3/wav) or None on error.
    """
    if openai.api_key is None:
        return None
    try:
        # Many SDKs expose audio generation under different paths; try a flexible approach:
        # new OpenAI python client might support: openai.audio.speech.create(...)
        # We'll try multiple call patterns for compatibility; handle exceptions and fallbacks.
        try:
            # pattern 1
            resp = openai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text
            )
            # If resp is bytes-like
            if hasattr(resp, "read"):
                return resp.read()
            if isinstance(resp, (bytes, bytearray)):
                return bytes(resp)
            if isinstance(resp, dict) and "audio" in resp:
                # maybe base64 string
                return base64.b64decode(resp["audio"])
            # fallback string
            return str(resp).encode("utf-8")
        except Exception:
            # pattern 2: openai.Audio.speech.create
            try:
                resp = openai.Audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
                if hasattr(resp, "read"):
                    return resp.read()
                if isinstance(resp, dict) and "data" in resp:
                    return base64.b64decode(resp["data"])
                return str(resp).encode("utf-8")
            except Exception as e2:
                # Can't generate TTS with current SDK
                return None
    except Exception:
        return None

# ---------------------------
# Reply generator (ties everything together)
# ---------------------------
def generate_reply(data_obj, user_msg, tts_voice: str = None):
    # update persona
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data_obj, score)
    tone = data_obj["profile"]["persona"].get("tone", "friendly")

    # memory context
    mems = [f"{m['title']}: {m['content']}" for m in data_obj.get("timeline", [])[-6:]]
    mem_text = "\n".join(mems) if mems else "No memories yet."

    profile_text = "\n".join([f"{k}: {v}" for k, v in data_obj["profile"].items() if k in ("name","age","hobbies","free_time") and v])

    system_prompt = textwrap.dedent(f"""
    You are EchoSoul, a helpful, empathetic, personal AI companion.
    Personality tone: {tone}
    Profile facts:
    {profile_text}

    Memories (use as context):
    {mem_text}

    Conversation behavior:
    - Stay on topic and link back to the user's previous messages.
    - Refer back to what you just said when helpful.
    - Keep the conversation feeling like an ongoing dialogue (not isolated Q&A).
    - If user asks "act like me", roleplay as instructed using stored profile facts.
    - Provide concise answers and offer to expand.
    """)

    reply_text, conf = call_gpt(system_prompt, data_obj.get("conversations", []), user_msg)

    # Auto-save user-provided facts if expressive patterns found
    low = user_msg.lower()
    if re.search(r"\bremember\b", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b", low):
        add_memory(data_obj, "User fact", user_msg)

    # Save conversation
    conv = {"user": user_msg, "bot": reply_text, "ts": ts_now()}
    if conf is not None:
        conv["confidence_heuristic"] = float(conf)

    # Generate TTS if requested and API available
    voice_b64 = None
    if tts_voice and openai.api_key is not None:
        audio_bytes = generate_tts_bytes(reply_text, voice=tts_voice)
        if audio_bytes:
            voice_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            conv["voice_b64"] = voice_b64

    conv["explain"] = f"Used {len(mems)} memory item(s); tone={tone}"
    data_obj.setdefault("conversations", []).append(conv)
    save_data(data_obj)
    return reply_text, conv["explain"], voice_b64, conv.get("confidence_heuristic")

# ---------------------------
# Onboarding flow
# ---------------------------
def run_onboarding(data_obj):
    st.header("Welcome — let's get to know each other")
    st.write("I'll ask 4 short questions so EchoSoul can remember you and speak like you.")
    if "onb_step" not in st.session_state:
        st.session_state.onb_step = 0

    step = st.session_state.onb_step

    if step == 0:
        name = st.text_input("What's your name?", value=(data_obj["profile"].get("name") or ""))
        if st.button("Next"):
            if name.strip():
                data_obj["profile"]["name"] = name.strip()
                add_memory(data_obj, "Name", f"My name is {name.strip()}")
                st.session_state.onb_step = 1
                save_data(data_obj)
                st.experimental_rerun()
            else:
                st.warning("Please enter your name.")
    elif step == 1:
        age = st.text_input("What's your age?", value=(data_obj["profile"].get("age") or ""))
        if st.button("Next"):
            if age.strip():
                data_obj["profile"]["age"] = age.strip()
                add_memory(data_obj, "Age", f"My age is {age.strip()}")
                st.session_state.onb_step = 2
                save_data(data_obj)
                st.experimental_rerun()
            else:
                st.warning("Please enter your age.")
    elif step == 2:
        hobbies = st.text_area("What are your hobbies? (comma separated)", value=(data_obj["profile"].get("hobbies") or ""))
        if st.button("Next"):
            if hobbies.strip():
                data_obj["profile"]["hobbies"] = hobbies.strip()
                add_memory(data_obj, "Hobbies", hobbies.strip())
                st.session_state.onb_step = 3
                save_data(data_obj)
                st.experimental_rerun()
            else:
                st.warning("Please enter at least one hobby.")
    elif step == 3:
        free_time = st.text_area("What do you like to do in your free time?", value=(data_obj["profile"].get("free_time") or ""))
        if st.button("Finish"):
            if free_time.strip():
                data_obj["profile"]["free_time"] = free_time.strip()
                add_memory(data_obj, "Free time", free_time.strip())
                data_obj["profile"]["intro_completed"] = True
                save_data(data_obj)
                st.success("Thanks — I will remember these things.")
                st.session_state.onb_step = 0
                st.experimental_rerun()
            else:
                st.warning("Please share something you like doing.")

# ---------------------------
# UI Style (dark + cyan accent)
# ---------------------------
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background-color: #0B0F1A; color: #EDEEF2; }
    h1, h2, h3, h4 { color: #CFF7F3; }
    .neon { color: #8EF6E4; text-shadow: 0 0 8px rgba(142,246,228,.25); }
    .glass { background: rgba(255,255,255,0.03); border-radius:12px; padding:12px; }
    .stButton>button { background:transparent; border:1px solid #8EF6E4; color:#EDEEF2; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar (profile, vault, appearance)
# ---------------------------
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='width:48px;height:48px;border-radius:12px;background:#0f1724;display:flex;align-items:center;justify-content:center;color:#8EF6E4;font-weight:700'>ES</div><div><strong>EchoSoul</strong><br><small>Personal AI companion</small></div></div>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("Profile")
    new_name = st.text_input("Display name", value=data.get("profile", {}).get("name") or "")
    if st.button("Save name"):
        if new_name.strip():
            data["profile"]["name"] = new_name.strip()
            add_memory(data, "Name (edited)", f"My name is {new_name.strip()}")
            save_data(data)
            st.success("Name saved.")
        else:
            st.warning("Name cannot be empty.")

    st.markdown("---")
    st.header("Vault (prototype)")
    st.write("Set a session vault password (stored only in your session) to encrypt/decrypt private notes.")
    vault_pw = st.text_input("Vault password (session)", type="password")
    if vault_pw:
        st.session_state["vault_pw"] = vault_pw
    if st.button("Clear session password"):
        st.session_state.pop("vault_pw", None)
        st.success("Cleared session password")

    st.markdown("---")
    st.header("Appearance")
    st.write("Upload a background image (saved locally).")
    bg_upload = st.file_uploader("Background image (jpg/png)", type=["jpg", "jpeg", "png"])
    if bg_upload:
        bbytes = bg_upload.read()
        data["settings"]["bg_image_b64"] = base64.b64encode(bbytes).decode("utf-8")
        save_data(data)
        st.success("Background saved.")

    st.markdown("---")
    st.header("Voice (TTS)")
    st.write("Choose a TTS voice or upload a voice sample (for future voice-cloning).")
    tts_choice = st.selectbox("TTS voice", ["default"] + DEFAULT_TTS_VOICES, index=0)
    data["settings"]["tts_voice"] = tts_choice
    voice_sample = st.file_uploader("Upload reference voice (wav/mp3) - optional", type=["wav","mp3","m4a"])
    if voice_sample:
        vb = voice_sample.read()
        key = hashlib.sha1(vb).hexdigest()
        data["voice_samples"][key] = {"filename": voice_sample.name, "b64": base64.b64encode(vb).decode("utf-8"), "uploaded_at": ts_now()}
        save_data(data)
        st.success("Voice sample uploaded (stored locally).")

    st.markdown("---")
    st.write("Deployment note: put your OpenAI key in Streamlit Secrets (not in this repo).")
    st.markdown("If TTS or transcription doesn't work, check your OpenAI account access and permissions.")

# Apply background if user uploaded one
bg_b64 = data.get("settings", {}).get("bg_image_b64")
if bg_b64:
    st.markdown(f"<style>body{{background-image:url('data:image/png;base64,{bg_b64}');background-size:cover;}}</style>", unsafe_allow_html=True)

# ---------------------------
# Onboarding if needed
# ---------------------------
if not data.get("profile", {}).get("intro_completed", False):
    run_onboarding(data)
    st.stop()

# ---------------------------
# Main app: tabs
# ---------------------------
tabs = st.tabs(["Chat", "Timeline", "Vault", "Legacy & Export", "About"])

# ----- Chat tab -----
with tabs[0]:
    st.header(f"Chat — Hello {data['profile'].get('name') or 'friend'}! (tone: {data['profile']['persona'].get('tone')})")

    # Show last n messages
    convs = data.get("conversations", [])
    if not convs:
        st.info("No messages yet. Say hello!")
    else:
        for c in convs[-12:]:
            st.markdown(f"**You:** {c.get('user')}")
            st.markdown(f"**EchoSoul:** {c.get('bot')}")
            if c.get("explain"):
                st.caption(f"Why: {c.get('explain')}")
            if c.get("confidence_heuristic") is not None:
                st.caption(f"Confidence (heuristic): {float(c.get('confidence_heuristic')):.2f}")
            if c.get("voice_b64"):
                audio_bytes = base64.b64decode(c["voice_b64"].encode("utf-8"))
                st.audio(audio_bytes, format="audio/mp3")
            st.markdown("---")

    # Audio upload → transcription
    st.write("You can also upload a voice message to transcribe and send:")
    uploaded_audio = st.file_uploader("Upload audio (mp3/wav/m4a/ogg)", type=["mp3","wav","m4a","ogg"], key="upload_audio")
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        st.info("Transcribing...")
        trans = transcribe_audio(audio_bytes, uploaded_audio.name)
        if trans:
            st.success("Transcription complete; sending as message")
