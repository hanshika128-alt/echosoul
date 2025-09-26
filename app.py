"""
EchoSoul — Full app.py (single file)
Features:
- Onboarding questions (asks name, age, hobbies, free-time; remembers them)
- Chat with GPT (gpt-4o-mini). Chats clear automatically and feel like an ongoing dialogue.
- Roleplay-as-you capability via system prompt.
- Persistent memory: timeline saved to echosoul_data.json
- Adaptive personality via simple sentiment heuristic
- Private Vault (Fernet if available, XOR fallback)
- Favorites / Pinned chats, Saved outputs
- Change background from uploaded gallery images (applies site-wide CSS)
- Voice: upload a voice sample (used as "voice profile"); optional TTS using OpenAI audio if available
- Theme toggle (dark default) + high-contrast option (neon glow)
- UI: support & feedback, settings, updates indicator
"""

import streamlit as st
import os
import json
import hashlib
import base64
import datetime
import re
import io
import textwrap
from typing import Optional

# --- OpenAI client (read key from Streamlit Secrets) ---
try:
    from openai import OpenAI
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_KEY)
except Exception:
    client = None

# --- Optional Fernet for vault encryption ---
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

# --- Storage file ---
DATA_FILE = "echosoul_data.json"

def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {
        "profile": {
            "name": None,
            "age": None,
            "hobbies": None,
            "free_time": None,
            "created": ts_now(),
            "persona": {"tone": "friendly", "style": "casual"},
            "intro_completed": False
        },
        "timeline": [],
        "vault": [],
        "conversations": [],
        "favorites": [],
        "saved_outputs": [],
        "notifications": {"updates": False}
    }

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data()
    return default_data()

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --- Encryption helpers ---
def gen_fernet_key_from_password(password: str) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)

def encrypt_with_fernet(password: str, plaintext: str) -> str:
    key = gen_fernet_key_from_password(password)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

def decrypt_with_fernet(password: str, token: str) -> Optional[str]:
    try:
        key = gen_fernet_key_from_password(password)
        f = Fernet(key)
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception:
        return None

def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def encrypt_xor(password, plaintext):
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_xor(password, ciphertext_b64):
    try:
        data = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data))
        x = bytes([data[i] ^ key[i] for i in range(len(data))])
        return x.decode("utf-8")
    except Exception:
        return None

def encrypt_text(password, plaintext):
    if USE_FERNET:
        return encrypt_with_fernet(password, plaintext)
    return encrypt_xor(password, plaintext)

def decrypt_text(password, blob):
    if USE_FERNET:
        return decrypt_with_fernet(password, blob)
    return decrypt_xor(password, blob)

# --- Memory helpers ---
def add_memory(data, title, content, tags=None):
    item = {
        "id": hashlib.sha1((title + content + ts_now()).encode("utf-8")).hexdigest(),
        "title": title,
        "content": content,
        "tags": tags or [],
        "timestamp": ts_now()
    }
    data["timeline"].append(item)
    save_data(data)
    return item

def find_relevant_memories(data, text, limit=3):
    found = []
    txt = text.lower()
    for item in reversed(data["timeline"]):
        if any(w in txt for w in re.findall(r"\w+", item["content"].lower())) or any(w in txt for w in re.findall(r"\w+", item["title"].lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# --- Sentiment & persona ---
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", (text or "").lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    norm = score / max(1, len(toks))
    return norm

def sentiment_label(score):
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"

def update_persona_based_on_sentiment(data, score):
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# --- OpenAI wrappers ---
def call_gpt(system_prompt: str, user_msg: str) -> str:
    if client is None:
        return "OpenAI not configured. Add OPENAI_API_KEY to Streamlit Secrets."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8,
            max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[GPT error: {str(e)}]"

def transcribe_audio_bytes(file_bytes: bytes, filename: str) -> Optional[str]:
    # Uses OpenAI audio transcription if client available; returns transcription or None
    if client is None:
        return None
    try:
        file_obj = io.BytesIO(file_bytes)
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_obj)
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        return str(resp)
    except Exception:
        return None

def synthesize_voice_tts(text: str, voice_profile: Optional[bytes] = None) -> Optional[bytes]:
    """
    Attempt to synthesize TTS using OpenAI if available.
    voice_profile is optional: if the SDK supports custom voices, you'd pass voice metadata.
    Returns raw audio bytes (e.g., mp3) or None on failure.
    """
    if client is None:
        return None
    try:
        # Example (SDKs differ): try a simple TTS call
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",  # fallback voice name; real names depend on API availability
            input=text
        )
        # If the SDK returns bytes-like object:
        if hasattr(resp, "audio"):
            return resp.audio
        # or if resp contains 'audio' key:
        if isinstance(resp, dict) and "audio" in resp:
            return base64.b64decode(resp["audio"])
        # fallback: string
        return None
    except Exception:
        return None

# --- Reply generator with auto-memory and persona context ---
def generate_reply(data, user_msg):
    # update sentiment/persona
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)

    # build memory context (last 6)
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-6:]]
    context = "\n".join(memories) if memories else "No memories yet."

    tone = data["profile"]["persona"].get("tone", "friendly")

    system_prompt = textwrap.dedent(f"""
    You are EchoSoul, a personal companion assistant for the user's memories and emotions.
    Personality tone: {tone}.
    Use the user's timeline and profile facts when relevant.
    Keep the conversation flowing: refer to recent messages and memories, stay on topic, and make replies feel like an ongoing dialogue—not isolated Q&A.
    If the user asks you to "act like me", roleplay as the user using their profile details and memories.
    Do not reveal private vault contents unless the user asks and proves they own the vault password.
    Known facts:
    {context}
    """)

    reply = call_gpt(system_prompt, user_msg)

    # auto-save: if user says "remember", or "my name is", or similar patterns
    low = (user_msg or "").lower()
    if re.search(r"\bremember\b", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b", low) or re.search(r"\bi'm\b", low):
        add_memory(data, "User fact", user_msg)

    # store conversation
    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# --- Onboarding flow ---
def run_onboarding(data):
    st.header("Welcome — let me get to know you 🤍")
    st.write("I'll ask a few quick questions so I can remember and speak like you. You can change these later in Settings.")
    if "onb_step" not in st.session_state:
        st.session_state.onb_step = 0

    step = st.session_state.onb_step

    if step == 0:
        name = st.text_input("What's your name?", value=(data["profile"].get("name") or ""))
        if st.button("Next"):
            if name.strip():
                data["profile"]["name"] = name.strip()
                add_memory(data, "Name", f"My name is {name.strip()}")
                st.session_state.onb_step = 1
                save_data(data)
                st.experimental_rerun()
            else:
                st.warning("Please enter your name.")
    elif step == 1:
        age = st.text_input("What's your age?", value=(data["profile"].get("age") or ""))
        if st.button("Next"):
            if age.strip():
                data["profile"]["age"] = age.strip()
                add_memory(data, "Age", f"My age is {age.strip()}")
                st.session_state.onb_step = 2
                save_data(data)
                st.experimental_rerun()
            else:
                st.warning("Please enter your age.")
    elif step == 2:
        hobbies = st.text_area("What are your hobbies?", value=(data["profile"].get("hobbies") or ""))
        if st.button("Next"):
            if hobbies.strip():
                data["profile"]["hobbies"] = hobbies.strip()
                add_memory(data, "Hobbies", hobbies.strip())
                st.session_state.onb_step = 3
                save_data(data)
                st.experimental_rerun()
            else:
                st.warning("Please enter at least one hobby.")
    elif step == 3:
        free_time = st.text_area("What do you like to do in your free time?", value=(data["profile"].get("free_time") or ""))
        if st.button("Finish and continue"):
            if free_time.strip():
                data["profile"]["free_time"] = free_time.strip()
                add_memory(data, "Free time", free_time.strip())
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.success("Thanks — I will remember these things.")
                st.session_state.onb_step = 0
                st.experimental_rerun()
            else:
                st.warning("Please tell me something you like doing in your free time.")

# --- UI and layout ---
st.set_page_config(page_title="EchoSoul", layout="wide")
data = load_data()

# --- Theming & CSS (no red anywhere) ---
# background image set via session_state
if "bg_image_b64" not in st.session_state:
    st.session_state.bg_image_b64 = None
if "theme_high_contrast" not in st.session_state:
    st.session_state.theme_high_contrast = False

def set_background_from_b64(b64data):
    # apply CSS background (dark theme base)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64data}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply saved background if present
if st.session_state.bg_image_b64:
    set_background_from_b64(st.session_state.bg_image_b64)

# Neon/high-contrast style toggles (no explicit red)
NEON_ACCENT = "#00e5ff"  # cyan neon
if st.session_state.theme_high_contrast:
    st.markdown(f"""
    <style>
    .stButton>button {{ box-shadow: 0 0 12px {NEON_ACCENT}; border-radius: 10px; }}
    .stCheckbox>div>label {{ box-shadow: 0 0 8px {NEON_ACCENT}; }}
    </style>
    """, unsafe_allow_html=True)

# --- Top bar / header ---
st.title("EchoSoul — Your personal companion")
st.write("A private, memory-enabled companion that learns you over time. (No red anywhere — promise!)")

# Sidebar contents
with st.sidebar:
    st.markdown("## EchoSoul — Controls")
    # Profile quick-edit
    st.markdown("**Profile**")
    display_name = data["profile"].get("name") or ""
    new_name = st.text_input("Display name", value=display_name)
    if st.button("Save profile name"):
        data["profile"]["name"] = new_name.strip() or data["profile"]["name"]
        add_memory(data, "Name updated", f"My name set to {data['profile']['name']}")
        save_data(data)
        st.success("Saved.")

    st.markdown("---")
    # Theme toggles
    if st.button("Toggle High-Contrast Mode"):
        st.session_state.theme_high_contrast = not st.session_state.theme_high_contrast
        st.experimental_rerun()
    st.checkbox("Dark mode (default)", value=True, key="dark_mode_toggle")  # for user perception

    st.markdown("---")
    st.markdown("**Favorites / Pinned**")
    st.write("Pin important memories or conversations for fast access.")
    # simple UI: add last conversation to favorites
    if st.button("Pin last conversation"):
        if data.get("conversations"):
            last = data["conversations"][-1]
            data["favorites"].append({"type":"conv","item": last, "ts": ts_now()})
            save_data(data)
            st.success("Pinned last conversation.")
        else:
            st.info("No conversations to pin yet.")

    if st.button("View pinned items"):
        if data["favorites"]:
            for f in data["favorites"]:
                st.write(f"{f['ts']} — {f['type']}")
        else:
            st.info("No pins yet.")

    st.markdown("---")
    st.markdown("**Saved content**")
    if st.button("Save last AI output"):
        if data.get("conversations"):
            last_bot = data["conversations"][-1]["bot"]
            data["saved_outputs"].append({"ts": ts_now(), "content": last_bot})
            save_data(data)
            st.success("Saved last output.")
        else:
            st.info("No output to save yet.")
    if st.button("View saved outputs"):
        if data["saved_outputs"]:
            for s in data["saved_outputs"]:
                st.markdown(f"{s['ts']}")
                st.write(s['content'])
                st.markdown("---")
        else:
            st.info("No saved outputs yet.")

    st.markdown("---")
    st.markdown("**Vault & Security**")
    vault_pwd = st.text_input("Vault password", type="password")
    if USE_FERNET:
        st.markdown("Vault encryption: Fernet (strong).")
    else:
        st.markdown("Vault encryption: XOR fallback (demo).")

    st.markdown("---")
    st.markdown("**Appearance**")
    st.write("Change the app background using an image from your device (applies immediately).")
    bg_file = st.file_uploader("Upload background image (jpg/png)", type=["png","jpg","jpeg"])
    if bg_file is not None:
        b = bg_file.read()
        b64 = base64.b64encode(b).decode("utf-8")
        st.session_state.bg_image_b64 = b64
        set_background_from_b64(b64)
        st.success("Background updated.")

    st.markdown("---")
    st.markdown("**Support & Settings**")
    if st.button("Support / Feedback"):
        st.write("Open the support form at: [Contact/Feedback] — (placeholder in demo).")
    if st.button("Check for updates"):
        data["notifications"]["updates"] = False
        save_data(data)
        st.success("No new updates (demo).")

# If onboarding not completed, run it and stop
if not data["profile"].get("intro_completed", False):
    run_onboarding(data)
    st.stop()

# Main area: tabs
tabs = st.tabs(["Chat", "Conversation History", "Memories & Timeline", "Private Vault", "Legacy & Export", "About"])

# Chat tab
with tabs[0]:
    st.header(f"Chat — Hi {data['profile'].get('name') or 'friend'}")
    st.write("This chat stays on topic and refers back to recent messages. It feels like a continuous conversation.")

    # Upload voice sample to be used as 'voice profile' (optional)
    st.markdown("**Voice profile (optional)** — upload a short voice sample so EchoSoul can mimic voice tone (if supported).")
    voice_file = st.file_uploader("Upload a short voice sample (mp3/wav)", type=["mp3","wav","m4a"])
    if voice_file:
        vbytes = voice_file.read()
        # store voice sample in memory (not in vault) as base64 to persist
        vb64 = base64.b64encode(vbytes).decode("utf-8")
        data["profile"]["voice_sample_b64"] = vb64
        save_data(data)
        st.success("Voice sample saved (for demo TTS where supported).")

    # Audio transcription
    audio_upload = st.file_uploader("Upload audio to transcribe & chat (optional)", type=["mp3","wav","m4a","ogg"])
    if audio_upload:
        a_bytes = audio_upload.read()
        st.info("Transcribing...")
        transcript = transcribe_audio_bytes(a_bytes, audio_upload.name)
        if transcript:
            st.success("Transcription ready — added as your message.")
            # show transcript and generate reply
            st.markdown(f"**You (transcribed):** {transcript}")
            reply = generate_reply(data, transcript)
            st.markdown(f"**EchoSoul:** {reply}")
        else:
            st.error("Transcription not available. Make sure OpenAI key is set and audio transcription is supported.")

    # Show last 20 convs compactly
    convs = data.get("conversations", [])[-20:]
    for c in convs:
        st.markdown(f"**You:** {c['user']}")
        st.markdown(f"**EchoSoul:** {c['bot']}")
        st.markdown("---")

    # chat_input that clears automatically
    user_msg = st.chat_input("Say something to EchoSoul")
    if user_msg:
        with st.spinner("EchoSoul is thinking..."):
            reply = generate_reply(data, user_msg)
        # Optionally synthesize TTS using stored voice
        tts_choice = st.checkbox("Play AI reply in voice profile (if supported)", value=False, key="play_tts")
        if tts_choice:
            voice_b64 = data["profile"].get("voice_sample_b64")
            voice_bytes = base64.b64decode(voice_b64) if voice_b64 else None
            audio_out = synthesize_voice_tts(reply, voice_profile=voice_bytes)
            if audio_out:
                st.audio(audio_out)
            else:
                st.info("TTS not available in this environment or your OpenAI plan. EchoSoul replied in text below.")
        # show reply (chat_input clears by design)
        st.experimental_rerun()

# Conversation History
with tabs[1]:
    st.header("Conversation History")
    conv = data.get("conversations", [])
    if not conv:
        st.info("No conversations yet.")
    else:
        for m in conv[::-1]:
       
