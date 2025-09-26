# app.py
"""
EchoSoul — Full Streamlit app (single file)
Features included:
- Onboarding (name, age, hobbies, free time)
- Conversational AI using OpenAI GPT (gpt-4o-mini)
- Chat input clears after send (st.chat_input)
- Persistent local memory in echosoul_data.json
- Adaptive persona (sentiment -> tone)
- Private Vault (XOR demo, optional Fernet if cryptography installed)
- Legacy export and readable snapshot
- Background customization from uploaded image
- Audio transcription (upload) and TTS reply (OpenAI TTS) — optional based on API availability
- Explainability: short note + confidence heuristic shown per reply
- Defensive coding and clear error messages
"""

import streamlit as st
import os, json, hashlib, base64, datetime, re, io, textwrap
from typing import Optional

# ---------------------------
# OpenAI client (from Streamlit Secrets)
# ---------------------------
client = None
try:
    from openai import OpenAI
    if "OPENAI_API_KEY" in st.secrets:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    else:
        client = None
except Exception:
    client = None

# ---------------------------
# Optional stronger encryption (Fernet) if available
# ---------------------------
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

# ---------------------------
# Data storage helpers
# ---------------------------
DATA_FILE = "echosoul_data.json"

def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {
        "profile": {
            "name": None, "age": None, "hobbies": None, "free_time": None,
            "created": ts_now(), "persona": {"tone": "friendly", "style": "casual"},
            "intro_completed": False
        },
        "timeline": [],
        "vault": [],
        "conversations": [],
        "settings": {"bg_image_b64": None, "theme": "dark", "tts_voice": "alloy"}
    }

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = default_data()
    else:
        data = default_data()
    # Ensure required keys exist (auto-heal)
    if "profile" not in data: data["profile"] = default_data()["profile"]
    if "timeline" not in data: data["timeline"] = []
    if "vault" not in data: data["vault"] = []
    if "conversations" not in data: data["conversations"] = []
    if "settings" not in data: data["settings"] = default_data()["settings"]
    return data

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------
# Vault encryption (Fernet optional; XOR fallback)
# ---------------------------
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

# XOR prototype
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

def decrypt_text(password, ciphertext):
    if USE_FERNET:
        return decrypt_with_fernet(password, ciphertext)
    return decrypt_xor(password, ciphertext)

# ---------------------------
# Memory helpers
# ---------------------------
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

# ---------------------------
# Simple sentiment heuristics for adaptive personality
# ---------------------------
POS_WORDS = {"good","great","happy","love","excellent","amazing","wonderful","nice","fun","delighted","calm","optimistic","excited"}
NEG_WORDS = {"bad","sad","angry","depressed","unhappy","terrible","awful","hate","lonely","anxious","stressed","worried","frustrated"}

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    return score / max(1, len(toks))

def update_persona_based_on_sentiment(data, score):
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# ---------------------------
# OpenAI helpers (GPT + audio). Defensive if client missing.
# ---------------------------
def call_gpt(system_prompt: str, conversation: list, user_message: str, max_tokens=600, temperature=0.8):
    """
    Returns (reply_text, confidence_heuristic) or (error_message, None).
    conversation is a list of dicts with keys "user" and "bot".
    """
    if client is None:
        return ("[OpenAI API key not configured — add OPENAI_API_KEY in Streamlit Secrets.]", None)
    try:
        # Build messages: system + last few conv turns + user
        messages = [{"role":"system","content":system_prompt}]
        # Convert stored conversation to chat messages (keep last 8 turns)
        trimmed = conversation[-8:] if conversation else []
        for turn in trimmed:
            messages.append({"role":"user","content":turn.get("user","")})
            messages.append({"role":"assistant","content":turn.get("bot","")})
        messages.append({"role":"user","content":user_message})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Flexible parsing for SDK versions
        try:
            reply = resp.choices[0].message.content
        except Exception:
            # fallback if response shape different
            reply = getattr(resp.choices[0], "message", None) or str(resp)
            if isinstance(reply, dict) and "content" in reply:
                reply = reply["content"]

        # heuristic confidence (placeholder)
        conf = min(0.99, max(0.2, 1 - len(user_message) / 2000))
        return (reply, float(conf))
    except Exception as e:
        return (f"[GPT error: {str(e)}]", None)

def transcribe_audio_file(file_bytes: bytes, filename: str) -> Optional[str]:
    if client is None:
        return None
    try:
        file_obj = io.BytesIO(file_bytes)
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_obj)
        # attempt common fields
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        if hasattr(resp, "text"):
            return resp.text
        return str(resp)
    except Exception:
        return None

def generate_tts_bytes(text: str, voice: str = "alloy") -> Optional[bytes]:
    if client is None:
        return None
    try:
        resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
        # If response is stream-like
        if hasattr(resp, "read"):
            return resp.read()
        if isinstance(resp, dict) and "data" in resp:
            return base64.b64decode(resp["data"])
        # fallback
        return str(resp).encode("utf-8")
    except Exception:
        return None

# ---------------------------
# Main generator tying memory + persona + GPT + explainability
# ---------------------------
def generate_reply(data, user_msg, tts_voice: Optional[str] = None):
    # Update persona from sentiment
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)
    tone = data["profile"]["persona"].get("tone", "friendly")

    # Build system prompt: uses recent memories and the user's profile
    memories = [f"{m['title']}: {m['content']}" for m in data.get("timeline", [])[-6:]]
    mem_text = "\n".join(memories) if memories else "No memories yet."

    profile_facts = []
    for k in ("name","age","hobbies","free_time"):
        v = data["profile"].get(k)
        if v:
            profile_facts.append(f"{k}: {v}")
    profile_text = "\n".join(profile_facts) if profile_facts else "No profile facts yet."

    system_prompt = textwrap.dedent(f"""
    You are EchoSoul — a personal AI companion.
    Personality tone: {tone}.
    Profile facts:
    {profile_text}

    Known memories (use as context):
    {mem_text}

    Conversation rules:
    - Stay on topic; build on the recent conversation.
    - Refer back to what you just said when helpful.
    - Be conversational and feel like an ongoing dialogue, not isolated Q&A.
    - If asked to "act like me", roleplay using profile facts and memories.
    - Provide short explainable notes when asked.
    """)

    conversation = data.get("conversations", [])
    reply_text, conf = call_gpt(system_prompt, conversation, user_msg)

    # Auto-save facts if user explicitly asked memory or gave profile info
    low = user_msg.lower()
    if re.search(r"\bremember\b", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b", low):
        # Save user-provided text as a memory (title indicates auto-saved)
        add_memory(data, "User fact", user_msg)

    # Save the conversation entry
    conv_entry = {"user": user_msg, "bot": reply_text, "ts": ts_now()}
    if conf is not None:
        conv_entry["confidence_heuristic"] = conf

    # TTS: produce audio bytes and save base64 if successful
    if tts_voice:
        audio_bytes = generate_tts_bytes(reply_text, voice=tts_voice)
        if audio_bytes:
            conv_entry["voice_b64"] = base64.b64encode(audio_bytes).decode("utf-8")

    # short explainability note
    explain_note = f"Used {len(memories)} memory item(s); tone={tone}"
    conv_entry["explain"] = explain_note

    data.setdefault("conversations", []).append(conv_entry)
    save_data(data)

    return reply_text, explain_note, conv_entry.get("voice_b64"), conf

# ---------------------------
# Onboarding UI
# ---------------------------
def run_onboarding(data):
    st.header("Welcome — let me get to know you")
    st.write("I will ask a few short questions so EchoSoul can speak like you and remember things.")
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
                st.warning("Please list at least one hobby.")
    elif step == 3:
        free_time = st.text_area("What do you like to do in your free time?", value=(data["profile"].get("free_time") or ""))
        if st.button("Finish"):
            if free_time.strip():
                data["profile"]["free_time"] = free_time.strip()
                add_memory(data, "Free time", free_time.strip())
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.success("Thanks — I will remember these things.")
                st.session_state.onb_step = 0
                st.experimental_rerun()
            else:
                st.warning("Please share one thing you like doing.")

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
data = load_data()

# Sidebar (detailed)
with st.sidebar:
    st.markdown("<div style='display:flex;gap:10px;align-items:center'><div style='background:#222;color:white;padding:10px;border-radius:8px'>ES</div><div><strong>EchoSoul</strong><br><small>Personal AI companion</small></div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Profile")
    pname = data.get("profile", {}).get("name", "") or ""
    new_name = st.text_input("Display name", value=pname)
    if st.button("Save name"):
        if new_name.strip():
            data["profile"]["name"] = new_name.strip()
            add_memory(data, "Name (edited)", f"My name is {new_name.strip()}")
            save_data(data)
            st.success("Name saved.")
        else:
            st.warning("Name cannot be empty.")

    st.markdown("---")
    st.subheader("Vault (prototype)")
    st.write("Set a session vault password to encrypt/decrypt private notes (not stored to disk).")
    vault_pw = st.text_input("Vault password (session)", type="password")
    if vault_pw:
        st.session_state["vault_password_session"] = vault_pw
    if st.button("Clear session password"):
        if "vault_password_session" in st.session_state:
            st.session_state.pop("vault_password_session")
        st.success("Cleared session password.")

    st.markdown("---")
    st.subheader("Appearance")
    st.write("Upload background image (will save to app settings).")
    bg_file = st.file_uploader("Background image (png/jpg)", type=["png","jpg","jpeg"])
    if bg_file is not None:
        try:
            b = bg_file.read()
            data["settings"]["bg_image_b64"] = base64.b64encode(b).decode("utf-8")
            save_data(data)
            st.success("Background saved.")
        except Exception as e:
            st.error(f"Failed to save background: {e}")

    theme_choice = st.radio("Theme", ("dark","light"), index=0 if data["settings"].get("theme","dark")=="dark" else 1)
    data["settings"]["theme"] = theme_choice
    save_data(data)

    st.markdown("---")
    st.subheader("TTS voice")
    st.write("Choose a voice for AI replies (prototype).")
    voice_choice = st.selectbox("TTS voice", ["default","alloy","verse","shimmer"], index=0)
    data["settings"]["tts_voice"] = voice_choice
    save_data(data)

    st.markdown("---")
    st.subheader("Deployment & privacy")
    st.write("• Keep your OpenAI key in Streamlit Secrets (not in the repo).")
    st.write("• Data stored locally in echosoul_data.json (timeline, conversations, vault).")
    st.write("• Vault uses XOR fallback unless 'cryptography' is installed.")

# Set background from settings if present
if data.get("settings", {}).get("bg_image_b64"):
    bg_b64 = data["settings"]["bg_image_b64"]
    st.markdown(f"<style>body{{background-image:url('data:image/png;base64,{bg_b64}');background-size:cover;}}</style>", unsafe_allow_html=True)

# If onboarding not done, run onboarding and halt
if not data.get("profile", {}).get("intro_completed", False):
    run_onboarding(data)
    st.stop()

# Main tabs
tab_names = ["Chat","Timeline","Vault","Legacy & Export","About"]
tabs = st.tabs(tab_names)

# -------- CHAT TAB --------
with tabs[0]:
    st.header(f"Chat — {data.get('profile', {}).get('name') or 'friend'} (tone: {data.get('profile', {}).get('persona',{}).get('tone','friendly')})")

    # Show last messages
    convs = data.get("conversations", [])
    if not convs:
        st.info("No conversation yet — say hello!")
    else:
        # Show most recent 12 entries
        for entry in convs[-12:]:
            st.markdown(f"**You:** {entry.get('user')}")
            st.markdown(f"**EchoSoul:** {entry.get('bot')}")
            if entry.get("explain"):
                st.caption(f"Why: {entry.get('explain')}")
            if entry.get("confidence_heuristic") is not None:
                try:
                    st.caption("Confidence (heuristic): " + f"{entry.get('confidence_heuristic'):.2f}")
                except Exception:
                    st.caption(f"Confidence (heuristic): {entry.get('confidence_heuristic')}")
            if entry.get("voice_b64"):
                audio_bytes = base64.b64decode(entry["voice_b64"].encode("utf-8"))
                st.audio(audio_bytes, format="audio/mp3")
            st.markdown("---")

    # Audio upload to transcribe and send
    st.write("Upload audio to transcribe and send as a message (optional).")
    uploaded_audio = st.file_uploader("Upload audio (mp3/wav/m4a/ogg)", type=["mp3","wav","m4a","ogg"], key="u_audio")
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        st.info("Transcribing audio...")
        transcription = transcribe_audio_file(audio_bytes, uploaded_audio.name)
        if transcription:
            st.success("Transcription done — sending as your message.")
            reply_text, explain_note, voice_b64, conf = generate_reply(data, transcription, tts_voice=data.get("settings",{}).get("tts_voice"))
            st.markdown(f"**You (transcribed):** {transcription}")
            st.markdown(f"**EchoSoul:** {reply_text}")
            if voice_b64:
                st.audio(base64.b64decode(voice_b64.encode("utf-8")))
        else:
            st.error("Transcription unavailable (check API key or service).")

    # Chat input that clears automatically
    user_message = st.chat_input("Say something to EchoSoul")
    if user_message:
        # generate reply with selected voice (if not "default")
        chosen_voice = data.get("settings", {}).get("tts_voice")
        chosen_voice_param = None if chosen_voice in (None, "default") else chosen_voice
        with st.spinner("EchoSoul is thinkin
