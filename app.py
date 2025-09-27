"""
EchoSoul — Full app.py
A single-file Streamlit app implementing:
 - Onboarding (name, age, hobbies, free-time) the first time the user opens the app
 - Persistent memory stored in echosoul_data.json
 - GPT-based conversational AI (OpenAI) with roleplay and persona
 - Chat input auto-clears after send (st.chat_input)
 - Auto-memory capture for user facts
 - Life Timeline, Private Vault (encrypted), Legacy export
 - Voice upload/transcription (optional; requires OPENAI_API_KEY and API support)
 - Adaptive personality via simple sentiment heuristics
 - Light / Dark theme toggle (UI-level)
 - Explainability / trust UI hints (small, inline)
IMPORTANT:
 - Put your OpenAI API key into Streamlit Secrets: OPENAI_API_KEY = "sk-..."
 - Do NOT put secrets into GitHub.
 - For stronger encryption, install `cryptography`. Otherwise a XOR fallback is used (demo only).
"""

import streamlit as st
import os, json, hashlib, base64, datetime, re, io, textwrap
from typing import Optional, List

# ---- OpenAI client ----
# Put your key in Streamlit Secrets (Manage app → Secrets)
client = None
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    client = None  # app will still run, but GPT/transcription will show a helpful message

# ---- Optional strong encryption via Fernet ----
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

# ---- Storage path ----
DATA_FILE = "echosoul_data.json"

# ---- Utility helpers ----
def ts_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data() -> dict:
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
        "conversations": []
    }

def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data()
    return default_data()

def save_data(data: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---- Encryption helpers ----
def _derive_key(password: str, length: int) -> bytes:
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def encrypt_xor(password: str, plaintext: str) -> str:
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_xor(password: str, ciphertext_b64: str) -> Optional[str]:
    try:
        data = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data))
        x = bytes([data[i] ^ key[i] for i in range(len(data))])
        return x.decode("utf-8")
    except Exception:
        return None

def _fernet_key_from_password(password: str) -> bytes:
    # For prototype only: deterministic key derived from password.
    # For production use PBKDF2HMAC with salt and iterations.
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)

def encrypt_fernet(password: str, plaintext: str) -> str:
    key = _fernet_key_from_password(password)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

def decrypt_fernet(password: str, token: str) -> Optional[str]:
    try:
        key = _fernet_key_from_password(password)
        f = Fernet(key)
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception:
        return None

def encrypt_text(password: str, plaintext: str) -> str:
    if USE_FERNET:
        return encrypt_fernet(password, plaintext)
    else:
        return encrypt_xor(password, plaintext)

def decrypt_text(password: str, blob: str) -> Optional[str]:
    if USE_FERNET:
        return decrypt_fernet(password, blob)
    else:
        return decrypt_xor(password, blob)

# ---- Memory helpers ----
def add_memory(data: dict, title: str, content: str, tags: Optional[List[str]] = None) -> dict:
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

def find_relevant_memories(data: dict, text: str, limit: int = 3) -> List[dict]:
    found = []
    txt = text.lower()
    for item in reversed(data["timeline"]):
        if any(w in txt for w in re.findall(r"\w+", item["content"].lower())) or any(w in txt for w in re.findall(r"\w+", item["title"].lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# ---- Sentiment heuristics for adaptive personality ----
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text: str) -> float:
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    norm = score / max(1, len(toks))
    return norm

def sentiment_label(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"

def update_persona_based_on_sentiment(data: dict, score: float):
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# ---- OpenAI integration helpers ----
def call_gpt(system_prompt: str, user_msg: str) -> str:
    if client is None:
        return "⚠️ OpenAI not configured. Add OPENAI_API_KEY to Streamlit Secrets to enable AI replies."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8,
            max_tokens=600
        )
        # Response parsing - depending on SDK, this may vary
        # Common: resp.choices[0].message.content
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            c = resp.choices[0]
            # Many SDK objects have .message or .text
            if hasattr(c, "message") and hasattr(c.message, "content"):
                return c.message.content
            if hasattr(c, "text"):
                return c.text
        # Fallback to string
        return str(resp)
    except Exception as e:
        return f"⚠️ GPT call error: {e}"

def transcribe_audio_bytes(audio_bytes: bytes, filename: str) -> Optional[str]:
    if client is None:
        return None
    try:
        # Many SDKs accept file-like for audio transcription; this code attempts to be portable.
        file_obj = io.BytesIO(audio_bytes)
        # Model id may vary; using a transcription-capable model name placeholder.
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_obj)
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        return str(resp)
    except Exception:
        return None

# ---- Main reply generator (uses GPT and memory) ----
def generate_reply(data: dict, user_msg: str) -> str:
    # sentiment / persona
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)

    # collect last N memories
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-8:]]
    mem_text = "\n".join(memories) if memories else "No memories yet."

    tone = data["profile"]["persona"].get("tone", "friendly")
    profile_summary = f"Name: {data['profile'].get('name')}\nAge: {data['profile'].get('age')}\nHobbies: {data['profile'].get('hobbies')}\nFree time: {data['profile'].get('free_time')}"

    system_prompt = textwrap.dedent(f"""
    You are EchoSoul, a personal, warm, and adaptive digital companion.
    Use the user's profile and memory to answer naturally and continue a conversation flow.
    Profile summary:
    {profile_summary}

    Recent memories (most recent first):
    {mem_text}

    Behavior instructions:
    - Keep replies conversational and in the user's context, refer back to earlier messages when relevant.
    - If asked to "act like me", roleplay as the user using the profile and memories.
    - Keep answers helpful, concise, and warm. Ask follow-up questions to continue dialogue when appropriate.
    - When you are unsure, say you don't know and offer to help find out.
    - Provide brief explainability when you make a suggestion (one sentence).
    """)

    reply = call_gpt(system_prompt, user_msg)

    # auto-save personal facts (heuristic)
    low = user_msg.lower()
    fact_patterns = [
        r"\bmy name is ([a-zA-Z\u00C0-\u017F0-9' \-]+)",
        r"\bi am ([0-9]{1,3})\b",  # age capture (simple)
        r"\bi'm ([0-9]{1,3})\b",
        r"\bi like ([a-zA-Z ,']+)",
        r"\bremember that (.+)"
    ]
    for pat in fact_patterns:
        m = re.search(pat, low)
        if m:
            val = m.group(1).strip()
            add_memory(data, "User fact", f"{m.group(0).strip()}")
            break

    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ---- Onboarding flow ----
def run_onboarding(data: dict):
    st.header("Let's get to know each other — quick setup")
    st.write("I'll ask four short questions so EchoSoul can remember you and talk like you. You can edit these later in Profile settings.")
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
        hobbies = st.text_area("What are your hobbies? (separate with commas)", value=(data["profile"].get("hobbies") or ""))
        if st.button("Next"):
            if hobbies.strip():
                data["profile"]["hobbies"] = hobbies.strip()
                add_memory(data, "Hobbies", hobbies.strip())
                st.session_state.onb_step = 3
                save_data(data)
                st.experimental_rerun()
            else:
                st.warning("Please enter something.")
    elif step == 3:
        free_time = st.text_area("What do you like to do in your free time?", value=(data["profile"].get("free_time") or ""))
        if st.button("Finish"):
            if free_time.strip():
                data["profile"]["free_time"] = free_time.strip()
                add_memory(data, "Free time", free_time.strip())
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.success("All set — I will remember these details.")
                st.session_state.onb_step = 0
                st.experimental_rerun()
            else:
                st.warning("Please tell me one or two things you like to do.")

# ---- UI / Layout ----
st.set_page_config(page_title="EchoSoul", layout="wide")
data = load_data()

# Top bar / theme toggle
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1 style='margin-bottom:0.2rem'>✨ EchoSoul — Your personal companion</h1>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.1rem;color:var(--secondary)'>A trusted memory-backed conversational AI.</div>", unsafe_allow_html=True)
with col2:
    theme_choice = st.selectbox("Theme", ["Dark (default)", "Light"], index=0)
    # Apply a tiny cosmetic style when Light is chosen using CSS
    if theme_choice == "Light":
        st.markdown("""
        <style>
        .reportview-container { background-color: #FFFFFF; color: #111111; }
        .stApp { background-color: #FFFFFF; color: #111111; }
        </style>
        """, unsafe_allow_html=True)

# Sidebar (detailed as per user's design)
with st.sidebar:
    st.markdown("## Profile")
    st.write("Your profile info (used to personalize replies).")
    current_name = data["profile"].get("name") or ""
    new_name = st.text_input("Display name", value=current_name)
    if st.button("Save name"):
        if new_name.strip():
            data["profile"]["name"] = new_name.strip()
            add_memory(data, "Name", f"My name is {new_name.strip()}")
            save_data(data)
            st.success("Saved name.")
        else:
            st.warning("Name cannot be empty.")

    st.markdown("---")
    st.markdown("## Voice & Chat")
    st.write("Upload a short voice sample for voice-mode or transcription.")
    st.write("Voice calls (TTS) are supported if you upload an audio sample and configure a TTS engine externally.")
    sample_upload = st.file_uploader("Upload sample voice (optional) — mp3, wav, m4a", type=["mp3","wav","m4a","ogg"])
    st.markdown("---")
    st.markdown("## Vault (prototype)")
    st.write("Set a session vault password (stored only in this session) to encrypt/decrypt private notes.")
    if "vault_password" not in st.session_state:
        st.session_state.vault_password = ""
    vault_pw = st.text_input("Vault password (session)", type="password", key="vault_pw")
    if st.button("Clear session password"):
        st.session_state.vault_password = ""
        st.success("Vault session password cleared.")

    st.markdown("---")
    st.markdown("## Settings & Explainability")
    st.checkbox("Enable adaptive learning (persona updates)", value=True, key="adaptive_toggle")
    st.write("Explainability hints will appear inline (confidence when available).")
    st.markdown("---")
    st.markdown("## Deployment & Privacy")
    st.write("This app stores data locally (echosoul_data.json). Keep your API key in Streamlit Secrets, do not share it in the repo.")
    st.markdown("---")
    st.write("Need help? Use the About tab for guidance.")

# Onboarding
if not data["profile"].get("intro_completed", False):
    run_onboarding(data)
    st.stop()

# Main tabs
tabs = st.tabs(["Chat", "History", "Timeline", "Vault", "Export", "About"])
tab_chat, tab_history, tab_timeline, tab_vault, tab_export, tab_about = tabs

with tab_chat:
    st.header(f"Chat — Hi {data['profile'].get('name','there')}")
    st.write("This chat feels like an ongoing dialogue. After you send a message, the input clears automatically.")
    # show quick suggested prompts (small chips)
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        if st.button("Hi"):
            user_input = "Hi"
            reply = generate_reply(data, user_input)
            st.experimental_rerun()
    with qcol2:
        if st.button("How are you ?"):
            user_input = "How are you ?"
            reply = generate_reply(data, user_input)
            st.experimental_rerun()
    with qcol3:
        if st.button("Act like you are me"):
            user_input = "Act like you are me and reply like I would."
            reply = generate_reply(data, user_input)
            st.experimental_rerun()
    with qcol4:
        if st.button("Remember my name"):
            user_input = f"My name is {data['profile'].get('name','')}"
            add_memory(data, "User fact", user_input)
            st.success("Saved as memory.")
            st.experimental_rerun()

    # Audio upload (transcription)
    audio_file = st.file_uploader("Or upload voice to transcribe and chat (optional)", type=["mp3","wav","m4a","ogg"])
    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.info("Transcribing audio...")
        transcription = transcribe_audio_bytes(audio_bytes, audio_file.name)
        if transcription:
            st.success("Transcription complete")
            # Append to conversation and generate reply
            data["conversations"].append({"user": transcription, "bot": "[transcription posted]", "ts": ts_now()})
            save_data(data)
            reply = generate_reply(data, transcription)
            st.markdown(f"**You (transcribed):** {transcription}")
            st.markdown(f"**EchoSoul:** {reply}")
        else:
            st.error("Transcription failed or OpenAI not configured.")

    # show last 20 messages
    conv = data.get("conversations", [])[-20:]
    for message in conv:
        st.markdown(f"**You:** {message['user']}")
        st.markdown(f"**EchoSoul:** {message['bot']}")
        st.markdown("---")

    # Chat input that auto-clears
    user_input = st.chat_input("Say something to EchoSoul...")
    if user_input:
        with st.spinner("EchoSoul is thinking..."):
            reply = generate_reply(data, user_input)
        # show the reply right away by rerunning
        st.experimental_rerun()

with tab_history:
    st.header("Conversation History")
    conv = data.get("conversations", [])
    if not conv:
        st.info("No conversations yet.")
    else:
        for m in conv[::-1]:
            st.markdown(f"**You:** {m['user']}")
            st.markdown(f"**EchoSoul:** {m['bot']}")
            st.markdown("---")

with tab_timeline:
    st.header("Life Timeline")
    st.write("Chronological record of memories, automatically saved or added manually.")
    if data["timeline"]:
        for item in sorted(data["timeline"], key=lambda x: x["timestamp"], reverse=True):
            st.markdown(f"**{item['title']}** — {item['timestamp']}")
            st.write(item["content"])
            st.markdown("---")
    else:
        st.info("No memories yet. They'll appear here as you add them or say 'remember'.")

    st.markdown("### Add memory manually")
    mt = st.text_input("Title", key="manual_title")
    mc = st.text_area("Content", key="manual_content")
    if st.button("Save Memory"):
        if mc.strip():
            add_memory(data, mt or "Memory", mc.strip())
            st.success("Memory saved.")
            st.experimental_rerun()
        else:
            st.warning("Please add some content.")

with tab_vault:
    st.header("Private Vault")
    st.write("Store sensitive notes here. Use the same password to decrypt later. If cryptography is installed, Fernet is used (stronger). Otherwise XOR demo is used (not secure).")
    if not st.session_state.get("vault_password"):
        st.warning("Set a session vault password in the sidebar to use the vault.")
    pw = st.session_state.get("vault_password") or st.text_input("Session vault password (temporary)", type="password", key="vpass_tmp")
    # If user ty
