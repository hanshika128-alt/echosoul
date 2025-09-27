"""
EchoSoul — Streamlit app.py (complete)
- Onboarding (name, age, hobbies, free-time)
- Persistent Memory (echosoul_data.json)
- GPT replies (gpt-4o-mini) when OPENAI_API_KEY is present
- Auto-clear chat input via st.chat_input()
- Adaptive persona (sentiment-based)
- Life Timeline, Manual memories, Private Vault (XOR prototype or Fernet if cryptography installed)
- Legacy export
- Neon visual theme + optional Light theme toggle
- Explainability hints (sentiment/confidence)
- Voice upload for transcription (optional; requires OpenAI configured)
Notes:
 - Put your OPENAI_API_KEY into Streamlit Secrets: OPENAI_API_KEY = "sk-..."
 - If cryptography installed, Fernet will be used for vault; otherwise demo XOR fallback.
"""

import streamlit as st
import os, json, datetime, hashlib, base64, re, io, textwrap
from typing import Optional

# -------------------------
# Basic config & CSS
# -------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")

NEON_CSS = """
<style>
/* Base app */
.stApp { background-color: #0D0D2E; color: #E6F7F7; font-family: Inter, Poppins, sans-serif; }

/* Neon headers */
h1, h2, h3 { color: #00f0ea !important; text-shadow: 0 0 12px rgba(0,240,234,0.6); }

/* Card-style chat lines */
.chat-bubble {
  background: rgba(12, 18, 44, 0.55);
  border-radius: 12px;
  padding: 10px 14px;
  margin: 6px 0;
  box-shadow: 0 0 14px rgba(0,240,234,0.06);
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0b0c16; color: #E6F7F7; }

/* Buttons */
button[kind="primary"] {
  background: linear-gradient(90deg,#00f0ea,#7be0ff) !important;
  color: #00121a !important;
  font-weight: 600;
  border-radius: 8px;
  box-shadow: 0 0 12px rgba(0,240,234,0.22);
}

/* Inputs subtle */
input, textarea { background: rgba(255,255,255,0.03) !important; color: #E6F7F7 !important; }

/* Light theme overrides (applied if user selects Light) */
.light .stApp { background-color: #FFFFFF; color: #111827; }
.light h1,h2,h3 { color: #0B76FF !important; text-shadow: none !important; }
.light [data-testid="stSidebar"] { background-color: #F4F6F8 !important; color: #111827 !important; }
</style>
"""

st.markdown(NEON_CSS, unsafe_allow_html=True)

# -------------------------
# OpenAI client (safe)
# -------------------------
client = None
openai_available = False
try:
    # `from openai import OpenAI` style client (Streamlit Secrets must contain OPENAI_API_KEY)
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        openai_available = True
    else:
        client = None
        openai_available = False
except Exception:
    client = None
    openai_available = False

# -------------------------
# Encryption helpers (Fernet optional)
# -------------------------
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

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
    return encrypt_xor(password, plaintext)

def decrypt_text(password: str, blob: str) -> Optional[str]:
    if USE_FERNET:
        return decrypt_fernet(password, blob)
    return decrypt_xor(password, blob)

# -------------------------
# File / data helpers
# -------------------------
DATA_FILE = "echosoul_data.json"

def ts_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data() -> dict:
    return {
        "profile": {"name": None, "age": None, "hobbies": None, "free_time": None, "persona": {"tone":"friendly"}, "intro_completed": False},
        "timeline": [],
        "vault": [],
        "conversations": []
    }

def load_data() -> dict:
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                # Ensure keys exist (avoid KeyError)
                if "profile" not in d: d["profile"] = default_data()["profile"]
                if "timeline" not in d: d["timeline"] = []
                if "vault" not in d: d["vault"] = []
                if "conversations" not in d: d["conversations"] = []
                return d
    except Exception:
        pass
    return default_data()

def save_data(d: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

data = load_data()

# -------------------------
# Sentiment & persona helpers
# -------------------------
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
    if score > 0.05: return "positive"
    if score < -0.05: return "negative"
    return "neutral"

def update_persona(data: dict, score: float):
    if score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    elif score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# -------------------------
# GPT & transcription wrappers
# -------------------------
def call_gpt(system_prompt: str, user_msg: str) -> (str, Optional[float]):
    """
    Returns: (reply_text, confidence_score_or_None)
    Confidence is a heuristic based on sentiment magnitude and message length (not real model confidence).
    """
    if not openai_available:
        return ("⚠️ OpenAI key not set in Streamlit Secrets. Add OPENAI_API_KEY to enable AI.", None)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": system_prompt},
                {"role":"user","content": user_msg}
            ],
            temperature=0.7,
            max_tokens=600
        )
        # Parse result robustly
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            # Many SDKs: choice.message.content
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content
            elif hasattr(choice, "text"):
                text = choice.text
            else:
                text = str(choice)
        else:
            text = str(response)
        # heuristic confidence: longer reply + neutral sentiment -> higher (just UI hint)
        conf = min(0.99, max(0.05, min(0.95, len(text)/800)))
        return (text, conf)
    except Exception as e:
        return (f"⚠️ GPT error: {e}", None)

def transcribe_audio(audio_bytes: bytes, filename: str) -> Optional[str]:
    if not openai_available:
        return None
    try:
        file_obj = io.BytesIO(audio_bytes)
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_obj)
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        return str(resp)
    except Exception:
        return None

# -------------------------
# Onboarding UI
# -------------------------
if "onboard_done" not in st.session_state:
    st.session_state.onboard_done = data["profile"].get("intro_completed", False)
if not st.session_state.onboard_done:
    st.header("Welcome — quick setup")
    st.write("I'll ask a few short questions so EchoSoul can remember you. You can edit these later in Profile.")
    if "onb_step" not in st.session_state:
        st.session_state.onb_step = 0
    questions = [
        ("name", "What's your name?"),
        ("age", "What's your age?"),
        ("hobbies", "What are your hobbies?"),
        ("free_time", "What do you like to do in your free time?")
    ]
    key, qtext = questions[st.session_state.onb_step]
    answer = st.text_input(qtext, value=(data["profile"].get(key) or ""), key=f"onb_{st.session_state.onb_step}")
    if st.button("Next"):
        if answer.strip():
            data["profile"][key] = answer.strip()
            # Add to timeline as memory
            data["timeline"].append({"id": hashlib.sha1((key+answer+ts_now()).encode()).hexdigest(), "title": f"{key.capitalize()}", "content": answer.strip(), "timestamp": ts_now()})
            save_data(data)
            st.session_state.onb_step += 1
            if st.session_state.onb_step >= len(questions):
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.session_state.onboard_done = True
        else:
            st.warning("Please write an answer to continue.")
        st.rerun()
    st.stop()

# -------------------------
# Main layout: Sidebar & Tabs
# -------------------------
# Sidebar design per your spec
with st.sidebar:
    st.markdown("<h2 style='color:#00f0ea'>Profile</h2>", unsafe_allow_html=True)
    pname = st.text_input("Display name", value=data["profile"].get("name") or "")
    if st.button("Save profile name"):
        data["profile"]["name"] = pname.strip() or data["profile"]["name"]
        data["timeline"].append({"id": hashlib.sha1((pname+ts_now()).encode()).hexdigest(), "title":"Name updated", "content": f"Display name set to {pname}", "timestamp": ts_now()})
        save_data(data)
        st.success("Saved profile name.")

    st.markdown("---")
    st.markdown("<h3 style='color:#00f0ea'>Voice & Chat</h3>", unsafe_allow_html=True)
    voice_sample = st.file_uploader("Upload voice sample (optional) — used for voice-mode", type=["mp3","wav","m4a","ogg"])
    st.markdown("---")
    st.markdown("<h3 style='color:#00f0ea'>Private Vault (prototype)</h3>", unsafe_allow_html=True)
    if "vault_pw" not in st.session_state:
        st.session_state.vault_pw = ""
    vault_pw = st.text_input("Vault password (session only)", type="password", key="vault_pw_input")
    if st.button("Set session vault password"):
        st.session_state.vault_pw = vault_pw
        st.success("Vault password set for this session (not saved to file).")
    st.markdown("Use the Vault tab to save private notes (encrypted locally).")
    st.markdown("---")
    st.markdown("<h3 style='color:#00f0ea'>Appearance</h3>", unsafe_allow_html=True)
    theme = st.selectbox("Theme", ["Neon Dark (default)", "Light"])
    if theme == "Light":
        # add a body class so CSS can style (we used .light earlier)
        st.markdown("<script>document.querySelector('body').classList.add('light')</script>", unsafe_allow_html=True)
    else:
        st.markdown("<script>document.querySelector('body').classList.remove('light')</script>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 style='color:#00f0ea'>Export & Help</h3>", unsafe_allow_html=True)
    if st.button("Export all data (download)"):
        st.download_button("Download JSON", json.dumps(data, indent=2), file_name=f"echosoul_export_{datetime.datetime.utcnow().date()}.json", mime="application/json")
    st.markdown("OpenAI: " + ("configured" if openai_available else "not configured"))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "History", "Timeline", "Vault", "About"])

# -------------------------
# Chat tab (core)
# -------------------------
with tab1:
    st.markdown("<h2 class=''>Chat / EchoSoul</h2>", unsafe_allow_html=True)
    st.write("EchoSoul keeps context, refers back to recent messages, and aims to continue an ongoing dialogue.")

    # quick explainability / persona status
    tone = data["profile"].get("persona", {}).get("tone", "friendly")
    st.info(f"Persona tone: **{tone}** — adaptive, changes with mood and conversation.")

    # Show last 12 messages (compact)
    convs = data.get("conversations", [])[-12:]
    for c in convs:
        st.mark.markdown if False else None  # placeholder to avoid linter complaint
        st.markdown(f"<div class='chat-bubble'><strong>You:</strong> {c['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><strong>EchoSoul:</strong> {c['bot']}</div>", unsafe_allow_html=True)

    # optional audio upload to transcribe
    audio = st.file_uploader("Upload short voice to transcribe and chat (optional)", type=["wav","mp3","m4a","ogg"], key="audio_chat_upload")
    if audio is not None:
        audio_bytes = audio.read()
        st.info("Attempting transcription...")
        transcription = transcribe_audio(audio_bytes, audio.name)
        if transcription:
            st.success("Transcription succeeded; sent as your message.")
            # treat transcription as a user message
            user_message = transcription
            # do reply below (reuse same flow)
        else:
            st.error("Transcription not available (OpenAI not configured or unsupported).")
            user_message = None
    else:
        user_message = None

    # Chat input using st.chat_input (auto clears)
    typed = st.chat_input("Say something to EchoSoul — it will clear after you send.")

    # If either typed or audio transcription exists, process it
    message_to_process = None
    if typed:
        message_to_process = typed
    elif user_message:
        message_to_process = user_message

    if message_to_process:
        # Save incoming text briefly
        score = sentiment_score(message_to_process)
        label = sentiment_label(score)
        update_persona(data, score)

        # Build system prompt using profile + recent memories
        # Use last 6 conversation turns as context
        recent_conv = data.get("conversations", [])[-6:]
        history_text = ""
        for turn in recent_conv:
            history_text += f"User: {turn['user']}\nEchoSoul: {turn['bot']}\n"
        mems = data.get("timeline", [])[-6:]
        mem_text = "\n".join([f"- {m['title']}: {m['content']}" for m in mems]) if mems else "No memories yet."

        system_prompt = textwrap.dedent(f"""
        You are EchoSoul, a friendly, memory-backed personal AI companion.
        Profile: name={data['profile'].get('name')}, age={data['profile'].get('age')}, hobbies={data['profile'].get('hobbies')}.
        Tone: {data['profile'].get('persona',{}).get('tone','friendly')}.
        Recent memories:
        {mem_text}

        Behavior:
        - Stay on topic and refer back to what you just said when relevant.
        - Continue the conversation as an ongoing dialogue (not disjoint Q&A).
        - If asked to 'act like me', roleplay the user using the profile and timeline.
        - Provide short explainability hints (one-line) about why a suggestion was made when giving advice.
        - If unsure, say so and offer a path to find out.
        """)

        reply_text, confidence = call_gpt(system_prompt, message_to_process)
        # Append to conversations and timeline (auto memory heuristics)
        data["conversations"].append({"user": message_to_process, "bot": reply_text, "ts": ts_now()})
        # Heuristics to store memory: "remember", "my name is", "i am X years", or short user facts
        low = message_to_process.lower()
        if re.search(r"\bremember\b", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b", low) or re.search(r"\bi'm\b", low):
            data["timeline"].append({"id": hashlib.sha1((message_to_process+ts_now()).encode()).hexdigest(), "title":"Auto-memory", "content": message_to_process, "timestamp": ts_now()})
        save_data(data)

        # Update persona tone (already done via update_persona)
        # Render reply and explainability hint
        st.markdown(f"<div class='chat-bubble'><strong>You:</strong> {message_to_process}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><strong>EchoSoul:</strong> {reply_text}</div>", unsafe_allow_html=True)
        if confidence is not None:
            st.caption(f"AI confidence (heuristic): {confidence:.2f} — sentiment: {label}")
        else:
            st.caption(f"Sentiment: {label} — (OpenAI not configured or confidence unavailable)")

        # after sending, do not set st.session_state chat keys (st.chat_input auto-clears). Rerun to reflect saved history.
        st.rerun()

# -------------------------
# History tab
# -------------------------
with tab2:
    st.header("Conversation History")
    conv = data.get("conversations", [])[::-1]
    if not conv:
        st.info("No conversations yet.")
    else:
        for item in conv:
            st.markdown(f"**You:** {item['user']}")
            st.markdown(f"**EchoSoul:** {item['bot']}")
            st.markdown(f"*{item.get('ts','')}*")
            st.markdown("---")

# -------------------------
# Timeline tab (memories)
# -------------------------
with tab3:
    st.header("Life Timeline — Memories")
    st.write("Memories are used as context in replies. You can add, edit, or delete items.")
    if data.get("timeline"):
        for idx, m in enumerate(sorted(data["timeline"], key=lambda x: x["timestamp"], reverse=True)):
            st.markdown(f"**{m['title']}**  ·  {m['timestamp']}")
            st.write(m["content"])
            if st.button(f"Delete memory {idx}", key=f"del_mem_{idx}"):
                # safe deletion by id
                data["timeline"] = [x for x in data["timeline"] if x.get("id") != m.get("id")]
                save_data(data)
                st.success("Deleted memory.")
                st.rerun()
            st.markdown("---")
    else:
        st.info("No memories found yet. EchoSoul will save obvious facts or you can add them manually below.")

    st.markdown("### Add memory manually")
    mt = st.text_input("Title", key="manual_title")
    mc = st.text_area("Content", key="manual_content")
    if st.button("Save Memory"):
        if mc.strip():
            data["timeline"].append({"id": hashlib.sha1((mt+mc+ts_now()).encode()).hexdigest(), "title": mt or "Memory", "content": mc.strip(), "timestamp": ts_now()})
            save_data(data)
            st.success("Saved memory.")
            st.rerun()
        else:
            st.warning("Please add content for the memory.")

# -------------------------
# Vault tab
# -------------------------
with tab4:
    st.header("Private Vault (Prototype)")
    st.write("Save sensitive notes. This uses XOR demo encryption unless `cryptography` is installed (Fernet). Ke
