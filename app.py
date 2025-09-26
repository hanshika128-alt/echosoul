"""
EchoSoul — Full single-file Streamlit app
Features:
- Onboarding: asks name, age, hobbies, free-time; stores permanently
- Conversational AI: OpenAI GPT (gpt-4o-mini) for replies; stays on topic; refers back to recent messages
- Chat input auto-clears (st.chat_input)
- Persistent Memory: echosoul_data.json (profile, timeline, vault, conversations)
- Adaptive personality: sentiment-based tone (positive→energetic, negative→empathetic, neutral→friendly)
- Private Vault: password protected (XOR fallback). For stronger encryption, install 'cryptography' to use Fernet.
- Legacy & Export: JSON export + human-readable snapshot
- Profile settings & sidebar UI
- Background customization from gallery
- Voice: upload audio to transcribe; generate TTS replies (gpt-4o-mini-tts). Saves voice replies (base64) if produced.
- Explainability: brief "why" note shown with replies (heuristic/placeholder); confidence estimate from heuristic.
- Designed for Streamlit Cloud (store API key in Secrets).
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

# --------- OpenAI client (reads key from Streamlit Secrets) ---------
client = None
try:
    from openai import OpenAI
    if "OPENAI_API_KEY" in st.secrets:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    else:
        # client remains None — app will display clear message
        client = None
except Exception:
    client = None

# --------- Optional strong encryption (Fernet) ---------
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

# --------- Data file ---------
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
        "timeline": [],         # memories: list of {id,title,content,timestamp}
        "vault": [],            # vault items: {title,cipher,timestamp}
        "conversations": [],    # convs: {user,bot,ts,voice_b64(optional),explain(optional)}
        "settings": {
            "bg_image_b64": None,
            "theme": "dark"
        }
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

# --------- Vault encryption helpers ---------
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

# XOR fallback (demo prototype only)
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

def decrypt_text(password, cipher):
    if USE_FERNET:
        return decrypt_with_fernet(password, cipher)
    return decrypt_xor(password, cipher)

# --------- Memory helpers ---------
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
        # simple overlap check
        if any(w in txt for w in re.findall(r"\w+", item["content"].lower())) or any(w in txt for w in re.findall(r"\w+", item["title"].lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# --------- Simple sentiment heuristic for adaptive personality ---------
POS_WORDS = {"good","great","happy","love","excellent","amazing","wonderful","nice","grateful","fun","delighted","excited","calm","optimistic"}
NEG_WORDS = {"bad","sad","angry","depressed","unhappy","terrible","awful","hate","lonely","anxious","stressed","worried","frustrated"}

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
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

# --------- OpenAI wrapper helpers ---------
def call_gpt(system_prompt: str, conversation: list, user_message: str, max_tokens=600, temperature=0.8):
    """
    system_prompt: big system message
    conversation: list of prior messages in dict format [{"role":"user","content":...}, ...] (we'll send trimmed history)
    user_message: the latest user message to append
    """
    if client is None:
        # clear, actionable message if API key missing
        return ("[No OpenAI API key configured. Add OPENAI_API_KEY in Streamlit Secrets to enable AI replies.]", None)

    try:
        # Build messages: system + conversation + user
        messages = [{"role":"system","content":system_prompt}]
        # keep token usage small: include last ~8 messages from conversation param
        # conversation items are stored as {"user":..., "bot":...}
        # We convert them to chat messages
        for conv in conversation[-8:]:
            messages.append({"role":"user","content":conv.get("user","")})
            messages.append({"role":"assistant","content":conv.get("bot","")})
        messages.append({"role":"user","content":user_message})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # choice parsing depends on SDK version; this aims to be flexible
        content = resp.choices[0].message.content if hasattr(resp.choices[0].message, "content") else resp.choices[0].message["content"]
        # We don't have a real confidence score from the model; produce a heuristic confidence number
        conf = min(0.99, max(0.2, 1 - len(user_message)/1000))  # heuristic: shorter messages → slightly higher heuristic confidence
        return content, conf
    except Exception as e:
        return f"[GPT error: {str(e)}]", None

# --------- Audio transcription (upload) and TTS generation ---------
def transcribe_audio_file(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    Uses OpenAI audio.transcriptions if client present.
    Returns transcription text or None.
    """
    if client is None:
        return None
    try:
        file_obj = io.BytesIO(file_bytes)
        # depending on SDK, call may differ; attempt common pattern
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_obj)
        # many SDKs return 'text' field
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        if hasattr(resp, "text"):
            return resp.text
        # fallback to str
        return str(resp)
    except Exception:
        return None

def generate_tts_bytes(text: str, voice: str = "alloy") -> Optional[bytes]:
    """
    Generate TTS using OpenAI speech API (gpt-4o-mini-tts).
    Returns raw bytes of mp3/wav or None if not available/error.
    """
    if client is None:
        return None
    try:
        # SDKs vary; common modern call:
        # resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
        # many SDK responses support .read() to get bytes. We'll attempt that pattern.
        resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
        # If resp is a stream-like object with .read()
        if hasattr(resp, "read"):
            return resp.read()
        if isinstance(resp, dict) and "data" in resp:
            # some SDKs return base64 audio or bytes field
            return base64.b64decode(resp["data"])
        # fallback: convert to string bytes
        return str(resp).encode("utf-8")
    except Exception:
        return None

# --------- Main reply generator (ties memory, persona, gpt, explainability, auto-save) ---------
def generate_reply(data, user_msg, tts_voice: Optional[str] = None):
    # update persona based on sentiment
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)
    tone = data["profile"]["persona"].get("tone", "friendly")

    # collect memory context (last up to 5 memories)
    memories = [f"{m['title']}: {m['content']}" for m in data["timeline"][-6:]]
    mem_text = "\n".join(memories) if memories else "No memories yet."

    # Build system prompt with clear instructions for staying on topic and referring back
    system_prompt = textwrap.dedent(f"""
    You are EchoSoul — a personal AI companion for the user.
    Personality tone: {tone}.
    Known memories (use as context and refer back where relevant):
    {mem_text}

    Conversation rules:
    - Stay on topic with the user's last messages and the stored memories.
    - Refer back to the conversation and to memories when helpful.
    - If asked to "act like me", adopt the user's persona using stored profile facts.
    - Provide short explainable notes (one-liners) of why you responded that way when asked.
    - Do not expose or leak private data unless the user explicitly requests it.
    """)

    # build conversation history for context (use stored conversations)
    conversation = data.get("conversations", [])

    # call GPT
    reply_text, conf = call_gpt(system_prompt, conversation, user_msg)
    if reply_text is None:
        reply_text = "[Error generating reply — check your API key and network.]"

    # Auto-save important facts (onboarding-like or explicit "remember" requests)
    low = user_msg.lower()
    # Patterns to auto-save: "remember that ...", "my name is ...", "i am ..." (with caution)
    if re.search(r"\bremember that\b", low) or re.search(r"\bremember\b\s+\w+", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b\s+\w+", low):
        # Save the full user_msg as a memory but with a clear title
        add_memory(data, "User fact", user_msg)

    # Save conversation with optional TTS
    conv_entry = {"user": user_msg, "bot": reply_text, "ts": ts_now()}
    if conf is not None:
        conv_entry["confidence_heuristic"] = float(conf)

    # TTS if requested
    if tts_voice and client is not None:
        audio_bytes = generate_tts_bytes(reply_text, voice=tts_voice)
        if audio_bytes:
            # Save base64 audio in the conversation entry for persistence
            conv_entry["voice_b64"] = base64.b64encode(audio_bytes).decode("utf-8")

    data.setdefault("conversations", []).append(conv_entry)
    save_data(data)

    # Explainability note (very short)
    explain_note = f"Refers to {len(memories)} memory item(s); tone set to {tone}."
    conv_entry["explain"] = explain_note
    save_data(data)

    return reply_text, explain_note, conv_entry.get("voice_b64", None)

# --------- Onboarding flow (first run) ---------
def run_onboarding(data):
    st.header("Welcome — let's get to know each other")
    st.write("I'll ask a few short questions so I can remember you and speak like you.")
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
        hobbies = st.text_area("What are your hobbies? (comma separated)", value=(data["profile"].get("hobbies") or ""))
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
        if st.button("Finish"):
            if free_time.strip():
                data["profile"]["free_time"] = free_time.strip()
                add_memory(data, "Free time", free_time.strip())
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.success("Thanks — I will remember these things.")
                # reset step
                st.session_state.onb_step = 0
                st.experimental_rerun()
            else:
                st.warning("Please tell me something you like doing in your free time.")

# --------- Streamlit UI ---------
st.set_page_config(page_title="EchoSoul", layout="wide")
data = load_data()

# Sidebar: UI layout and settings (detailed)
with st.sidebar:
    # Logo / branding
    st.markdown("<div style='display:flex;align-items:center;gap:8px'>"
                "<div style='width:48px;height:48px;border-radius:10px;background:#222;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700'>ES</div>"
                "<div><strong>EchoSoul</strong><br style='margin:0'/><small>Personal AI companion</small></div></div>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Profile")
    # Name field
    pname = data["profile"].get("name") or ""
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
    st.subheader("Vault & Security (prototype)")
    st.markdown("Set a vault password (stored only in this session) to lock/unlock private notes.")
    vault_password = st.text_input("Vault password (sidebar)", type="password")
    if st.button("Clear session password"):
        vault_password = ""
        st.session_state.pop("vault_password_session", None)
        st.info("Vault password cleared for this session.")
    # store vault password in session only (not written to disk)
    if vault_password:
        st.session_state["vault_password_session"] = vault_password

    st.markdown("---")
    st.subheader("Appearance")
    st.write("Customize background (choose an image from your device).")
    bg_file = st.file_uploader("Upload background", type=["png","jpg","jpeg"], key="bg_upload")
    if bg_file is not None:
        bbytes = bg_file.read()
        b64 = base64.b64encode(bbytes).decode("utf-8")
        data["settings"]["bg_image_b64"] = b64
        save_data(data)
        st.success("Background saved.")
    # Theme toggle
    theme = st.radio("Theme", ("dark","light"), index=0 if data["settings"].get("theme","dark")=="dark" else 1)
    data["settings"]["theme"] = theme
    save_data(data)

    st.markdown("---")
    st.subheader("Voice")
    st.write("TTS voice for replies (prototype). If you want a custom voice, you'll need a voice-cloning API key (not included).")
    tts_voice_choice = st.selectbox("TTS voice", ["alloy","verse","shimmer","default"], index=0)
    st.markdown("---")
    st.write("Deployment & Privacy")
    st.markdown("• Keep your OpenAI key in Streamlit Secrets (not the repo).")
    st.markdown("• This app stores data locally in `echosoul_data.json` inside the app.")
    st.markdown("• Vault uses XOR fallback unless `cryptography` is enabled (Fernet).")
    st.markdown("---")
    st.caption("Need help? Use the About tab in the app.")

# Set background style if provided
if data["settings"].get("bg_image_b64"):
    bg_b64 = data["settings"]["bg_image_b64"]
    st.markdown(
        f"<style>body{{background-image:url('data:image/png;base64,{bg_b64}');background-size:cover;}}</style>",
        unsafe_allow_html=True
    )

# If onboarding not complete, run onboarding
if not data["profile"].get("intro_completed", False):
    run_onboarding(data)
    st.stop()

# Top bar and tabs
st.title("EchoSoul — your personal AI companion")
tabs = st.tabs(["Chat", "Timeline", "Vault", "Legacy & Export", "About"])

# ----- CHAT TAB -----
with tabs[0]:
    st.header(f"Chat — {data['profile'].get('name') or 'friend'} (tone: {data['profile']['persona'].get('tone')})")

    # show last conversation snippets (scrollable)
    convs = data.get("conversations", [])
    if not convs:
        st.info("No conversations yet — say hello!")
    else:
        # show last 12 messages in order
        for entry in convs[-12:]:
            st.markdown(f"**You:** {entry.get('user')}")
            st.markdown(f"**EchoSoul:** {entry.get('bot')}")
            # show small explainability note if present
            ex = entry.get("explain")
            if ex:
                st.caption(f"Why: {ex}")
            # show confidence heuristic if present
            if "confidence_heuristic" in entry:
                st.caption(f"Confidence (heuristic): {entry['confidence_heuristic']:.2f}")
