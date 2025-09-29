# app.py — EchoSoul (final update with Call mode, wallpaper upload, and UI fixes)
import streamlit as st
import os, json, datetime, tempfile, base64
from openai import OpenAI

# ----------------------
# Initialization & Config
# ----------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
DATA_FILE = "echosoul_data.json"
WALLPAPER_FILE = "echosoul_wallpaper.bin"

# ----------------------
# Storage helpers
# ----------------------
def default_data():
    return {
        "user_name": "User",
        "memories": [],
        "vault": {},
        "chat_history": [],   # list of dicts: {"role": "user"/"assistant", "text": "...", "ts": "..."}
        "voice_mode": "alloy",
        "mimic_voice": None
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
        json.dump(data, f, indent=2)

data = load_data()

# ----------------------
# Small utilities
# ----------------------
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def add_chat(role, text):
    data["chat_history"].append({"role": role, "text": text, "ts": ts_now()})
    save_data(data)

# ----------------------
# Simple vault XOR encryption (prototype)
# ----------------------
def xor_encrypt_decrypt(text, key):
    if not key:
        return None
    return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(text))

# ----------------------
# OpenAI wrappers (prototype interface used earlier)
# ----------------------
def generate_reply_with_history(user_msg):
    # Build system prompt with persona and recent memories & history
    persona = "friendly"
    memories = "\n".join(data["memories"][-5:]) if data["memories"] else "No memories yet."
    # Build conversation history (last 8 messages)
    hist_msgs = []
    for turn in data["chat_history"][-8:]:
        role = "user" if turn["role"] == "user" else "assistant"
        hist_msgs.append({"role": role, "content": turn["text"]})
    system_prompt = (
        f"You are EchoSoul, an empathetic, adaptive digital companion for {data.get('user_name','User')}.\n"
        f"Persona tone: {persona}\n"
        f"Relevant memories:\n{memories}\n"
        "Maintain context, continue the conversation naturally, be concise and kind."
    )

    messages = [{"role": "system", "content": system_prompt}] + hist_msgs + [{"role": "user", "content": user_msg}]

    # Call the chat completion endpoint (prototype usage)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.75,
        max_tokens=300
    )
    reply = response.choices[0].message.content
    return reply

def synthesize_speech(text, voice="alloy", mimic_sample=None):
    """
    Returns bytes of an MP3 audio clip synthesized from `text`.
    If mimic_sample is provided (prototype), this function attempts to use it (placeholder).
    """
    # Example prototype call (the exact SDK method may differ in your environment)
    # For prototyping we call model 'gpt-4o-mini-tts' per earlier examples.
    # The return is assumed to be bytes or a stream we can write out.
    if mimic_sample:
        # prototype behavior: the API call would include voice cloning parameters
        response = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    else:
        response = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
    # In the prototypical wrapper we assume .read() returns bytes
    audio_bytes = response.read()
    return audio_bytes

def transcribe_audio_file(path):
    # Prototype transcription call
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
        return resp.text

# ----------------------
# UI helpers: background wallpaper
# ----------------------
def save_wallpaper(file):
    # Save raw file bytes to disk and set CSS background via base64
    content = file.getvalue()
    with open(WALLPAPER_FILE, "wb") as f:
        f.write(content)
    # Return base64 string
    return base64.b64encode(content).decode("utf-8")

def clear_wallpaper():
    if os.path.exists(WALLPAPER_FILE):
        os.remove(WALLPAPER_FILE)

def get_wallpaper_css():
    if os.path.exists(WALLPAPER_FILE):
        with open(WALLPAPER_FILE, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        return css
    return ""

# ----------------------
# Theme & CSS (neon + orb)
# ----------------------
BASE_CSS = """
<style>
body { background-color: #0D0D2E; color: #F0F0F0; font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #0D0D2E 60%, #121212); color: #F0F0F0; }
.stSidebar { background: rgba(20,20,40,0.78); backdrop-filter: blur(12px); border-right: 1px solid rgba(0,255,255,0.08); padding: 16px; }
h1,h2,h3 { color:#00FFFF; text-shadow: 0 0 8px rgba(0,255,255,0.4); }
.stButton>button { background-color:#091021; border:1px solid #00FFFF; color:#F0F0F0; border-radius:10px; padding:8px 12px; }
.stButton>button:hover { background:#00FFFF; color:#0D0D2E; box-shadow:0 0 12px #00FFFF; }
.stTextInput>div>div>input, .stTextArea>div>textarea { background: rgba(20,20,40,0.5); border:1px solid rgba(0,255,255,0.15); color:#F0F0F0; border-radius:8px; }
#ai-status { position: fixed; bottom: 20px; right: 25px; width: 22px; height: 22px; border-radius: 50%; background: radial-gradient(circle,#00ffcc,#009999); box-shadow: 0 0 12px #00ffff, 0 0 24px #00ffff; animation: pulse 2s infinite; z-index: 9999; }
@keyframes pulse { 0%{transform:scale(1);opacity:.9} 50%{transform:scale(1.3);opacity:.6} 100%{transform:scale(1);opacity:.9} }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown(get_wallpaper_css(), unsafe_allow_html=True)

# ----------------------
# AI status rendering
# ----------------------
if "ai_status" not in st.session_state:
    st.session_state.ai_status = "ready"  # ready | thinking | listening | on_call

def render_ai_status():
    color_map = {
        "ready": "radial-gradient(circle,#00ffcc,#009999)",
        "thinking": "radial-gradient(circle,#ff00ff,#990099)",
        "listening": "radial-gradient(circle,#00ccff,#0033ff)",
        "on_call": "radial-gradient(circle,#ffd700,#ff8800)"
    }
    style = f"<div id='ai-status' style='background:{color_map.get(st.session_state.ai_status)}'></div>"
    st.markdown(style, unsafe_allow_html=True)

# ----------------------
# Sidebar (detailed)
# ----------------------
with st.sidebar:
    st.markdown("## 🌌 EchoSoul Control Center")
    st.text_input("Your name", value=data.get("user_name", "User"), key="sidebar_name")
    if st.button("💾 Save Profile"):
        data["user_name"] = st.session_state.sidebar_name
        save_data(data)
        st.success("Profile updated.")

    st.markdown("---")
    st.markdown("### 🎙 Voice Settings")
    # voice selection radios
    v = st.radio("Choose AI voice", ["alloy", "verse", "amber"], index=["alloy", "verse", "amber"].index(data.get("voice_mode","alloy")))
    data["voice_mode"] = v

    st.caption("Upload a short voice sample to enable mimic mode (prototype).")
    mimic_file = st.file_uploader("Upload sample voice to mimic (mp3/wav)", type=["mp3","wav"])
    if mimic_file:
        # store raw file bytes in data (prototype — don't include huge files)
        try:
            data["mimic_voice"] = {"name": mimic_file.name, "bytes": base64.b64encode(mimic_file.getvalue()).decode("utf-8")}
            save_data(data)
            st.success("Mimic sample saved (prototype).")
        except Exception as e:
            st.error("Could not save mimic sample: " + str(e))

    st.markdown("---")
    st.markdown("### 🖼 Background Wallpaper")
    st.caption("Upload an image from your gallery and it becomes the app background.")
    wallpaper = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg","jpeg","png"])
    if wallpaper:
        try:
            save_wallpaper(wallpaper)
            st.success("Wallpaper set. If you don't see it, refresh the page.")
            st.experimental_rerun()
        except Exception as e:
            st.error("Wallpaper save failed: " + str(e))

    if st.button("Clear wallpaper"):
        clear_wallpaper()
        st.success("Wallpaper cleared.")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### ⚙ Settings")
    st.checkbox("Enable adaptive learning (persona update)", value=True, key="adaptive_toggle")
    st.checkbox("Show accessibility hints", value=True, key="access_hints")

    st.markdown("---")
    st.markdown("### 🆘 Help & Ethics")
    st.caption("🔒 Data kept locally in app storage. Voice mimic is prototype — only use your own voice sample.")

# ----------------------
# App header and navigation
# ----------------------
st.title(f"✨ EchoSoul — Hi {data.get('user_name', 'User')}")
nav = st.sidebar.radio("Navigate", ["Home", "Chat", "Voice Call", "Life Timeline", "Vault", "Export", "About"], index=1)

render_ai_status()

# ----------------------
# HOME
# ----------------------
if nav == "Home":
    st.header("Welcome")
    st.write("EchoSoul is your personal AI companion. Use Chat for text, Voice Call for a live-call experience, Timeline to save memories, Vault for secrets, and Export for legacy.")
    render_ai_status()

# ----------------------
# TEXT CHAT UI
# ----------------------
elif nav == "Chat":
    st.header("Text Chat")
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # input field stored in session_state so we can clear it after send
    chat_col, send_col = st.columns([8,1])
    with chat_col:
        st.text_input("Say something to EchoSoul", key="chat_input", placeholder="Type here...")
    with send_col:
        if st.button("Send"):
            user_text = st.session_state.chat_input.strip()
            if user_text:
                # set status to thinking and render
                st.session_state.ai_status = "thinking"
                render_ai_status()

                # add user message to chat_history and get reply
                add_chat("user", user_text)
                reply = generate_reply_with_history(user_text)
                add_chat("assistant", reply)

                # clear input
                st.session_state.chat_input = ""

                # show assistant message
                st.success("EchoSoul replied:")
                st.write(reply)

                # back to ready
                st.session_state.ai_status = "ready"
                render_ai_status()
            else:
                st.warning("Please type something before sending.")

    st.markdown("#### Conversation (latest)")
    # show only text lines — avoid raw data display that produced NULLs earlier
    for turn in data["chat_history"][-12:]:
        role = "You" if turn["role"] == "user" else "EchoSoul"
        if role == "You":
            st.markdown(f"**You:** {turn['text']}")
        else:
            st.markdown(f"**EchoSoul:** {turn['text']}")

    render_ai_status()

# ----------------------
# VOICE CALL MODE (live-call feel)
# ----------------------
elif nav == "Voice Call":
    st.header("Voice Call — live conversation")
    # call controls
    if "on_call" not in st.session_state:
        st.session_state.on_call = False

    call_col1, call_col2, call_col3 = st.columns([1,1,1])
    with call_col1:
        if not st.session_state.on_call:
            if st.button("📞 Start Call"):
                st.session_state.on_call = True
                st.session_state.ai_status = "on_call"
                render_ai_status()
                st.success("Call started. Speak using the mic below; EchoSoul will respond with voice.")
        else:
            if st.button("⛔ End Call"):
                st.session_state.on_call = False
                st.session_state.ai_status = "ready"
                render_ai_status()
                st.info("Call ended.")
    with call_col2:
        if st.session_state.on_call:
            st.write("Status: ON CALL")
        else:
            st.write("Status: Not on call")

    # While on call: accept audio input and respond with voice automatically
    if st.session_state.on_call:
        st.session_state.ai_status = "listening"
        render_ai_status()
        st.caption("Speak into the mic. After you stop, EchoSoul will transcribe and reply automatically.")

        audio = st.audio_input("🎙️ Speak now (press stop when done)")
        if audio:
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.getbuffer())
                tmp_path = tmp.name

            # Transcribe audio
            try:
                st.session_state.ai_status = "thinking"
                render_ai_status()
                user_text = transcribe_audio_file(tmp_path)
                add_chat("user", user_text)
                st.markdown(f"**You said:** {user_text}")

                # Generate reply
                reply = generate_reply_with_history(user_text)
                add_chat("assistant", reply)
                st.markdown(f"**EchoSoul:** {reply}")

                # Synthesize speech using selected voice or mimic sample if present
                mimic_sample = data.get("mimic_voice")
                voice_choice = data.get("voice_mode", "alloy")
                try:
                    audio_bytes = synthesize_speech(reply, voice=voice_choice, mimic_sample=mimic_sample)
                    # Play audio
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.error("Speech generation failed (prototype): " + str(e))

            except Exception as e:
                st.error("Transcription or reply failed: " + str(e))
            finally:
                st.session_state.ai_status = "on_call"
                render_ai_status()

    render_ai_status()

# ----------------------
# Life Timeline
# ----------------------
elif nav == "Life Timeline":
    st.header("Life Timeline")
    new_mem = st.text_area("Add a new memory")
    if st.button("➕ Save Memory"):
        if new_mem.strip():
            data["memories"].append(new_mem.strip())
            save_data(data)
            st.success("Memory saved.")
        else:
            st.warning("Type something to save as memory.")
    st.markdown("**Your saved memories:**")
    for m in data["memories"][-50:]:
        st.write("• " + m)
    render_ai_status()

# ----------------------
# Vault
# ----------------------
elif nav == "Vault":
    st.header("Private Vault (prototype)")
    vault_pw = st.text_input("Vault password", type="password")
    vault_text = st.text_area("Secret note to store")
    if st.button("Encrypt & Save"):
        if vault_pw and vault_text:
            enc = xor_encrypt_decrypt(vault_text, vault_pw)
            data["vault"]["note"] = enc
            save_data(data)
            st.success("Saved to vault (prototype).")
        else:
            st.warning("Provide both a password and a note.")
    if st.button("Decrypt & Show"):
        if vault_pw and data.get("vault", {}).get("note"):
            try:
                dec = xor_encrypt_decrypt(data["vault"]["note"], vault_pw)
                st.info(f"Decrypted note: {dec}")
            except Exception as e:
                st.error("Decryption failed: " + str(e))
        else:
            st.warning("No vault note or password missing.")
    render_ai_status()

# ----------------------
# Export
# ----------------------
elif nav == "Export":
    st.header("Export / Legacy")
    st.download_button("📥 Download your EchoSoul data (JSON)", data=json.dumps(data, indent=2), file_name="echosoul_data.json")
    st.markdown("**Legacy snapshot (recent chat):**")
    # show only last 10 turns neatly
    for t in data["chat_history"][-10:]:
        who = "You" if t["role"] == "user" else "EchoSoul"
        st.write(f"**{who}:** {t['text']}")
    render_ai_status()

# ----------------------
# About
# ----------------------
elif nav == "About":
    st.header("About EchoSoul")
    st.write("EchoSoul is a prototype personal AI companion with text & voice I/O, memory, vault, and a neon UI.")
    st.markdown("**Privacy & Ethics**: Data stored in-app. Voice mimic is prototype — don't upload other people's voices.")
    render_ai_status()

# ----------------------
# End of file
# ----------------------
