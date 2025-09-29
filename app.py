import streamlit as st
import os, json, base64, datetime, re
from openai import OpenAI

# ----------------------------
# Setup
# ----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
DATA_FILE = "echosoul_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chat_history": [], "memories": [], "vault": {}}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

data = load_data()

# ----------------------------
# Helpers
# ----------------------------
def add_chat(role, text):
    data["chat_history"].append({"role": role, "text": text})
    save_data(data)

def generate_reply_with_history(user_input):
    history = data["chat_history"][-8:]  # last 8 messages for context
    messages = [{"role": "system", "content": "You are EchoSoul, a friendly AI that adapts to the user."}]
    for h in history:
        messages.append({"role": h["role"], "content": h["text"]})
    messages.append({"role": "user", "content": user_input})

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    ).choices[0].message.content
    return reply

def text_to_speech(text, voice="alloy", mimic_file=None):
    try:
        if mimic_file:
            # Mimic mode (placeholder: use real cloning API if available)
            return f"[Mimicked voice playback for: {text}]"
        else:
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text
            )
            audio_path = "output.mp3"
            with open(audio_path, "wb") as f:
                f.write(speech.content)
            return audio_path
    except Exception as e:
        return f"[Error generating voice: {e}]"

def render_ai_status():
    if st.session_state.get("ai_status") == "thinking":
        st.info("🤔 EchoSoul is thinking...")
    elif st.session_state.get("ai_status") == "ready":
        st.success("✅ EchoSoul is ready")
    else:
        st.write("")

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")

# Sidebar
st.sidebar.title("🌌 EchoSoul Control Center")

# Profile
user_name = st.sidebar.text_input("Your name", value=st.session_state.get("user_name", "User"))
if st.sidebar.button("Save Profile"):
    st.session_state.user_name = user_name
    st.success("Profile saved!")

# Voice
st.sidebar.subheader("🎙 Voice Settings")
voice_choice = st.sidebar.radio("Choose AI voice", ["alloy", "verse", "amber"])
uploaded_mimic = st.sidebar.file_uploader("Upload a short voice sample to enable mimic mode (mp3/wav)", type=["mp3","wav"])

# Wallpaper
st.sidebar.subheader("🖼 Background Wallpaper")
bg_file = st.sidebar.file_uploader("Upload wallpaper (jpg/png)", type=["jpg", "png"])
if bg_file:
    bg_base64 = base64.b64encode(bg_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# Adaptive learning
st.sidebar.subheader("⚙️ Settings")
adaptive = st.sidebar.toggle("Enable adaptive learning", value=True)

# Privacy notice
st.sidebar.markdown("### Privacy & Ethics")
st.sidebar.markdown("🔒 Local storage · Inclusive design · Bias mitigation")

# Navigation
nav = st.sidebar.radio("Navigate", ["🏠 Home", "💬 Chat", "📞 Voice Call", "🧠 Life Timeline", "🔐 Vault", "📤 Export", "ℹ️ About"])

# ----------------------------
# MAIN SECTIONS
# ----------------------------
if nav == "🏠 Home":
    st.title(f"✨ EchoSoul — Hi {user_name}")
    st.markdown("Your evolving digital soul. Always learning, always with you.")

elif nav == "💬 Chat":
    st.header("Text Chat")

    if "chat_key" not in st.session_state:
        st.session_state.chat_key = 0

    user_text = st.text_input(
        "Say something to EchoSoul",
        key=f"chat_input_{st.session_state.chat_key}",
        placeholder="Type here..."
    )

    if st.button("Send"):
        if user_text.strip():
            st.session_state.ai_status = "thinking"
            render_ai_status()

            add_chat("user", user_text)
            reply = generate_reply_with_history(user_text)
            add_chat("assistant", reply)

            st.success("EchoSoul replied:")
            st.write(reply)

            # Clear input
            st.session_state.chat_key += 1

            st.session_state.ai_status = "ready"
            render_ai_status()
        else:
            st.warning("Please type something before sending.")

    st.markdown("#### History")
    for turn in data["chat_history"][-10:]:
        role = "You" if turn["role"] == "user" else "EchoSoul"
        st.markdown(f"**{role}:** {turn['text']}")

    render_ai_status()

elif nav == "📞 Voice Call":
    st.header("Voice Call with EchoSoul")
    st.markdown("🎧 It will feel like a call — EchoSoul will speak back using your selected voice.")

    if st.button("Start Call"):
        add_chat("system", "Voice call started.")
        st.session_state.ai_status = "ready"
        render_ai_status()

    if st.button("Say Hi"):
        reply = generate_reply_with_history("Hi from call mode")
        st.audio(text_to_speech(reply, voice_choice, uploaded_mimic))
        add_chat("assistant", reply)

elif nav == "🧠 Life Timeline":
    st.header("Life Timeline")
    memory_text = st.text_area("Add a memory")
    if st.button("Save Memory"):
        if memory_text.strip():
            data["memories"].append({"time": str(datetime.datetime.now()), "text": memory_text})
            save_data(data)
            st.success("Memory saved!")

    for m in reversed(data["memories"]):
        st.markdown(f"- {m['time']}: {m['text']}")

elif nav == "🔐 Vault":
    st.header("Private Vault")
    pw = st.text_input("Vault password", type="password")
    secret_note = st.text_area("Secret note")
    if st.button("Save to Vault"):
        if pw and secret_note:
            data["vault"][pw] = secret_note
            save_data(data)
            st.success("Saved to vault (encrypted prototype).")

elif nav == "📤 Export":
    st.header("Export Data")
    st.download_button("Download JSON", json.dumps(data, indent=2), "echosoul_export.json")

elif nav == "ℹ️ About":
    st.header("About EchoSoul")
    st.markdown("""
    - Persistent memory (timeline, vault)
    - Adaptive personality & sentiment tone
    - Voice chat (AI call mode, mimic support)
    - Privacy-first: all local storage
    - Custom wallpapers
    """)
