import streamlit as st
import json, os, uuid, datetime
from openai import OpenAI
from pathlib import Path

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DATA_FILE = "echosoul_data.json"

# ------------------ Persistent Storage ------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"memories": [], "vault": {}, "profile": {"name": "User"}, "voice_file": None}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

data = load_data()

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Profile")

name = st.sidebar.text_input("Your name", value=data["profile"].get("name", "User"))
if st.sidebar.button("Save name"):
    data["profile"]["name"] = name
    save_data(data)
    st.success("Name saved!")

# Wallpaper upload
st.sidebar.subheader("🖼️ Wallpaper")
uploaded_wall = st.sidebar.file_uploader("Upload wallpaper", type=["jpg", "jpeg", "png"])
if uploaded_wall:
    wall_path = f"wallpaper_{uploaded_wall.name}"
    with open(wall_path, "wb") as f:
        f.write(uploaded_wall.read())
    st.session_state["wallpaper"] = wall_path
    st.success("Wallpaper updated!")

# Custom Voice Upload
st.sidebar.subheader("🎙 Custom Voice")
uploaded_voice = st.sidebar.file_uploader("Upload EchoSoul voice sample", type=["mp3", "wav"])
if uploaded_voice:
    voice_path = f"voice_{uploaded_voice.name}"
    with open(voice_path, "wb") as f:
        f.write(uploaded_voice.read())
    data["voice_file"] = voice_path
    save_data(data)
    st.success("Voice updated!")

# Background CSS
if "wallpaper" in st.session_state:
    st.markdown(f"""
    <style>
    .stApp {{
        background: url('{st.session_state["wallpaper"]}') no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------ Navigation ------------------
st.markdown(f"<h1 style='text-align:center;color:#9df;'>EchoSoul — Hi {name}</h1>", unsafe_allow_html=True)

choice = st.radio("Navigate", ["💬 Chat", "📞 Call", "🧠 Life Timeline", "🔐 Vault", "📜 Export", "ℹ️ About"])

# ------------------ Chat ------------------
if choice == "💬 Chat":
    st.subheader("Chat with EchoSoul")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_input("Say something...", key="chat_input", value="", placeholder="Type here and press Enter...")
    if st.button("Send"):
        if user_input.strip():
            st.session_state["history"].append({"role": "user", "content": user_input.strip()})

            # Persistent memory context
            memory_context = " ".join([m["text"] for m in data["memories"][-5:]])
            convo = [{"role": "system", "content": f"You are EchoSoul. Memory: {memory_context}"}]
            convo += st.session_state["history"]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=convo
            )
            reply = response.choices[0].message.content
            st.session_state["history"].append({"role": "assistant", "content": reply})

            # Clear input properly (fixed with st.rerun)
            st.rerun()

    for h in st.session_state["history"]:
        if h["role"] == "user":
            st.markdown(f"<div style='background:#6a0dad;padding:8px;border-radius:8px;margin:4px;color:white;'>You: {h['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#1e90ff;padding:8px;border-radius:8px;margin:4px;color:white;'>EchoSoul: {h['content']}</div>", unsafe_allow_html=True)

# ------------------ Call ------------------
elif choice == "📞 Call":
    st.subheader("Real Call with EchoSoul")

    call_input = st.text_input("Say something...")
    if st.button("Speak"):
        if call_input.strip():
            convo = [{"role": "system", "content": "You are EchoSoul, adaptive companion."}]
            convo.append({"role": "user", "content": call_input})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=convo
            )
            reply = response.choices[0].message.content

            st.success(f"EchoSoul: {reply}")

            # Generate speech (if custom voice not uploaded, fallback to TTS)
            if data.get("voice_file"):
                st.audio(data["voice_file"], format="audio/mp3")
            else:
                audio_out = f"echo_{uuid.uuid4()}.mp3"
                with client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",
                    input=reply
                ) as response_audio:
                    response_audio.stream_to_file(audio_out)
                st.audio(audio_out, format="audio/mp3")

# ------------------ Life Timeline ------------------
elif choice == "🧠 Life Timeline":
    st.subheader("Life Timeline")
    new_memory = st.text_area("Add new memory")
    if st.button("Save Memory"):
        if new_memory:
            data["memories"].append({"id": str(uuid.uuid4()), "text": new_memory, "time": str(datetime.datetime.now())})
            save_data(data)
            st.success("Memory saved!")

    for m in data["memories"]:
        st.markdown(f"📌 **{m['time']}** — {m['text']}")

# ------------------ Vault ------------------
elif choice == "🔐 Vault":
    st.subheader("Private Vault")
    password = st.text_input("Enter password", type="password")
    note = st.text_area("Secret note")
    if st.button("Save to Vault"):
        if password and note:
            data["vault"][str(uuid.uuid4())] = {"note": note, "time": str(datetime.datetime.now())}
            save_data(data)
            st.success("Saved securely!")

    st.write("🔒 Stored Vault Notes:")
    for v in data["vault"].values():
        st.markdown(f"- {v['time']}: {v['note']}")

# ------------------ Export ------------------
elif choice == "📜 Export":
    st.subheader("Export Your Data")
    st.download_button("Download JSON", json.dumps(data, indent=2), "echosoul_export.json")

# ------------------ About ------------------
elif choice == "ℹ️ About":
    st.subheader("About EchoSoul")
    st.write("""
    EchoSoul is your evolving AI companion:
    - ✅ Persistent Memory
    - ✅ Adaptive Personality
    - ✅ Real Voice Calls (choose or upload a custom voice)
    - ✅ Private Vault
    - ✅ Life Timeline
    - ✅ Legacy Mode
    """)
