import streamlit as st
import os
from openai import OpenAI
from pathlib import Path
import tempfile

# --- Init OpenAI Client ---
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))

st.set_page_config(page_title="EchoSoul", layout="wide")

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "wallpaper" not in st.session_state:
    st.session_state["wallpaper"] = None
if "voice_file" not in st.session_state:
    st.session_state["voice_file"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = "User"
if "chat_input" not in st.session_state:
    st.session_state["chat_input"] = ""

# --- Helper Functions ---
def ask_model(messages, model="gpt-4o-mini"):
    """Send conversation to GPT and return reply"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.9,
            max_tokens=600
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Error contacting OpenAI API: {e}]"

def transcribe_audio(file_path):
    """Transcribe speech to text"""
    try:
        with open(file_path, "rb") as af:
            transcript_resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=af
            )
        return transcript_resp.text
    except Exception as e:
        return f"[Transcription error: {e}]"

def tts_reply(text, output_path="reply.mp3"):
    """Generate speech audio from text"""
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        ) as response:
            response.stream_to_file(output_path)
        return output_path
    except Exception:
        return None

# --- Sidebar ---
st.sidebar.header("Profile")
user_name = st.sidebar.text_input("Your name", st.session_state["user_name"])
if st.sidebar.button("Save name"):
    st.session_state["user_name"] = user_name

st.sidebar.header("Wallpaper")
wallpaper_file = st.sidebar.file_uploader("Upload wallpaper (jpg/png)", type=["jpg", "jpeg", "png"])
if wallpaper_file:
    tmp_wall = Path(tempfile.gettempdir()) / wallpaper_file.name
    tmp_wall.write_bytes(wallpaper_file.getvalue())
    st.session_state["wallpaper"] = str(tmp_wall)

if st.sidebar.button("Apply wallpaper") and st.session_state["wallpaper"]:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("file://{st.session_state['wallpaper']}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.sidebar.header("Custom voice (sample)")
voice_file = st.sidebar.file_uploader("Upload EchoSoul voice sample", type=["mp3", "wav"])
if voice_file:
    tmp_v = Path(tempfile.gettempdir()) / voice_file.name
    tmp_v.write_bytes(voice_file.getvalue())
    st.session_state["voice_file"] = str(tmp_v)

# --- Title ---
st.markdown(
    f"<h1 style='text-align:center;color:#7FDBFF;'>EchoSoul — Hi {st.session_state['user_name']}</h1>",
    unsafe_allow_html=True
)

# --- Navigation ---
page = st.radio("Navigate", ["💬 Chat", "📞 Call", "🧠 Life Timeline", "🔒 Vault", "📜 Export", "ℹ️ About"], horizontal=True)

# --- Chat Page ---
if page == "💬 Chat":
    st.subheader("Chat with EchoSoul")

    # Display previous messages
    for msg in st.session_state["messages"]:
        role = msg["role"]
        if role == "user":
            st.markdown(f"<div style='background:#8e44ad;color:white;padding:8px;border-radius:10px;margin:5px;'>You: {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#3498db;color:white;padding:8px;border-radius:10px;margin:5px;'>EchoSoul: {msg['content']}</div>", unsafe_allow_html=True)

    # Input box
    chat_val = st.text_input("Say something...", value=st.session_state["chat_input"], key="chat_box", placeholder="Type here and press Enter...")

    if st.button("Send", key="send_btn") and chat_val.strip():
        st.session_state["messages"].append({"role": "user", "content": chat_val})
        reply = ask_model(st.session_state["messages"])
        st.session_state["messages"].append({"role": "assistant", "content": reply})

        # clear the input correctly
        st.session_state["chat_input"] = ""
        st.session_state["chat_box"] = ""
        st.rerun()

# --- Call Page ---
elif page == "📞 Call":
    st.subheader("Real Call with EchoSoul")

    audio_input = st.file_uploader("Upload your voice (mp3/wav)", type=["mp3", "wav"])
    if audio_input:
        tmp_in = Path(tempfile.gettempdir()) / audio_input.name
        tmp_in.write_bytes(audio_input.getvalue())

        user_text = transcribe_audio(str(tmp_in))
        st.write(f"🗣 You said: {user_text}")

        reply = ask_model([{"role": "user", "content": user_text}])
        st.write(f"EchoSoul says: {reply}")

        out_file = tts_reply(reply)
        if out_file:
            audio_bytes = open(out_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")

# --- Other Pages ---
elif page == "🧠 Life Timeline":
    st.info("Life Timeline: Coming soon.")
elif page == "🔒 Vault":
    st.warning("Vault feature coming soon.")
elif page == "📜 Export":
    st.success("Export feature coming soon.")
elif page == "ℹ️ About":
    st.markdown("EchoSoul is your evolving digital companion.")
