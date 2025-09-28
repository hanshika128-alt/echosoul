import streamlit as st
from openai import OpenAI
from gtts import gTTS
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# CSS Neon Styling
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #02010a;
        color: #e0e0ff;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.5em;
        text-align: center;
        color: #00f7ff;
        text-shadow: 0 0 10px #00f7ff, 0 0 20px #00f7ff, 0 0 30px #00f7ff;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #aaaaff;
        margin-bottom: 30px;
    }
    .chat-bubble-user {
        background: #2c2c54;
        color: #fff;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: right;
    }
    .chat-bubble-ai {
        background: #0f3460;
        color: #fff;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: left;
    }
    .small-muted {
        text-align: center;
        font-size: 0.8em;
        color: #777;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Header
# -------------------------------
st.markdown("<div class='title'>✨ EchoSoul</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your personal AI companion</div>", unsafe_allow_html=True)

# -------------------------------
# Session State
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input("Say something:", "", key="input_text")

if st.button("Send"):
    if user_input.strip():
        # Save user message
        st.session_state.history.append({"role": "user", "content": user_input})

        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.history
            )
            ai_reply = response.choices[0].message.content
        except Exception as e:
            ai_reply = f"⚠️ API Error: {e}"

        st.session_state.history.append({"role": "assistant", "content": ai_reply})

        # Text-to-Speech with gTTS
        try:
            tts = gTTS(text=ai_reply, lang="en")
            tts.save("reply.mp3")
            st.audio("reply.mp3")
        except Exception as e:
            st.warning(f"TTS error: {e}")

# -------------------------------
# Chat History Display
# -------------------------------
st.markdown("## 💬 Chat History")
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <div class="small-muted">
        EchoSoul ready — start a chat above.
    </div>
    """,
    unsafe_allow_html=True
)
