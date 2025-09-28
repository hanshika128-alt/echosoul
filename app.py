import streamlit as st
import openai
import json
import os
import datetime
import base64
import random

# ---------------------------
# Setup
# ---------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")

# Load API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

DATA_FILE = "echosoul_data.json"

# ---------------------------
# Persistent Memory
# ---------------------------
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump({"memories": [], "vault": {}, "conversations": []}, f)

def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

data = load_data()

# ---------------------------
# XOR-based simple vault encryption
# ---------------------------
def xor_encrypt_decrypt(text, key):
    return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(text))

# ---------------------------
# Adaptive Personality (simple sentiment analysis)
# ---------------------------
def detect_sentiment(user_input):
    if any(word in user_input.lower() for word in ["good", "great", "awesome", "happy", "love"]):
        return "energetic"
    elif any(word in user_input.lower() for word in ["bad", "sad", "upset", "angry", "tired"]):
        return "empathetic"
    return "friendly"

# ---------------------------
# Conversational AI
# ---------------------------
def get_ai_reply(user_input, persona="friendly"):
    # Prepare context
    history = data["conversations"][-5:]  # last 5 turns
    messages = [{"role": "system", "content": f"You are EchoSoul, a {persona} AI companion."}]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["ai"]})
    messages.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ AI Error: {e}"

    # Save to memory
    data["conversations"].append({"user": user_input, "ai": reply})
    save_data(data)
    return reply

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://emojicdn.elk.sh/✨", width=40)
    st.title("EchoSoul")
    
    st.subheader("Voice Settings")
    voice = st.radio("Choose AI voice", ["alloy", "verse", "amber"], index=0)
    uploaded_voice = st.file_uploader("Upload sample voice to mimic", type=["mp3", "wav"])

    st.subheader("Settings")
    adaptive_learning = st.toggle("Enable adaptive learning", value=True)

    st.divider()
    st.subheader("Navigate")
    page = st.radio(" ", ["🏠 Home", "💬 Chat", "🎙️ Voice Chat", "🕒 Life Timeline", "🔐 Vault", "📜 Export", "ℹ️ About"])

# ---------------------------
# Top-right Call Button
# ---------------------------
st.markdown(
    """
    <style>
    .call-button {
        position: fixed;
        top: 10px;
        right: 20px;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("📞 Call", key="call", help="Start AI voice call", use_container_width=False):
    st.session_state["on_call"] = True
    st.toast("🔊 EchoSoul is now speaking with you...")

# ---------------------------
# Pages
# ---------------------------
if page == "🏠 Home":
    st.title("✨ EchoSoul — Your AI Companion")
    st.write("A personal AI with memory, voice, and transparency.")
    st.info("Be transparent: EchoSoul is **not perfect**, but it adapts, remembers, and grows with you.")

elif page == "💬 Chat":
    st.title("💬 EchoSoul Chat")

    # Chat history
    history_placeholder = st.empty()
    history_text = ""
    for h in data["conversations"]:
        history_text += f"**You:** {h['user']}\n\n**EchoSoul:** {h['ai']}\n\n"
    history_placeholder.markdown(history_text)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Say something", "")
        submitted = st.form_submit_button("Send")
        if submitted and user_input.strip():
            sentiment = detect_sentiment(user_input)
            ai_reply = get_ai_reply(user_input, persona=sentiment)
            st.experimental_rerun()

elif page == "🎙️ Voice Chat":
    st.title("🎙️ Voice Chat with EchoSoul")
    st.write("Talk to EchoSoul using your voice.")
    st.warning("Prototype — depends on uploaded voice + text-to-speech.")

elif page == "🕒 Life Timeline":
    st.title("🕒 Life Timeline")
    new_memory = st.text_input("Add a memory")
    if st.button("Save Memory"):
        data["memories"].append({"text": new_memory, "time": str(datetime.datetime.now())})
        save_data(data)
        st.success("Memory saved!")
    for mem in data["memories"]:
        st.write(f"- {mem['time']}: {mem['text']}")

elif page == "🔐 Vault":
    st.title("🔐 Vault (Prototype)")
    pw = st.text_input("Enter Vault Password", type="password")
    vault_action = st.radio("Action", ["View", "Add"])
    if pw:
        if vault_action == "Add":
            note = st.text_area("New Note")
            if st.button("Save Note"):
                encrypted = xor_encrypt_decrypt(note, pw)
                data["vault"][str(len(data['vault']))] = encrypted
                save_data(data)
                st.success("Note saved securely!")
        elif vault_action == "View":
            for k, v in data["vault"].items():
                try:
                    decrypted = xor_encrypt_decrypt(v, pw)
                    st.write(f"Note {k}: {decrypted}")
                except:
                    st.error("Wrong password!")

elif page == "📜 Export":
    st.title("📜 Export Data")
    json_data = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_data.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="echosoul_export.json">📥 Download Export</a>'
    st.markdown(href, unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.title("ℹ️ About EchoSoul")
    st.write("""
    - **Conversational AI** powered by GPT-4o-mini  
    - **Memory** stored locally (`echosoul_data.json`)  
    - **Vault** with prototype encryption  
    - **Adaptive personality**  
    - **Explainable AI elements**: tooltips, transparency, error messages  
    """)
