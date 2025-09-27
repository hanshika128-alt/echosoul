import streamlit as st
import os, json, re, datetime
from openai import OpenAI

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")

# ---- Neon Theme CSS ----
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #0D0D2E;
    color: #E0E0E0;
    font-family: 'Poppins', sans-serif;
}

/* Chat bubbles */
div[data-testid="stMarkdown"] p {
    background: rgba(20, 20, 60, 0.6);
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    box-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
}

/* Buttons with neon glow */
button[kind="primary"] {
    background: linear-gradient(90deg, #00ffff, #0088ff);
    color: black !important;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 0 12px #00ffff;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 18px #00ffff;
}

/* Sidebar */
.css-1d391kg, .stSidebar {
    background-color: #111122 !important;
    color: #E0E0E0 !important;
}

/* Headers neon */
h1, h2, h3 {
    color: #00ffff !important;
    text-shadow: 0 0 10px #00ffff;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Init OpenAI
# ------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DATA_FILE = "echosoul_data.json"

# ------------------------------
# Data helpers
# ------------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"profile": {}, "chat_history": [], "memories": []}
    return {"profile": {}, "chat_history": [], "memories": []}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

data = load_data()

# ------------------------------
# Profile intro
# ------------------------------
if "profile_done" not in st.session_state:
    st.session_state.profile_done = False
if "intro_step" not in st.session_state:
    st.session_state.intro_step = 0

intro_questions = [
    ("name", "What's your name?"),
    ("age", "What's your age?"),
    ("hobbies", "What are your hobbies?"),
    ("free_time", "What do you like to do in your free time?")
]

if not st.session_state.profile_done:
    st.title("✨ Let's get to know each other")
    key, q = intro_questions[st.session_state.intro_step]
    ans = st.text_input(q, key=f"q{st.session_state.intro_step}")
    if st.button("Next"):
        if ans.strip():
            data["profile"][key] = ans.strip()
            save_data(data)
            st.session_state.intro_step += 1
            if st.session_state.intro_step >= len(intro_questions):
                st.session_state.profile_done = True
        st.rerun()

# ------------------------------
# Main App
# ------------------------------
else:
    profile = data.get("profile", {})
    st.sidebar.title("Profile")
    name = st.sidebar.text_input("Display name", value=profile.get("name", "User"))
    if st.sidebar.button("Save name"):
        data["profile"]["name"] = name
        save_data(data)
        st.success("Profile updated!")

    st.sidebar.title("Voice & Chat")
    st.sidebar.file_uploader("Upload sample voice (optional)", type=["mp3","wav","m4a"])

    st.sidebar.title("Vault (prototype)")
    vault_pass = st.sidebar.text_input("Vault password", type="password")

    st.title(f"✨ EchoSoul — Hi {profile.get('name','User')}")

    st.write("This chat feels like an ongoing dialogue. After you send a message, the input clears automatically.")

    # Chat history
    for entry in data["chat_history"]:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**EchoSoul:** {entry['ai']}")

    # Chat input
    user_msg = st.text_input("Say something to EchoSoul...", key="chat_input", value="")
    if st.button("Send"):
        if user_msg.strip():
            # Build context
            context = f"User profile: {profile}. Past memories: {data['memories']}. Chat so far: {data['chat_history'][-5:]}"
            prompt = f"{context}\n\nUser: {user_msg}\nEchoSoul:"
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You are EchoSoul, a personal AI companion. Stay on topic, keep dialogue natural, adapt tone (friendly, empathetic, energetic)."},
                        {"role":"user","content":prompt}
                    ]
                )
                ai_reply = response.choices[0].message.content
            except Exception as e:
                ai_reply = f"(Error: {e})"

            # Save history
            data["chat_history"].append({"user": user_msg, "ai": ai_reply})
            save_data(data)

            # Clear input + refresh
            st.session_state.chat_input = ""
            st.rerun()
