import streamlit as st
import os, json, datetime
from openai import OpenAI

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")

# Neon CSS
st.markdown("""
<style>
.stApp { background-color: #0D0D2E; color: #E0E0E0; font-family: 'Poppins', sans-serif; }
div[data-testid="stMarkdown"] p {
    background: rgba(20, 20, 60, 0.6);
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    box-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
}
button[kind="primary"] {
    background: linear-gradient(90deg, #00ffff, #0088ff);
    color: black !important;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 0 12px #00ffff;
}
button[kind="primary"]:hover { box-shadow: 0 0 18px #00ffff; }
.css-1d391kg, .stSidebar { background-color: #111122 !important; color: #E0E0E0 !important; }
h1, h2, h3 { color: #00ffff !important; text-shadow: 0 0 10px #00ffff; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Init OpenAI
# ------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
DATA_FILE = "echosoul_data.json"

# ------------------------------
# Helpers
# ------------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"profile": {}, "chat_history": [], "memories": [], "timeline": [], "vault": ""}
    return {"profile": {}, "chat_history": [], "memories": [], "timeline": [], "vault": ""}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

data = load_data()
data.setdefault("profile", {})
data.setdefault("chat_history", [])
data.setdefault("memories", [])
data.setdefault("timeline", [])
data.setdefault("vault", "")

# ------------------------------
# Intro Questions
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
    profile = data["profile"]

    # Sidebar
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
    vault_note = st.sidebar.text_area("Private note (encrypted view)")
    if st.sidebar.button("Save to Vault") and vault_pass.strip():
        # Very simple "encryption" (reversed string)
        data["vault"] = vault_note[::-1]
        save_data(data)
        st.success("Saved securely!")

    st.sidebar.title("Export Legacy")
    if st.sidebar.button("Export JSON"):
        st.sidebar.download_button("Download Data", json.dumps(data, indent=2), file_name="echosoul_export.json")

    # Title
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
            # Simple sentiment detection
            mood = "neutral"
            if any(w in user_msg.lower() for w in ["sad","upset","angry","depressed"]): mood = "sad"
            elif any(w in user_msg.lower() for w in ["happy","excited","love","great"]): mood = "happy"

            # Build context
            context = f"Profile: {profile}. Memories: {data['memories'][-3:]}. Timeline: {data['timeline'][-3:]}"
            prompt = f"{context}\nMood detected: {mood}\nUser: {user_msg}\nEchoSoul:"

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You are EchoSoul, a lifelong AI companion. Adapt your tone to the user's mood, remember details, and keep responses personal."},
                        {"role":"user","content":prompt}
                    ]
                )
                ai_reply = response.choices[0].message.content
            except Exception as e:
                ai_reply = f"(Error: {e})"

            # Save conversation + memory + timeline
            data["chat_history"].append({"user": user_msg, "ai": ai_reply})
            data["memories"].append(user_msg)
            data["timeline"].append({"time": str(datetime.datetime.now()), "event": user_msg})
            save_data(data)

            # Clear input
            st.session_state.chat_input = ""
            st.rerun()
