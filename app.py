import streamlit as st
import os, json, datetime
from openai import OpenAI

# Init OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
DATA_FILE = "echosoul_data.json"

def ts_now():
    return datetime.datetime.now().isoformat(timespec="seconds")

# Load/Save
def load_data():
    if os.path.exists(DATA_FILE):
        return json.load(open(DATA_FILE, "r"))
    return {
        "profile": {},
        "timeline": [],
        "conversations": []
    }

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

data = load_data()

# If no profile → ask for introduction
if not data["profile"]:
    st.title("✨ Welcome to EchoSoul")
    with st.form("intro_form"):
        name = st.text_input("Your Name")
        age = st.number_input("Your Age", min_value=5, max_value=100, step=1)
        hobbies = st.text_area("Your Hobbies (comma separated)")
        submitted = st.form_submit_button("Save & Continue")
    if submitted and name:
        data["profile"] = {
            "name": name,
            "age": int(age),
            "hobbies": [h.strip() for h in hobbies.split(",") if h.strip()],
            "persona": {"tone": "friendly"}
        }
        save_data(data)
        st.success("Profile saved! Reload the app to start chatting.")
        st.stop()

# Generate AI reply
def generate_reply(user_msg):
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-5:]]
    context = "\n".join(memories) if memories else "No memories yet."
    tone = data["profile"].get("persona", {}).get("tone", "friendly")

    system_prompt = f"""
    You are EchoSoul, {data['profile']['name']}'s evolving AI companion.
    Tone: {tone}
    Known memories:\n{context}
    Always reply warmly and naturally.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )
    reply = response.choices[0].message.content
    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# --- UI Layout ---
st.set_page_config(page_title="EchoSoul", layout="wide")

with st.sidebar:
    st.header("⚙️ Voice Settings")
    st.radio("Choose AI voice", ["alloy", "verse", "amber"])
    st.file_uploader("Upload short voice sample (mp3/wav)", type=["mp3", "wav"])
    st.header("🖼️ Background Wallpaper")
    st.file_uploader("Upload wallpaper", type=["jpg", "png"])
    st.header("🔒 Settings")
    st.toggle("Enable adaptive learning", value=True)
    st.caption("Privacy & Ethics: Local storage • Inclusive design • Bias mitigation")
    st.markdown("---")
    st.subheader("📂 Navigate")
    st.radio("Go to", ["🏠 Home", "💬 Chat", "📞 Voice Call", "🧠 Life Timeline", "🔐 Vault", "📤 Export", "ℹ️ About"])

st.title("💬 EchoSoul Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show conversation with bubbles
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='text-align:right; background:#DCF8C6; padding:8px; border-radius:10px; margin:5px; display:inline-block;'>"
            f"**You:** {msg['content']}</div>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align:left; background:#F1F0F0; padding:8px; border-radius:10px; margin:5px; display:inline-block;'>"
            f"**EchoSoul:** {msg['content']}</div>", unsafe_allow_html=True
        )

# Input box → clears after send
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Say something to EchoSoul...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    reply = generate_reply(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.rerun()
