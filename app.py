import streamlit as st
import os, json, datetime
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DATA_FILE = "echosoul_data.json"

def ts_now():
    return datetime.datetime.now().isoformat(timespec="seconds")

# Load / Save
def load_data():
    if os.path.exists(DATA_FILE):
        return json.load(open(DATA_FILE, "r"))
    return {
        "profile": {"name": "Hanshika", "age": 15, "hobbies": [], "persona": {"tone": "friendly"}},
        "timeline": [],
        "conversations": []
    }

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

data = load_data()

# AI reply
def generate_reply(data, user_msg):
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-5:]]
    context = "\n".join(memories) if memories else "No memories yet."
    tone = data["profile"]["persona"].get("tone", "friendly")

    system_prompt = f"""
    You are EchoSoul, Hanshika's evolving AI companion.
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

# --- UI ---
st.set_page_config(page_title="EchoSoul", layout="wide")

st.title("✨ EchoSoul — Your AI Companion")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat display with bubbles
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='text-align: right; background-color: #DCF8C6; padding: 8px; border-radius: 10px; margin:5px; display:inline-block;'>"
            f"**You:** {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align: left; background-color: #F1F0F0; padding: 8px; border-radius: 10px; margin:5px; display:inline-block;'>"
            f"**EchoSoul:** {msg['content']}</div>",
            unsafe_allow_html=True
        )

# Input at bottom
with st.form("chat_input", clear_on_submit=True):  # 👈 clears automatically
    user_input = st.text_input("Type here...", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    reply = generate_reply(data, user_input)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.rerun()
