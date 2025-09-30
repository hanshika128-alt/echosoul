# app.py — EchoSoul with real NLP
import streamlit as st
import os, json, datetime, base64, html, re
from openai import OpenAI

st.set_page_config(page_title="EchoSoul", layout="wide")

DATA_FILE = "echosoul_data.json"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------- Helpers ----------
def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def ensure_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump({
                "profile": {"name": "User"},
                "conversations": [],
                "memories": [],
                "vault": [],
                "wallpapers": []
            }, f)

def load_data():
    ensure_file()
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(d):
    with open(DATA_FILE, "w") as f:
        json.dump(d, f, indent=2)

def escape_html(s):
    return html.escape(s).replace("\n", "<br>")

def analyze_sentiment(text):
    if re.search(r"\b(good|great|happy|awesome|love)\b", text, re.I):
        return "energetic"
    elif re.search(r"\b(sad|bad|angry|tired|upset)\b", text, re.I):
        return "empathetic"
    return "friendly"

def gpt_reply(history, memories, user_msg, name):
    persona = analyze_sentiment(user_msg)
    system_prompt = f"""
You are EchoSoul, an AI companion for {name}.
You adapt your personality: energetic if user is positive, empathetic if negative, friendly otherwise.
You remember important details from past conversations and memories.
Tone: supportive, engaging, personal. Avoid robotic language.
"""
    context = ""
    for mem in memories[-5:]:
        context += f"- {mem['text']}\n"
    msgs = [{"role": "system", "content": system_prompt}]
    if context:
        msgs.append({"role": "system", "content": f"Here are recent memories:\n{context}"})
    for c in history[-10:]:
        msgs.append({"role": "user", "content": c["user"]})
        msgs.append({"role": "assistant", "content": c["bot"]})
    msgs.append({"role": "user", "content": user_msg})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        max_tokens=200
    )
    return response.choices[0].message.content

# ---------- Load ----------
data = load_data()

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {background: #0e0e1a; font-family: 'Inter', sans-serif; color: #f0f0f0;}
.app-header {font-size: 30px; font-weight: bold; text-align: center; margin-bottom: 20px; color: #fff; text-shadow: 0 0 6px #00e5ff;}
.chat-bubble {padding: 12px 16px; border-radius: 12px; margin: 6px 0; max-width: 75%;}
.user {background: #4e2a84; color: #fff; margin-left: auto;}
.bot {background: #1e3a5f; color: #e0f7ff; margin-right: auto;}
.sidebar .stSidebarContent {background: rgba(25,25,40,0.8); backdrop-filter: blur(10px);}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Profile")
    name = st.text_input("Your name", data["profile"].get("name", "User"))
    if st.button("Save name"):
        data["profile"]["name"] = name
        save_data(data)
        st.rerun()

# ---------- Header ----------
st.markdown(f"<div class='app-header'>EchoSoul — Hi {data['profile']['name']}</div>", unsafe_allow_html=True)

# ---------- Chat ----------
st.subheader("Chat with EchoSoul")

for c in data["conversations"][-20:]:
    st.markdown(f"<div class='chat-bubble user'>You: {escape_html(c['user'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble bot'>EchoSoul: {escape_html(c['bot'])}</div>", unsafe_allow_html=True)

msg = st.text_input("Say something...")
if st.button("Send") and msg.strip():
    reply = gpt_reply(data["conversations"], data["memories"], msg, data["profile"]["name"])
    data["conversations"].append({"user": msg, "bot": reply, "ts": now_iso()})
    save_data(data)
    st.rerun()
