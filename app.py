# app.py (UI improved)
import streamlit as st
import json, os, base64, datetime, hashlib, re, html

st.set_page_config(page_title="EchoSoul — Neon", layout="wide")

DATA_FILE = "echosoul_data.json"

def now_iso():
    return datetime.datetime.utcnow().isoformat()+"Z"

def ensure_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE,"w") as f:
            json.dump({"profile":{"name":"User"},"conversations":[]},f)

def load(): ensure_file(); return json.load(open(DATA_FILE,"r"))
def save(d): json.dump(d,open(DATA_FILE,"w"),indent=2)

def short_id(s): return hashlib.sha1(s.encode()).hexdigest()[:8]
def esc(s): return html.escape(s).replace("\n","<br>")

# Load
ensure_file()
data=load()

# Neon CSS style
st.markdown(f"""
<style>
.stApp {{
  background: linear-gradient(135deg, #0d0d2e, #1a0033);
  color: #e8faff;
  font-family: 'Inter', sans-serif;
}}
.app-header {{
  font-size: 34px;
  font-weight: bold;
  text-align: center;
  margin: 10px 0 30px 0;
  color: #fff;
  text-shadow: 0 0 8px #4ee0ff, 0 0 16px #bb1b67;
}}
.chat-bubble {{
  padding: 12px 16px;
  border-radius: 14px;
  margin: 10px 0;
  max-width: 75%;
  font-size: 16px;
  line-height: 1.5em;
}}
.user-bubble {{
  background: linear-gradient(135deg, #ff4ecd, #9c27b0);
  color: white;
  box-shadow: 0 0 10px rgba(255,78,205,0.7);
  align-self: flex-end;
}}
.bot-bubble {{
  background: linear-gradient(135deg, #00e5ff, #0066ff);
  color: white;
  box-shadow: 0 0 10px rgba(0,229,255,0.7);
  align-self: flex-start;
}}
.sidebar .stSidebarContent {{
  background: rgba(20,20,40,0.6);
  backdrop-filter: blur(10px);
}}
div.stButton>button {{
  background: linear-gradient(90deg, #4ee0ff, #bb1b67);
  border: none;
  color: white;
  padding: 0.6em 1.2em;
  border-radius: 10px;
  font-weight: bold;
  box-shadow: 0 0 10px #4ee0ff;
}}
div.stButton>button:hover {{
  box-shadow: 0 0 20px #bb1b67;
}}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    name = st.text_input("Your name", data["profile"]["name"])
    if st.button("Save name"):
        data["profile"]["name"]=name; save(data); st.experimental_rerun()

# Header
st.markdown(f"<div class='app-header'>✨ EchoSoul — Hi {data['profile']['name']} ✨</div>", unsafe_allow_html=True)

# Chat page (simplified for demo)
st.subheader("💬 Chat")
for c in data["conversations"][-30:]:
    st.markdown(f"<div class='chat-bubble user-bubble'>You: {esc(c['user'])}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble bot-bubble'>EchoSoul: {esc(c['bot'])}</div>",unsafe_allow_html=True)

msg = st.text_input("Say something to EchoSoul...")
if st.button("Send") and msg.strip():
    bot = f"EchoSoul (demo): You said '{msg}'"
    data["conversations"].append({"user":msg,"bot":bot,"ts":now_iso()})
    save(data); st.experimental_rerun()
