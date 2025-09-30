# app.py — EchoSoul Full
import streamlit as st
import os, json, datetime, base64, html, re
from openai import OpenAI
from cryptography.fernet import Fernet

# -------- CONFIG --------
st.set_page_config(page_title="EchoSoul", layout="wide")
DATA_FILE = "echosoul_data.json"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------- DATA MGMT --------
def ensure_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump({
                "profile": {"name": "User"},
                "conversations": [],
                "memories": [],
                "vault": [],
                "timeline": [],
                "wallpaper": None,
                "key": Fernet.generate_key().decode()
            }, f)

def load_data():
    ensure_file()
    return json.load(open(DATA_FILE))

def save_data(d):
    json.dump(d, open(DATA_FILE, "w"), indent=2)

data = load_data()
fernet = Fernet(data["key"].encode())

# -------- HELPERS --------
def now(): return datetime.datetime.utcnow().isoformat()

def sentiment(text):
    if re.search(r"(good|great|happy|awesome|love)", text, re.I): return "energetic"
    if re.search(r"(sad|bad|angry|tired|upset)", text, re.I): return "empathetic"
    return "friendly"

def gpt_reply(user_msg, history, memories, persona, name):
    sys = f"""
You are EchoSoul, personal AI companion for {name}.
You have memory, adapt tone, and grow with the user.
Persona now: {persona}.
Recent memories: {', '.join(m['text'] for m in memories[-5:])}.
"""
    msgs = [{"role":"system","content":sys}]
    for c in history[-8:]:
        msgs.append({"role":"user","content":c["user"]})
        msgs.append({"role":"assistant","content":c["bot"]})
    msgs.append({"role":"user","content":user_msg})
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=250)
    return r.choices[0].message.content

def escape(s): return html.escape(s).replace("\n","<br>")

# -------- CSS --------
st.markdown(f"""
<style>
.stApp {{background: #0d0d2e url('{data.get("wallpaper","")}') no-repeat center/cover; color:#f0f0f0;}}
.app-header {{font-size:26px; font-weight:700; text-align:center; color:#fff; text-shadow:0 0 6px #00e5ff;}}
.chat-bubble {{padding:10px 14px;border-radius:12px;margin:6px 0;max-width:75%;}}
.user {{background:#4e2a84;color:#fff;margin-left:auto;}}
.bot {{background:#1e3a5f;color:#e0f7ff;margin-right:auto;}}
.sidebar .stSidebarContent {{background:rgba(25,25,40,0.85);backdrop-filter:blur(10px);}}
</style>
""", unsafe_allow_html=True)

# -------- SIDEBAR --------
with st.sidebar:
    st.subheader("👤 Profile")
    name = st.text_input("Your name", data["profile"]["name"])
    if st.button("Save name"): data["profile"]["name"]=name; save_data(data); st.rerun()
    st.subheader("🖼 Wallpaper")
    file = st.file_uploader("Upload", type=["png","jpg","jpeg"])
    if file:
        path = f"wall_{file.name}"
        with open(path,"wb") as f: f.write(file.getbuffer())
        data["wallpaper"] = "data:image/png;base64,"+base64.b64encode(open(path,"rb").read()).decode()
        save_data(data); st.rerun()
    st.subheader("🔒 Vault")
    vault_pw = st.text_input("Password", type="password")
    vault_text = st.text_area("New secret")
    if st.button("Save to Vault") and vault_pw and vault_text:
        enc = fernet.encrypt(vault_text.encode()).decode()
        data["vault"].append({"ts":now(),"val":enc})
        save_data(data); st.success("Saved secret")
    if st.checkbox("Show Vault"):
        for v in data["vault"]:
            st.write(fernet.decrypt(v["val"].encode()).decode())

# -------- HEADER --------
st.markdown(f"<div class='app-header'>EchoSoul — Hi {data['profile']['name']}</div>", unsafe_allow_html=True)

# -------- NAV --------
tab = st.radio("Navigate", ["💬 Chat","📞 Call","🧠 Life Timeline","🔐 Vault","📜 Legacy"])

# -------- CHAT --------
if tab=="💬 Chat":
    st.subheader("Chat with EchoSoul")
    for c in data["conversations"][-15:]:
        st.markdown(f"<div class='chat-bubble user'>You: {escape(c['user'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble bot'>EchoSoul: {escape(c['bot'])}</div>", unsafe_allow_html=True)
    msg = st.text_input("Say something...")
    if st.button("Send") and msg.strip():
        reply = gpt_reply(msg, data["conversations"], data["memories"], sentiment(msg), name)
        data["conversations"].append({"user":msg,"bot":reply,"ts":now()})
        # store memory occasionally
        if len(msg.split())>3: data["memories"].append({"text":msg,"ts":now()})
        save_data(data); st.rerun()

# -------- CALL --------
elif tab=="📞 Call":
    st.subheader("Call EchoSoul")
    st.write("🎤 Voice chat mode (prototype). Speak & EchoSoul will reply with TTS.")
    st.info("For demo: type a message and hear AI voice.")
    text = st.text_input("Say...")
    if st.button("Speak"):
        reply = gpt_reply(text, data["conversations"], data["memories"], sentiment(text), name)
        st.write("EchoSoul:", reply)
        # TTS
        speech = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=reply)
        st.audio(speech.stream_to_file("voice.mp3"))

# -------- TIMELINE --------
elif tab=="🧠 Life Timeline":
    st.subheader("Life Timeline")
    ev = st.text_input("Add life event")
    if st.button("Save Event"): data["timeline"].append({"text":ev,"ts":now()}); save_data(data); st.rerun()
    for e in data["timeline"]:
        st.write(f"{e['ts']}: {e['text']}")

# -------- LEGACY --------
elif tab=="📜 Legacy":
    st.subheader("Legacy Snapshot")
    st.json({"profile":data["profile"],"timeline":data["timeline"],"memories":data["memories"][-10:]})
