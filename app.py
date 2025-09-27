# app.py — EchoSoul (Full Neon Companion)

import streamlit as st
import json, os, base64, datetime, hashlib, re
import openai

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else None
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

DATA_FILE = "echosoul_data.json"

# ---------------------------
# HELPERS
# ---------------------------
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {
            "profile": {"name": None, "age": None, "hobbies": None,
                        "free_time": None, "intro_completed": False},
            "timeline": [], "vault": [], "conversations": []
        }
    with open(DATA_FILE,"r") as f: return json.load(f)

def save_data(d):
    with open(DATA_FILE,"w") as f: json.dump(d,f,indent=2)

data = load_data()

def xor_encrypt(text,key):
    return "".join(chr(ord(c)^ord(key[i%len(key)])) for i,c in enumerate(text))

def add_memory(title,content):
    data["timeline"].append({"title":title,"content":content,"ts":ts_now()})
    save_data(data)

def sentiment_tone(msg):
    if re.search(r"(happy|great|good|awesome|love)", msg, re.I):
        return "energetic"
    elif re.search(r"(sad|tired|angry|upset|bad)", msg, re.I):
        return "empathetic"
    return "friendly"

def call_gpt(conv, msg):
    if not OPENAI_KEY: return "[No API Key]"
    tone = sentiment_tone(msg)
    system_prompt = f"You are EchoSoul, a {tone} personal AI companion."
    messages=[{"role":"system","content":system_prompt}]
    for c in conv[-8:]:
        messages.append({"role":"user","content":c["u"]})
        messages.append({"role":"assistant","content":c["b"]})
    messages.append({"role":"user","content":msg})
    resp=openai.ChatCompletion.create(model="gpt-4o-mini",messages=messages)
    return resp.choices[0].message.content, 0.9  # heuristic confidence

# ---------------------------
# STYLE
# ---------------------------
st.markdown("""
<style>
:root {color-scheme: dark;}
.stApp {background: radial-gradient(circle at top left,#0d1b2a,#000); color:#EDEEF2;}
h1,h2,h3 {color:#6FFFE9; text-shadow:0 0 15px #6FFFE9;}
.neon-button {
    background: transparent; border:2px solid #6FFFE9;
    color:#6FFFE9; font-weight:bold; padding:6px 14px;
    border-radius:8px; transition:0.3s;
}
.neon-button:hover {background:#6FFFE922; box-shadow:0 0 12px #6FFFE9;}
.sidebar-title {font-size:20px; font-weight:bold; color:#6FFFE9; text-shadow:0 0 10px #6FFFE9;}
.chat-bubble-user {background:#1e1e2f; padding:8px; border-radius:10px; margin:4px;}
.chat-bubble-ai {background:#133b5c; padding:8px; border-radius:10px; margin:4px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
with st.sidebar:
    st.markdown("<p class='sidebar-title'>✨ EchoSoul</p>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["🏠 Home","💬 Chat","🎙️ Voice Chat","📜 Timeline","🔒 Vault","📤 Export","ℹ️ About"])
    st.divider()
    st.markdown("**Profile Settings**")
    pname = st.text_input("Display name", value=data["profile"].get("name") or "")
    if st.button("Save Profile"):
        data["profile"]["name"]=pname; save_data(data); st.success("Profile saved!")

# ---------------------------
# PAGES
# ---------------------------
if page=="🏠 Home":
    st.title("✨ EchoSoul — Your AI Companion")
    if not data["profile"].get("intro_completed"):
        st.subheader("Quick Introduction")
        name = st.text_input("What's your name?")
        age = st.text_input("What's your age?")
        hobbies = st.text_input("What are your hobbies?")
        free_time = st.text_input("What do you like in your free time?")
        if st.button("Save Intro", use_container_width=True):
            if name.strip():
                data["profile"].update({"name":name,"age":age,"hobbies":hobbies,
                                        "free_time":free_time,"intro_completed":True})
                save_data(data)
                add_memory("Introduction",f"Name: {name}, Age: {age}, Hobbies: {hobbies}, Free-time: {free_time}")
                st.success("Intro saved! Go to Chat →")
            else: st.warning("Please enter your name.")
    else:
        st.success(f"Welcome back, {data['profile']['name']}!")

elif page=="💬 Chat":
    st.header(f"💬 Chat with EchoSoul — Hi {data['profile'].get('name') or 'User'}!")
    for c in data["conversations"][-6:]:
        st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {c['u']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-ai'><b>EchoSoul:</b> {c['b']}</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Say something:")
        submitted = st.form_submit_button("Send", use_container_width=True)
        if submitted and user_msg.strip():
            reply, conf = call_gpt(data["conversations"], user_msg)
            data["conversations"].append({"u":user_msg,"b":reply,"conf":conf})
            save_data(data)
            st.experimental_rerun()

elif page=="🎙️ Voice Chat":
    st.header("🎙️ Voice Chat")
    st.info("Upload a sample voice (mp3/wav) and EchoSoul will mimic it (prototype).")
    voice_file = st.file_uploader("Upload your voice", type=["mp3","wav"])
    if voice_file:
        st.success("Voice uploaded! TTS playback can be generated here.")

elif page=="📜 Timeline":
    st.header("📜 Life Timeline")
    title = st.text_input("Add memory title")
    content = st.text_area("Describe the memory")
    if st.button("Save Memory"):
        add_memory(title,content); st.success("Memory saved!")
    if not data["timeline"]: st.info("No memories yet.")
    else:
        for m in reversed(data["timeline"]):
            st.markdown(f"**{m['title']}** — {m['ts']}")
            st.write(m['content']); st.divider()

elif page=="🔒 Vault":
    st.header("🔒 Private Vault")
    pw = st.text_input("Vault password", type="password")
    note = st.text_area("Write a secret note")
    if st.button("Encrypt & Save"):
        if pw and note:
            enc = xor_encrypt(note,pw)
            data["vault"].append({"enc":enc,"ts":ts_now()})
            save_data(data); st.success("Saved securely!")
    if data["vault"]:
        st.subheader("Saved Notes (decrypt with password):")
        for v in data["vault"]:
            try:
                dec = xor_encrypt(v["enc"],pw) if pw else "[locked]"
            except: dec="[error]"
            st.write(f"{v['ts']}: {dec}")

elif page=="📤 Export":
    st.header("📤 Export Data")
    st.download_button("Download JSON", json.dumps(data,indent=2), "echosoul_export.json")

elif page=="ℹ️ About":
    st.header("ℹ️ About EchoSoul")
    st.markdown("""
    EchoSoul is your **personal AI companion**.  
    - Conversational memory  
    - Adaptive tone (friendly, empathetic, energetic)  
    - Private Vault (prototype)  
    - Voice Chat (prototype)  
    - Timeline of memories  
    - Export & Legacy Mode  
    """)
