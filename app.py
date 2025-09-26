import streamlit as st
import os, json
from openai import OpenAI
import tempfile
import base64

# ======================
# INITIALIZATION
# ======================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
DATA_FILE = "echosoul_data.json"

def default_data():
    return {"user_name":"User","memories":[],"vault":{},"chat_history":[],"voice_mode":"alloy","mimic_voice":None}
def load_data():
    return json.load(open(DATA_FILE,"r",encoding="utf-8")) if os.path.exists(DATA_FILE) else default_data()
def save_data(data):
    with open(DATA_FILE,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)
data = load_data()

# ======================
# ENCRYPTION
# ======================
def xor_encrypt_decrypt(text,key):
    return "".join(chr(ord(c)^ord(key[i%len(key)])) for i,c in enumerate(text))

# ======================
# AI REPLY
# ======================
def generate_reply(user_input, history, memories, persona="friendly"):
    context = (
        f"You are EchoSoul, a digital companion that learns and adapts.\n"
        f"User's name: {data.get('user_name','User')}\n"
        f"Persona: {persona}\n"
        f"Memories: {', '.join(memories[-5:])}\n"
        f"History: {history[-10:]}\n"
        f"Be empathetic, continue naturally, avoid abrupt topic changes."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":context},{"role":"user","content":user_input}],
        temperature=0.8,
        max_tokens=250
    )
    return response.choices[0].message.content

# ======================
# SPEECH HANDLING
# ======================
def transcribe_audio(file):
    with open(file,"rb") as f:
        resp = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
        return resp.text

def synthesize_speech(text, voice="alloy"):
    speech = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".mp3")
    tmp.write(speech.read())
    tmp.flush()
    return tmp.name

def mimic_voice_from_upload(file):
    # Prototype: in real version we’d train embedding. Here we just save path.
    return file.getvalue()

# ======================
# CUSTOM THEME (NEON)
# ======================
st.markdown("""
<style>
body {background:#0D0D2E;color:#F0F0F0;font-family:'Poppins',sans-serif;}
.stApp {background:linear-gradient(135deg,#0D0D2E 60%,#121212);}
.stSidebar {background:rgba(20,20,40,0.7);backdrop-filter:blur(12px);}
.stButton>button {background:#111122;border:1px solid #00FFFF;color:#F0F0F0;border-radius:8px;transition:0.3s;}
.stButton>button:hover {background:#00FFFF;color:#0D0D2E;box-shadow:0 0 12px #00FFFF;}
.stTextInput>div>div>input,.stTextArea>div>textarea {background:rgba(30,30,50,0.6);border:1px solid #00FFFF;border-radius:8px;color:#F0F0F0;}
h1,h2,h3 {color:#00FFFF;text-shadow:0 0 8px rgba(0,255,255,0.7);}
#ai-status {position:fixed;bottom:20px;right:25px;width:22px;height:22px;border-radius:50%;background:radial-gradient(circle,#00ffcc,#009999);box-shadow:0 0 12px #00ffff,0 0 24px #00ffff,0 0 36px #0088ff;animation:pulse 2s infinite;z-index:9999;}
@keyframes pulse{0%{transform:scale(1);opacity:.9;}50%{transform:scale(1.3);opacity:.6;}100%{transform:scale(1);opacity:.9;}}
</style>
""",unsafe_allow_html=True)

# ======================
# STATUS INDICATOR
# ======================
if "ai_status" not in st.session_state: st.session_state.ai_status="ready"
def render_status():
    color_map={"ready":"radial-gradient(circle,#00ffcc,#009999)","thinking":"radial-gradient(circle,#ff00ff,#990099)","listening":"radial-gradient(circle,#00ccff,#0033ff)"}
    st.markdown(f"<div id='ai-status' style='background:{color_map[st.session_state.ai_status]}'></div>",unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("## 🌌 EchoSoul Control Center")
    st.text_input("Your name",value=data.get("user_name","User"),key="sidebar_name")
    if st.button("💾 Save Profile"):
        data["user_name"]=st.session_state.sidebar_name; save_data(data); st.success("Profile updated!")

    st.markdown("---")
    st.markdown("### 🎤 Voice Settings")
    voice_opt=st.radio("Choose AI voice",["alloy","verse","amber"],index=0)
    data["voice_mode"]=voice_opt
    mimic_file=st.file_uploader("Upload sample voice to mimic",type=["mp3","wav"])
    if mimic_file: data["mimic_voice"]=mimic_voice_from_upload(mimic_file); save_data(data); st.success("Mimic voice loaded!")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.toggle("Enable adaptive learning",value=True)
    st.markdown("**Privacy & Ethics**")
    st.caption("🔒 Local storage · Inclusive design · Bias mitigation")

# ======================
# MAIN APP
# ======================
st.title(f"✨ EchoSoul — Hi {data.get('user_name','User')}")
menu=st.sidebar.radio("Navigate",["🏠 Home","💬 Chat","🎤 Voice Chat","🕰 Life Timeline","🔐 Vault","📤 Export","ℹ️ About"])

# HOME
if menu=="🏠 Home":
    st.write("Your digital soul that learns, remembers, and now speaks.")
    render_status()

# CHAT
elif menu=="💬 Chat":
    st.subheader("Text Chat")
    txt=st.text_input("Say something")
    if st.button("Send"):
        if txt:
            st.session_state.ai_status="thinking"; render_status()
            data["chat_history"].append(f"You: {txt}")
            reply=generate_reply(txt,data["chat_history"],data["memories"])
            data["chat_history"].append(f"EchoSoul: {reply}"); save_data(data)
            st.session_state.ai_status="ready"; render_status()
            st.markdown(f"**EchoSoul:** {reply}")
    st.write("### History"); [st.write(m) for m in data["chat_history"][-10:]]
    render_status()

# VOICE CHAT
elif menu=="🎤 Voice Chat":
    st.subheader("Talk with EchoSoul")
    audio=st.audio_input("🎙 Speak something")
    if audio:
        st.session_state.ai_status="listening"; render_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(audio.getbuffer()); path=f.name
        text=transcribe_audio(path)
        st.markdown(f"**You said:** {text}")
        reply=generate_reply(text,data["chat_history"],data["memories"])
        data["chat_history"].append(f"You: {text}"); data["chat_history"].append(f"EchoSoul: {reply}"); save_data(data)
        st.markdown(f"**EchoSoul:** {reply}")
        # Speech output
        voice=data["voice_mode"] if not data.get("mimic_voice") else "alloy"
        speech_file=synthesize_speech(reply,voice)
        audio_bytes=open(speech_file,"rb").read()
        st.audio(audio_bytes,format="audio/mp3")
        st.session_state.ai_status="ready"; render_status()

# TIMELINE
elif menu=="🕰 Life Timeline":
    new=st.text_area("Add a memory")
    if st.button("➕ Save"): data["memories"].append(new); save_data(data); st.success("Memory saved!")
    st.write("### Memories"); [st.write("• "+m) for m in data["memories"]]
    render_status()

# VAULT
elif menu=="🔐 Vault":
    pw=st.text_input("Password",type="password")
    note=st.text_area("Secret note")
    if st.button("Encrypt & Save"): 
        if pw and note: data["vault"]["note"]=xor_encrypt_decrypt(note,pw); save_data(data); st.success("Saved")
    if st.button("🔓 Decrypt"):
        if pw and "note" in data["vault"]: st.info(xor_encrypt_decrypt(data["vault"]["note"],pw))
    render_status()

# EXPORT
elif menu=="📤 Export":
    st.download_button("📥 Download",data=json.dumps(data,indent=2),file_name="echosoul.json",mime="application/json")
    st.json(data); render_status()

# ABOUT
elif menu=="ℹ️ About":
    st.write("EchoSoul: An AI that chats, remembers, and speaks with you.")
    st.write("Features: Neon UI · Voice I/O · Mimic mode · Vault · Timeline · Export · Ethics")
    render_status()
