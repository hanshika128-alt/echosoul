# app.py
import streamlit as st
import streamlit.components.v1 as components
import os, json, base64, datetime, html, hashlib

st.set_page_config(page_title="EchoSoul — Neon", layout="wide")

DATA_FILE = "echosoul_data.json"

# ---------------- Helpers ----------------
def ts_now():
    return datetime.datetime.utcnow().isoformat()

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            return json.load(open(DATA_FILE, "r", encoding="utf-8"))
        except:
            return {"conversations": [], "wallpaper": None, "name": "User"}
    return {"conversations": [], "wallpaper": None, "name": "User"}

def save_data(d):
    json.dump(d, open(DATA_FILE, "w", encoding="utf-8"), indent=2)

def escape_html(s): return html.escape(s).replace("\n", "<br>")

# ---------------- Neon CSS ----------------
NEON_CSS = """
<style>
body, .stApp {
  background: radial-gradient(circle at top, #0a0f2c, #000000);
  color: #e0f7ff;
}
h1 {
  font-size: 2.2em;
  color: #8be9fd;
  text-align: center;
  text-shadow: 0 0 8px #00eaff, 0 0 16px #00eaff, 0 0 24px #00eaff;
}
.chat-box {
  border-radius: 12px;
  padding: 12px 16px;
  margin: 8px 0;
  max-width: 80%;
}
.user {
  background: linear-gradient(135deg, #ff0080aa, #7928caaa);
  color: #fff;
  align-self: flex-end;
  text-shadow: 0 0 6px #ff4da6;
}
.bot {
  background: linear-gradient(135deg, #00eaffaa, #0066ffaa);
  color: #fff;
  text-shadow: 0 0 6px #00eaff;
}
#chat-area {
  padding: 12px;
  border-radius: 16px;
  min-height: 60vh;
  overflow-y: auto;
}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- Data ----------------
data = load_data()

# ---------------- Wallpaper ----------------
def apply_wallpaper():
    if data.get("wallpaper") and os.path.exists(data["wallpaper"]):
        with open(data["wallpaper"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        #chat-area {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, unsafe_allow_html=True)

apply_wallpaper()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.text_input("Your name", value=data.get("name", "User"), key="name_input")
    if st.button("Save name"):
        data["name"] = st.session_state.name_input
        save_data(data)
        st.rerun()

    wp = st.file_uploader("Upload wallpaper", type=["jpg","png"])
    if wp and st.button("Set as wallpaper"):
        fname = "wall_" + hashlib.sha1(wp.name.encode()).hexdigest() + ".png"
        with open(fname, "wb") as f: f.write(wp.getbuffer())
        data["wallpaper"] = fname
        save_data(data)
        st.rerun()

# ---------------- Nav ----------------
st.markdown(f"<h1>✨ EchoSoul — {escape_html(data['name'])}</h1>", unsafe_allow_html=True)
nav = st.radio("", ["Chat", "Call", "About"], horizontal=True)

# ---------------- Chat ----------------
if nav == "Chat":
    st.markdown("<div id='chat-area'>", unsafe_allow_html=True)
    for c in data["conversations"]:
        st.markdown(f"<div class='chat-box user'><b>You:</b> {escape_html(c['user'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-box bot'><b>EchoSoul:</b> {escape_html(c['bot'])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    msg = st.text_input("Type a message...")
    if st.button("Send"):
        reply = f"(EchoSoul): I heard you say: {msg}"
        data["conversations"].append({"user": msg, "bot": reply, "ts": ts_now()})
        save_data(data)
        st.rerun()

# ---------------- Call ----------------
if nav == "Call":
    st.write("🎤 **Voice Call with EchoSoul** (browser speech recognition + AI reply).")
    call_ui = """
    <button onclick="startCall()">Start Talking</button>
    <div id="transcript"></div>
    <script>
    function startCall(){
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition){ alert('Not supported'); return; }
      let recog = new SpeechRecognition();
      recog.lang = "en-US";
      recog.continuous = true;
      recog.interimResults = false;
      recog.onresult = (event)=>{
        let txt = event.results[event.results.length-1][0].transcript;
        document.getElementById("transcript").innerHTML = "<b>You:</b> "+txt;
        // Fake reply
        let reply = "EchoSoul says: " + txt;
        let u = new SpeechSynthesisUtterance(reply);
        u.rate=1; speechSynthesis.speak(u);
        document.getElementById("transcript").innerHTML += "<br><b>EchoSoul:</b> "+reply;
      };
      recog.start();
    }
    </script>
    """
    components.html(call_ui, height=300)

# ---------------- About ----------------
if nav == "About":
    st.write("✨ EchoSoul Neon — Chat, Voice Call, Wallpaper, Glowing Neon UI.")
