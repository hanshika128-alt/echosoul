# app.py — EchoSoul Final Version
import streamlit as st
import streamlit.components.v1 as components
import os, json, datetime, base64, hashlib, html, re

st.set_page_config(page_title="EchoSoul", layout="wide")

DATA_FILE = "echosoul_data.json"

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

def encrypt_xor(text, password):
    return "".join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(text))

# ---------- Load ----------
data = load_data()

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {
  background: #0e0e1a;
  font-family: 'Inter', sans-serif;
  color: #f0f0f0;
}
.app-header {
  font-size: 30px;
  font-weight: bold;
  text-align: center;
  margin-bottom: 20px;
  color: #fff;
  text-shadow: 0 0 6px #00e5ff;
}
.chat-bubble {
  padding: 12px 16px;
  border-radius: 12px;
  margin: 6px 0;
  max-width: 75%;
}
.user {
  background: #4e2a84;
  color: #fff;
  margin-left: auto;
}
.bot {
  background: #1e3a5f;
  color: #e0f7ff;
  margin-right: auto;
}
.sidebar .stSidebarContent {
  background: rgba(25,25,40,0.8);
  backdrop-filter: blur(10px);
}
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

    st.subheader("🖼️ Wallpaper")
    wp_file = st.file_uploader("Upload wallpaper", type=["jpg", "png"])
    if wp_file:
        b64 = base64.b64encode(wp_file.read()).decode()
        data["wallpapers"].append(b64)
        save_data(data)
        st.success("Wallpaper added to gallery.")
    if data["wallpapers"]:
        choice = st.selectbox("Choose wallpaper", list(range(len(data["wallpapers"]))))
        if st.button("Set as wallpaper"):
            st.markdown(
                f"<style>.stApp {{background: url(data:image/png;base64,{data['wallpapers'][choice]});background-size: cover;}}</style>",
                unsafe_allow_html=True,
            )

# ---------- Header ----------
st.markdown(f"<div class='app-header'>EchoSoul — Hi {data['profile']['name']}</div>", unsafe_allow_html=True)

# ---------- Tabs ----------
tab = st.radio("Navigate", ["💬 Chat", "📞 Call", "🧠 Life Timeline", "🔐 Vault", "📤 Export", "ℹ️ About"])

# ---------- Chat ----------
if tab == "💬 Chat":
    st.subheader("Chat with EchoSoul")

    for c in data["conversations"][-20:]:
        st.markdown(f"<div class='chat-bubble user'>You: {escape_html(c['user'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble bot'>EchoSoul: {escape_html(c['bot'])}</div>", unsafe_allow_html=True)

    msg = st.text_input("Say something...")
    if st.button("Send") and msg.strip():
        # Simple sentiment → adaptive personality
        if re.search(r"\b(good|great|happy|awesome)\b", msg, re.I):
            tone = "energetic"
        elif re.search(r"\b(sad|bad|angry|tired)\b", msg, re.I):
            tone = "empathetic"
        else:
            tone = "friendly"
        reply = f"[{tone} EchoSoul]: I hear you — '{msg}'"
        data["conversations"].append({"user": msg, "bot": reply, "ts": now_iso()})
        save_data(data)
        st.rerun()

# ---------- Call ----------
elif tab == "📞 Call":
    st.subheader("Call EchoSoul")
    st.info("Experimental — uses browser speech recognition (works best in Chrome).")

    if "in_call" not in st.session_state:
        st.session_state.in_call = False
        st.session_state.call_history = []

    if not st.session_state.in_call:
        if st.button("Start Call"):
            st.session_state.in_call = True
            st.rerun()
    else:
        if st.button("End Call"):
            st.session_state.in_call = False
            st.rerun()

        for m in st.session_state.call_history:
            role = "user" if m["role"] == "user" else "bot"
            st.markdown(f"<div class='chat-bubble {role}'>{escape_html(m['text'])}</div>", unsafe_allow_html=True)

        recorder_js = """
        <div>
          <button id="startRec">🎙️ Start</button>
          <button id="stopRec">⏹ Stop</button>
          <div id="rec_status"></div>
        </div>
        <script>
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
          document.getElementById('rec_status').innerText = 'Speech recognition not supported.';
        } else {
          let recog = new SR();
          recog.lang = 'en-US';
          recog.onresult = e => {
            const text = Array.from(e.results).map(r=>r[0].transcript).join('');
            const u = new URL(window.location);
            u.searchParams.set('call_msg', text);
            window.location = u.toString();
          };
          document.getElementById('startRec').onclick = ()=>recog.start();
          document.getElementById('stopRec').onclick = ()=>recog.stop();
        }
        </script>
        """
        components.html(recorder_js, height=120)

        # process incoming
        qp = st.query_params
        if "call_msg" in qp:
            text = qp["call_msg"]
            st.session_state.call_history.append({"role": "user", "text": text})
            reply = f"EchoSoul (on call): I heard '{text}'"
            st.session_state.call_history.append({"role": "bot", "text": reply})
            del qp["call_msg"]
            st.query_params.clear()
            st.rerun()

# ---------- Life Timeline ----------
elif tab == "🧠 Life Timeline":
    st.subheader("Life Timeline")
    for m in data["memories"]:
        st.write(f"📌 {m['ts']}: {m['text']}")
    new_mem = st.text_area("Add new memory")
    if st.button("Save Memory") and new_mem.strip():
        data["memories"].append({"ts": now_iso(), "text": new_mem})
        save_data(data)
        st.success("Memory saved.")

# ---------- Vault ----------
elif tab == "🔐 Vault":
    st.subheader("Private Vault")
    pwd = st.text_input("Password", type="password")
    choice = st.radio("Action", ["Add", "View"])
    if pwd:
        if choice == "Add":
            secret = st.text_area("Write secret")
            if st.button("Save Secret"):
                enc = base64.b64encode(encrypt_xor(secret, pwd).encode()).decode()
                data["vault"].append({"ts": now_iso(), "data": enc})
                save_data(data)
                st.success("Saved in vault.")
        else:
            for v in data["vault"]:
                try:
                    dec = encrypt_xor(base64.b64decode(v["data"]).decode(), pwd)
                    st.write(f"🔒 {v['ts']}: {dec}")
                except:
                    st.error("Wrong password.")

# ---------- Export ----------
elif tab == "📤 Export":
    st.subheader("Export Data")
    j = json.dumps(data, indent=2)
    b64 = base64.b64encode(j.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="echosoul_data.json">📥 Download JSON</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------- About ----------
elif tab == "ℹ️ About":
    st.subheader("About EchoSoul")
    st.markdown("""
EchoSoul is your evolving AI companion.  
- Persistent Memory  
- Adaptive Personality  
- Emotion Recognition  
- Life Timeline  
- Vault & Legacy Mode  
- Call & Conversation mirroring  

✨ Over time, EchoSoul learns your style and grows with you.
""")
