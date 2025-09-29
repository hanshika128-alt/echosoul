# app.py
"""
EchoSoul — with working Chat, Call, Life Timeline, Vault, Export, and Wallpaper
New features:
- 📞 Voice Call: real in-app call simulation with browser TTS
- 🖼 Wallpaper: choose from gallery uploads
"""

import streamlit as st
import streamlit.components.v1 as components
import os, json, hashlib, base64, datetime, re

# ----------------- Config -----------------
st.set_page_config(page_title="EchoSoul Chat", layout="wide")
DATA_FILE = "echosoul_data.json"
WALLPAPER_PREFIX = "echosoul_wallpaper"

# ----------------- OpenAI client -----------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except Exception:
    client = None
    OPENAI_AVAILABLE = False

# ----------------- Helpers -----------------
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {"profile":{"name":"User","created":ts_now(),"persona":{"tone":"friendly","style":"casual"}},
            "timeline":[], "vault":[], "conversations":[], "wallpaper":None}

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    d = default_data(); save_data(d); return d

def save_data(data):
    with open(DATA_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)

def apply_wallpaper_css(data):
    if data.get("wallpaper"):
        try:
            with open(data["wallpaper"],"rb") as f: b64 = base64.b64encode(f.read()).decode()
            css = f"""
            <style>
            .stApp {{
              background-image: url("data:image/png;base64,{b64}");
              background-size: cover;
              background-position: center;
            }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
        except: pass

# JS helper for speech
def html_escape_js(s): return s.replace("\\","\\\\").replace("`","\\`").replace("$","\\$").replace("\n","\\n")
def browser_speak_js():
    return """
    <script>
    function speakText(text) {
      if (!("speechSynthesis" in window)) return;
      const u = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
    </script>
    """

# ----------------- AI reply -----------------
def generate_reply(data, user_msg):
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-5:]]
    mem_block = "\n".join(memories) if memories else "No memories yet."
    tone = data["profile"]["persona"].get("tone","friendly")
    sys_prompt = f"You are EchoSoul, Hanshika's AI companion. Tone: {tone}. Memories: {mem_block}"
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini",
                    messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}])
        reply = resp.choices[0].message.content
    except:
        reply = "I heard you: " + user_msg
    data["conversations"].append({"user":user_msg,"bot":reply,"ts":ts_now()})
    save_data(data)
    return reply

# ----------------- Load data & apply wallpaper -----------------
data = load_data(); apply_wallpaper_css(data)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("EchoSoul Settings")
    name = st.text_input("Your name", value=data["profile"].get("name","User"))
    if st.button("Save name"):
        data["profile"]["name"] = name.strip() or "User"; save_data(data); st.success("Name saved."); st.rerun()
    st.markdown("---")
    st.header("Voice Settings")
    st.radio("Voice", ["alloy","verse","amber"], index=0)
    st.markdown("---")
    st.header("Background Wallpaper")
    wp = st.file_uploader("Choose wallpaper (from gallery)", type=["jpg","jpeg","png"])
    if st.button("Set wallpaper"):
        if wp: 
            ext = wp.name.split(".")[-1]
            fname = f"{WALLPAPER_PREFIX}.{ext}"
            with open(fname,"wb") as out: out.write(wp.getbuffer())
            data["wallpaper"]=fname; save_data(data); st.success("Wallpaper set."); st.rerun()
    if st.button("Clear wallpaper"):
        data["wallpaper"]=None; save_data(data); st.rerun()

# ----------------- Navigation -----------------
nav = st.radio("", ["Home","Chat","Voice Call","Life Timeline","Vault","Export","About"], index=1, horizontal=True)

# HOME
if nav=="Home":
    st.subheader("Welcome to EchoSoul")
    st.write("Persona tone:", data["profile"]["persona"].get("tone","friendly"))
    st.write("Memories stored:", len(data["timeline"]))
    if data.get("wallpaper"): st.image(data["wallpaper"], use_column_width=True)

# CHAT
if nav=="Chat":
    st.subheader("Chat with EchoSoul")
    for c in data["conversations"][-50:]:
        st.markdown(f"**You:** {c['user']}"); st.markdown(f"**EchoSoul:** {c['bot']}"); st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        msg = st.text_input("Say something..."); send = st.form_submit_button("Send"); mem = st.form_submit_button("Save as memory")
    if send and msg.strip(): generate_reply(data, msg); st.rerun()
    if mem and msg.strip(): data["timeline"].append({"id":hashlib.sha1(msg.encode()).hexdigest(),"title":"Note","content":msg,"timestamp":ts_now()}); save_data(data); st.rerun()

# CALL
if nav=="Voice Call":
    st.subheader("📞 EchoSoul Call")
    if "in_call" not in st.session_state: st.session_state["in_call"]=False; st.session_state["call_history"]=[]
    c1,c2=st.columns(2)
    with c1:
        if st.button("Start Call"): st.session_state["in_call"]=True; st.session_state["call_history"]=[]; st.rerun()
    with c2:
        if st.button("End Call"): st.session_state["in_call"]=False; st.session_state["call_history"]=[]; st.rerun()
    if st.session_state["in_call"]:
        hist=st.session_state["call_history"]
        for h in hist:
            st.markdown(f"**{'You' if h['role']=='user' else 'EchoSoul'}:** {h['text']}")
        new=st.text_input("Say something (call)")
        if st.button("Send in call") and new.strip():
            hist.append({"role":"user","text":new})
            reply=generate_reply(data,new); hist.append({"role":"bot","text":reply})
            st.session_state["call_history"]=hist; st.rerun()
        if hist and hist[-1]["role"]=="bot":
            last=hist[-1]["text"]; st.write("Last reply:"); st.write(last)
            components.html(browser_speak_js(), height=0)
            components.html(f"<button onclick='speakText(`{html_escape_js(last)}`)'>🔊 Speak reply</button>", height=70)

# LIFE TIMELINE
if nav=="Life Timeline":
    st.subheader("Life Timeline")
    q=st.text_input("Search")
    res=[m for m in data["timeline"] if q.lower() in (m["title"]+m["content"]).lower()] if q else data["timeline"]
    for m in reversed(res):
        st.markdown(f"**{m['title']}** — {m['timestamp']}"); st.write(m["content"]); st.markdown("---")
    t=st.text_input("New title"); c=st.text_area("New content")
    if st.button("Save memory") and c.strip():
        data["timeline"].append({"id":hashlib.sha1((t+c).encode()).hexdigest(),"title":t or "Memory","content":c,"timestamp":ts_now()})
        save_data(data); st.rerun()

# VAULT
if nav=="Vault":
    st.subheader("Vault"); st.info("Demo only — not real encryption.")
    pw=st.text_input("Password", type="password")
    for v in data["vault"]:
        st.write(f"**{v['title']}** — {v['timestamp']}"); st.write(v["content"])
    vt=st.text_input("New vault title"); vc=st.text_area("New secret")
    if st.button("Save vault item") and vc.strip():
        data["vault"].append({"title":vt or "Vault","content":vc,"timestamp":ts_now()}); save_data(data); st.rerun()

# EXPORT
if nav=="Export":
    st.subheader("Export"); st.download_button("Download JSON", json.dumps(data,indent=2),"echosoul.json")

# ABOUT
if nav=="About":
    st.subheader("About"); st.write("EchoSoul: Chat + Call + Timeline + Vault + Wallpaper + Export.")
