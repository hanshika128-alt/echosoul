# app.py
"""
EchoSoul — Corrected, neon-themed, call + gallery wallpaper
- Fixes deprecation: uses st.query_params (no experimental_get_query_params)
- Replaces experimental_rerun with st.rerun
- Neon UI styled similar to provided screenshot (teal/cyan glow)
- Call flow: browser SpeechRecognition -> adds ?call_msg=... -> app reads st.query_params -> handles once -> plays reply via SpeechSynthesis
Notes:
 - Put OPENAI_API_KEY in Streamlit Secrets to enable AI replies.
 - Vault encryption remains demo-level XOR; don't put sensitive secrets there.
"""

import streamlit as st
import streamlit.components.v1 as components
import os, json, hashlib, base64, datetime, re, html

# ---------------- Page config ----------------
st.set_page_config(page_title="EchoSoul — Neon", layout="wide")
DATA_FILE = "echosoul_data.json"
WALLPAPER_PREFIX = "echosoul_wallpaper"

# ---------------- OpenAI (optional) ----------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except Exception:
    client = None
    OPENAI_AVAILABLE = False

# ---------------- Utilities ----------------
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {
        "profile": {"name": "User", "created": ts_now(), "persona": {"tone": "friendly", "style": "casual"}},
        "timeline": [],
        "vault": [],
        "conversations": [],
        "wallpaper": None
    }

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data()
    else:
        d = default_data()
        save_data(d)
        return d

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------- Demo XOR encryption (do NOT use for real secrets) ----------------
def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def encrypt_text(password, plaintext):
    if not password:
        raise ValueError("Vault password required.")
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_text(password, ciphertext_b64):
    try:
        if not password:
            return None
        data_b = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data_b))
        x = bytes([data_b[i] ^ key[i] for i in range(len(data_b))])
        return x.decode("utf-8")
    except Exception:
        return None

# ---------------- Sentiment / persona (small heuristic) ----------------
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", (text or "").lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    return score / max(1, len(toks))

def update_persona_based_on_sentiment(data, score):
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# ---------------- Memories ----------------
def add_memory(data, title, content):
    item = {
        "id": hashlib.sha1((title + content + ts_now()).encode("utf-8")).hexdigest(),
        "title": title or "Memory",
        "content": content,
        "timestamp": ts_now()
    }
    data["timeline"].append(item)
    save_data(data)
    return item

# ---------------- JS / HTML helpers ----------------
def escape_js(s: str):
    # minimal escaping for JS template
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${").replace("\n", "\\n").replace("\r", "")

def escape_html(s: str):
    return html.escape(s).replace("\n", "<br>")

def browser_speak_js():
    return """
    <script>
    function speakText(text, rate=1.0) {
      if (!('speechSynthesis' in window)) return;
      try {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = rate;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
      } catch(e) {
        console.error(e);
      }
    }
    </script>
    """

# ---------------- OpenAI-backed reply (safe) ----------------
def generate_reply_openai(data, user_msg):
    # update persona
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)
    tone = data["profile"]["persona"].get("tone", "friendly")

    memories = [f"{m['title']}: {m['content']}" for m in data.get("timeline", [])[-6:]]
    mem_block = "\n".join(memories) if memories else "No memories."

    system_prompt = f"""You are EchoSoul, a friendly, attentive digital companion.
Personality tone: {tone}.
Use the following timeline as context if helpful:
{mem_block}
If the user asks 'act like me', roleplay as the user. Keep replies concise and kind.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]
    try:
        model = "gpt-4o-mini"
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512)
        c0 = resp.choices[0]
        # robust parsing for different client shapes
        if hasattr(c0, "message") and isinstance(c0.message, dict):
            reply = c0.message.get("content", "")
        elif hasattr(c0, "message") and hasattr(c0.message, "content"):
            reply = c0.message.content
        else:
            reply = getattr(c0, "text", "") or str(resp)
    except Exception:
        reply = f"(AI unavailable) I heard: {user_msg}"
    # save conversation
    data.setdefault("conversations", []).append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ---------------- Neon CSS (matches second picture feel) ----------------
NEON_CSS = """
<style>
:root{
  --bg:#071021;
  --panel:#0b1320;
  --muted:#9aa6b2;
  --accent:#5ef0ff;
  --accent-2:#8a6cff;
  --bubble-user: linear-gradient(90deg,#7b1fff08,#ff00788a);
  --bubble-bot: linear-gradient(90deg,#051937,#073a5a);
}
body, .stApp { background: radial-gradient(1200px 600px at 80% 10%, rgba(10,30,80,0.3), transparent), var(--bg) !important; color: #dbe8f5; }
.section-title { font-family: 'Inter', sans-serif; color: #bff7ff; text-shadow: 0 0 12px rgba(90,230,255,0.18); }
.stSidebar { background: linear-gradient(180deg, rgba(5,10,20,0.95), rgba(8,12,24,0.98)); }
h1 { color: #bff7ff; text-shadow: 0 4px 18px rgba(10,200,220,0.18); letter-spacing: 0.6px; font-size: 34px; }
.neon-user { background: linear-gradient(90deg, rgba(124,58,237,0.12), rgba(255,0,120,0.06)); padding:14px; border-radius:12px; margin:10px 0; color:#fff; box-shadow: 0 6px 24px rgba(124,58,237,0.06); }
.neon-bot { background: linear-gradient(90deg, rgba(0,220,255,0.06), rgba(50,0,120,0.06)); padding:14px; border-radius:12px; margin:10px 0; color:#eaf6ff; box-shadow: 0 10px 30px rgba(10,40,80,0.12); }
.send-btn { background: linear-gradient(90deg, #6ef0ff, #7be0ff); border: none; padding:10px 18px; border-radius:10px; box-shadow: 0 8px 30px rgba(94,240,255,0.14); color: #021223; font-weight:600; }
.small-muted { color: var(--muted); font-size:13px; }
.sidebar-title { color:#9fefff; font-weight:600; margin-bottom:6px; }
.card { background: rgba(255,255,255,0.02); border-radius:12px; padding:12px; }
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- Load data and apply wallpaper ----------------
data = load_data()

def apply_wallpaper_css(data):
    if data.get("wallpaper") and os.path.exists(data["wallpaper"]):
        try:
            with open(data["wallpaper"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            css = f"""
            <style>
            .stApp {{ background-image: url('data:image/png;base64,{b64}'); background-size: cover; background-position: center; }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
        except Exception:
            pass

apply_wallpaper_css(data)

# ---------------- Sidebar (gallery + settings) ----------------
with st.sidebar:
    st.markdown("<div style='padding:6px 6px 0 6px'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>EchoSoul Settings</div>", unsafe_allow_html=True)
    name = st.text_input("Your name", value=data["profile"].get("name", "User"))
    if st.button("Save name"):
        data["profile"]["name"] = name.strip() or "User"
        save_data(data)
        st.success("Name saved.")
        st.rerun()

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>Voice Settings</div>", unsafe_allow_html=True)
    st.radio("Choose AI voice", ["alloy", "verse", "amber"], index=0, key="voice_choice")
    st.markdown("<div class='small-muted'>Upload a short sample (optional)</div>", unsafe_allow_html=True)
    uploaded_voice = st.file_uploader("MP3/WAV sample (stored locally)", type=["mp3", "wav"])
    if uploaded_voice is not None:
        fpath = f"voice_sample_{hashlib.md5(uploaded_voice.name.encode()).hexdigest()}.bin"
        with open(fpath, "wb") as f:
            f.write(uploaded_voice.getbuffer())
        st.success("Voice sample uploaded (local).")

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>Background Wallpaper (Gallery)</div>", unsafe_allow_html=True)
    st.write("Upload wallpaper from gallery, or pick one you previously uploaded.")
    uploaded = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg", "jpeg", "png"])
    if st.button("Upload & Add to gallery"):
        if uploaded:
            fname = f"gallery_{hashlib.sha1((uploaded.name + ts_now()).encode()).hexdigest()}.{uploaded.name.split('.')[-1]}"
            with open(fname, "wb") as out:
                out.write(uploaded.getbuffer())
            st.success("Added to gallery.")
            st.rerun()
        else:
            st.warning("Choose an image first.")

    # show all images in working dir as gallery choices
    imgs = [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    imgs_sorted = ["(none)"] + sorted(imgs)
    sel = st.selectbox("Choose from gallery", options=imgs_sorted)
    if st.button("Set selected as wallpaper"):
        if sel and sel != "(none)":
            data["wallpaper"] = sel
            save_data(data)
            st.success("Wallpaper applied.")
            st.rerun()
    if st.button("Clear wallpaper"):
        data["wallpaper"] = None
        save_data(data)
        st.rerun()

    st.markdown("---")
    st.checkbox("Enable adaptive learning", value=True, key="adaptive_toggle")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Header + nav ----------------
st.markdown("<h1>✨ EchoSoul — " + html.escape(data["profile"].get("name", "User")) + "</h1>", unsafe_allow_html=True)
nav = st.radio("", ["Home", "Chat", "Call", "Life Timeline", "Vault", "Export", "About"], index=1, horizontal=True)

# ---------------- Handle call_msg from browser recorder
# Use st.query_params (not experimental)
params = st.query_params
if params and "call_msg" in params:
    # take first occurrence
    call_msg = params.get("call_msg")[0]
    # ensure not empty
    if call_msg:
        if "in_call" not in st.session_state:
            st.session_state["in_call"] = True
            st.session_state["call_history"] = []
        # append user message
        st.session_state["call_history"].append({"role": "user", "text": call_msg})
        # generate reply (AI if available)
        if OPENAI_AVAILABLE:
            reply = generate_reply_openai(data, call_msg)
        else:
            reply = f"EchoSoul heard: {call_msg}"
            data.setdefault("conversations", []).append({"user": call_msg, "bot": reply, "ts": ts_now()})
            save_data(data)
        st.session_state["call_history"].append({"role": "bot", "text": reply})
        # remove param from URL so it won't re-trigger on reload
        components.html(
            """<script>
            const u = new URL(window.location);
            u.searchParams.delete('call_msg');
            window.history.replaceState({}, document.title, u.pathname + u.search);
            </script>""",
            height=0
        )
        # speak reply immediately
        components.html(browser_speak_js() + f"<script>speakText(`{escape_js(reply)}`)</script>", height=0)

# ---------------- Home ----------------
if nav == "Home":
    st.subheader("Welcome back — EchoSoul")
    st.write("Use the navigation to chat, call, save memories, and manage your private vault.")
    st.write("Persona tone:", data["profile"]["persona"].get("tone", "friendly"))
    st.write("Memories stored:", len(data.get("timeline", [])))

# ---------------- Chat ----------------
if nav == "Chat":
    st.subheader("Chat with EchoSoul")
    convs = data.get("conversations", [])
    if convs:
        for c in convs[-50:]:
            st.markdown(f"<div class='neon-user'><b>You:</b> {escape_html(c['user'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='neon-bot'><b>EchoSoul:</b> {escape_html(c['bot'])}</div>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No conversation yet — say hi!")

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Say something to EchoSoul...", key="chat_input")
        col1, col2 = st.columns([1,1])
        with col1:
            send = st.form_submit_button("Send")
        with col2:
            save_mem = st.form_submit_button("Save as memory")

    if send:
        if not user_msg.strip():
            st.warning("Please type something.")
        else:
            if OPENAI_AVAILABLE:
                _ = generate_reply_openai(data, user_msg)
            else:
                reply = f"I heard you: {user_msg}"
                data.setdefault("conversations", []).append({"user": user_msg, "bot": reply, "ts": ts_now()})
                save_data(data)
            st.rerun()

    if save_mem:
        if not user_msg.strip():
            st.warning("Cannot save empty memory.")
        else:
            add_memory(data, "User note", user_msg.strip())
            st.success("Saved to timeline.")
            st.rerun()

# ---------------- Call (improved real-feel) ----------------
if nav == "Call":
    st.subheader("Voice Call — speak naturally (browser mic required)")
    if "in_call" not in st.session_state:
        st.session_state["in_call"] = False
        st.session_state["call_history"] = []
    c1, c2 = st.columns([1,1])
    with c1:
        if st.session_state.get("in_call"):
            if st.button("End Call"):
                st.session_state["in_call"] = False
                st.rerun()
        else:
            if st.button("Start Call"):
                st.session_state["in_call"] = True
                st.session_state["call_history"] = []
                st.rerun()
    with c2:
        if st.button("Clear call history"):
            st.session_state["call_history"] = []
            st.rerun()

    if st.session_state.get("in_call"):
        st.info("Call active — press Start recording (microphone permission required). When you stop, your recognized speech will be sent to EchoSoul and EchoSoul will reply (then be spoken by your device).")
        # show history
        for m in st.session_state.get("call_history", []):
            if m["role"] == "user":
                st.markdown(f"<div class='neon-user'><b>You:</b> {escape_html(m['text'])}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='neon-bot'><b>EchoSoul:</b> {escape_html(m['text'])}</div>", unsafe_allow_html=True)
        # recorder: uses browser SpeechRecognition and navigates to ?call_msg=...
        recorder_html = """
        <div style="margin-top:8px;">
          <button id="startRec" style="padding:10px;border-radius:8px;background:#06f;color:white;border:none">🎤 Start recording</button>
          <button id="stopRec" style="padding:10px;border-radius:8px;background:#444;color:white;border:none;margin-left:8px">⏹ Stop recording</button>
          <div id="status" style="margin-top:8px;color:#9aa6b2"></div>
        </div>
        <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          document.getElementById('status').innerText = 'Speech recognition not supported in this browser.';
          document.getElementById('startRec').disabled = true;
          document.getElementById('stopRec').disabled = true;
        } else {
          let recognition = new SpeechRecognition();
          recognition.lang = 'en-US';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
          recognition.onstart = () => { document.getElementById('status').innerText = 'Listening...'; };
          recognition.onend = () => { document.getElementById('status').innerText = 'Stopped.'; };
          recognition.onerror = (e) => { document.getElementById('status').innerText = 'Error: ' + e.error; };
          recognition.onresult = (e) => {
            const text = Array.from(e.results).map(r => r[0].transcript).join('');
            // send recognized text via URL param
            const u = new URL(window.location);
            u.searchParams.set('call_msg', text);
            window.location = u.toString();
          };
          document.getElementById('startRec').onclick = () => { try { recognition.start(); } catch(e) { document.getElementById('status').innerText = e.toString(); } };
          document.getElementById('stopRec').onclick = () => { recognition.stop(); };
        }
        </script>
        """
        components.html(recorder_html, height=140)

# ---------------- Life Timeline ----------------
if nav == "Life Timeline":
    st.subheader("Life Timeline — add, search, view memories")
    col1, col2 = st.columns([2,1])
    with col1:
        q = st.text_input("Search timeline (keyword)")
        items = data.get("timeline", [])
        if q:
            items = [it for it in items if q.lower() in (it["title"] + " " + it["content"]).lower()]
        if items:
            for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
                st.markdown(f"**{it['title']}** — {it['timestamp']}")
                st.write(it["content"])
                if st.button("Delete", key="del_"+it["id"]):
                    data["timeline"] = [x for x in data["timeline"] if x["id"] != it["id"]]
                    save_data(data)
                    st.success("Deleted.")
                    st.rerun()
                st.markdown("---")
        else:
            st.info("No memories yet.")
    with col2:
        st.markdown("### Add new memory")
        title = st.text_input("Title", key="tl_title")
        content = st.text_area("Content", key="tl_content")
        if st.button("Save memory"):
            if not content.strip():
                st.warning("Content cannot be empty.")
            else:
                add_memory(data, title or "Memory", content.strip())
                st.success("Saved.")
                st.rerun()

# ---------------- Vault ----------------
if nav == "Vault":
    st.subheader("Private Vault")
    st.write("Store encrypted notes (demo encryption). Use a session password to encrypt/decrypt.")
    vault_pw = st.text_input("Vault password (session-only)", type="password")
    if dat
