# app.py
"""
EchoSoul — Neon
- Neon UI
- Chat, Call (real-feel), Life Timeline, Vault, Wallpaper Gallery
- OpenAI GPT-4o-mini for replies (if OPENAI_API_KEY in Streamlit Secrets)
"""

import streamlit as st
import streamlit.components.v1 as components
import os, json, hashlib, base64, datetime, re, html

# ---------------- Page config ----------------
st.set_page_config(page_title="EchoSoul — Neon", layout="wide")
DATA_FILE = "echosoul_data.json"

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

# ---------------- Simple XOR encryption ----------------
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

# ---------------- Sentiment ----------------
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", (text or "").lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    return score / max(1, len(toks))

def update_persona(data, score):
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

# ---------------- JS helpers ----------------
def escape_js(s: str):
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
      } catch(e) { console.error(e); }
    }
    </script>
    """

# ---------------- OpenAI reply ----------------
def generate_reply_openai(data, user_msg):
    score = sentiment_score(user_msg)
    update_persona(data, score)
    tone = data["profile"]["persona"].get("tone", "friendly")

    memories = [f"{m['title']}: {m['content']}" for m in data.get("timeline", [])[-6:]]
    mem_block = "\n".join(memories) if memories else "No memories."

    system_prompt = f"""You are EchoSoul, a friendly digital companion.
Tone: {tone}.
Timeline:
{mem_block}
If user says 'act like me', roleplay as them. Keep replies warm and concise.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=512)
        c0 = resp.choices[0]
        if hasattr(c0, "message") and isinstance(c0.message, dict):
            reply = c0.message.get("content", "")
        elif hasattr(c0, "message") and hasattr(c0.message, "content"):
            reply = c0.message.content
        else:
            reply = getattr(c0, "text", "") or str(resp)
    except Exception:
        reply = f"(AI unavailable) I heard: {user_msg}"
    data.setdefault("conversations", []).append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ---------------- Neon CSS ----------------
NEON_CSS = """
<style>
body, .stApp { background: #071021; color: #dbe8f5; }
h1 { color: #bff7ff; text-shadow: 0 0 12px rgba(90,230,255,0.5); }
.neon-user { background: linear-gradient(90deg, rgba(124,58,237,0.2), rgba(255,0,120,0.2)); padding:12px; border-radius:10px; margin:6px 0; }
.neon-bot { background: linear-gradient(90deg, rgba(0,220,255,0.15), rgba(50,0,120,0.15)); padding:12px; border-radius:10px; margin:6px 0; }
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- Data + wallpaper ----------------
data = load_data()

def apply_wallpaper(data):
    if data.get("wallpaper") and os.path.exists(data["wallpaper"]):
        try:
            with open(data["wallpaper"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"<style>.stApp {{background-image: url('data:image/png;base64,{b64}');background-size: cover;}}</style>",
                unsafe_allow_html=True
            )
        except Exception:
            pass

apply_wallpaper(data)

# ---------------- Sidebar ----------------
with st.sidebar:
    name = st.text_input("Your name", value=data["profile"].get("name", "User"))
    if st.button("Save name"):
        data["profile"]["name"] = name.strip() or "User"
        save_data(data)
        st.rerun()

    uploaded = st.file_uploader("Upload wallpaper", type=["jpg", "png"])
    if st.button("Set as wallpaper"):
        if uploaded:
            fname = "wall_" + uploaded.name
            with open(fname, "wb") as f:
                f.write(uploaded.getbuffer())
            data["wallpaper"] = fname
            save_data(data)
            st.rerun()

# ---------------- Nav ----------------
st.markdown(f"<h1>✨ EchoSoul — {html.escape(data['profile']['name'])}</h1>", unsafe_allow_html=True)
nav = st.radio("", ["Home", "Chat", "Call", "Life Timeline", "Vault", "Export", "About"], index=1, horizontal=True)

# ---------------- Chat ----------------
if nav == "Chat":
    for c in data.get("conversations", [])[-30:]:
        st.markdown(f"<div class='neon-user'><b>You:</b> {escape_html(c['user'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='neon-bot'><b>EchoSoul:</b> {escape_html(c['bot'])}</div>", unsafe_allow_html=True)
    msg = st.text_input("Say something...")
    if st.button("Send"):
        if OPENAI_AVAILABLE:
            _ = generate_reply_openai(data, msg)
        else:
            reply = f"I heard you: {msg}"
            data["conversations"].append({"user": msg, "bot": reply, "ts": ts_now()})
            save_data(data)
        st.rerun()

# ---------------- Call ----------------
if nav == "Call":
    st.write("Press Start recording to talk with EchoSoul.")
    recorder_html = """
    <button onclick="startCall()">🎤 Start recording</button>
    <script>
    function startCall(){
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition){ alert('No speech recognition'); return; }
      let rec = new SpeechRecognition();
      rec.lang = 'en-US'; rec.onresult = (e)=>{
        const txt = e.results[0][0].transcript;
        const u = new URL(window.location);
        u.searchParams.set('call_msg', txt);
        window.location = u;
      };
      rec.start();
    }
    </script>
    """
    components.html(recorder_html, height=60)

    params = st.query_params
    if params and "call_msg" in params:
        msg = params.get("call_msg")[0]
        if msg:
            if OPENAI_AVAILABLE:
                reply = generate_reply_openai(data, msg)
            else:
                reply = f"EchoSoul heard: {msg}"
                data["conversations"].append({"user": msg, "bot": reply, "ts": ts_now()})
                save_data(data)
            st.markdown(f"<div class='neon-user'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='neon-bot'><b>EchoSoul:</b> {reply}</div>", unsafe_allow_html=True)
            components.html(browser_speak_js() + f"<script>speakText(`{escape_js(reply)}`)</script>", height=0)

# ---------------- Life Timeline ----------------
if nav == "Life Timeline":
    st.write("Memories:")
    for m in data.get("timeline", []):
        st.write(f"**{m['title']}** — {m['content']}")
    title = st.text_input("Memory title")
    cont = st.text_area("Memory content")
    if st.button("Save memory"):
        add_memory(data, title, cont)
        st.rerun()

# ---------------- Vault ----------------
if nav == "Vault":
    pw = st.text_input("Vault password", type="password")
    if data.get("vault"):
        for v in data["vault"]:
            st.write(f"**{v['title']}**")
            if pw:
                st.write(decrypt_text(pw, v["cipher"]))
    vt = st.text_input("Title")
    vc = st.text_area("Secret")
    if st.button("Save to vault"):
        if pw and vc.strip():
            cipher = encrypt_text(pw, vc.strip())
            data["vault"].append({"title": vt or "Item", "cipher": cipher, "timestamp": ts_now()})
            save_data(data)
            st.rerun()

# ---------------- Export ----------------
if nav == "Export":
    dump = json.dumps(data, indent=2)
    st.download_button("Download JSON", dump, "echosoul.json")

# ---------------- About ----------------
if nav == "About":
    st.write("EchoSoul Neon — AI companion with chat, call, timeline, vault, wallpaper.")
