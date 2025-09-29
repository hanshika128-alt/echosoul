# app.py
"""
EchoSoul — Neon + Real-feel Call + Gallery Wallpaper
- Keeps your current chat, timeline, vault, export features
- Adds a call experience using browser SpeechRecognition (client-side) -> server replies -> browser TTS
- Gallery wallpaper picker: upload images to gallery and select any image as chat background
- Neon visual styling for a modern look
Notes:
- For full AI replies add OPENAI_API_KEY in Streamlit Secrets with key name OPENAI_API_KEY
- Browser support: SpeechRecognition works best in Chrome/Edge on Android. Safari may not support it reliably.
- Vault encryption is a demonstration (XOR); do not store sensitive secrets here.
"""

import streamlit as st
import streamlit.components.v1 as components
import os, json, hashlib, base64, datetime, re, html

# ---------------- Config ----------------
st.set_page_config(page_title="EchoSoul — Neon", layout="wide")
DATA_FILE = "echosoul_data.json"
WALLPAPER_PREFIX = "echosoul_wallpaper"

# ---------------- OpenAI client (optional) ----------------
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

# ---------------- Demo XOR 'encryption' ----------------
def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    key = (h * (length // len(h) + 1))[:length]
    return key

def encrypt_text(password, plaintext):
    if password == "":
        raise ValueError("Vault password cannot be empty.")
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_text(password, ciphertext_b64):
    try:
        if password == "":
            return None
        data_b = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data_b))
        x = bytes([data_b[i] ^ key[i] for i in range(len(data_b))])
        return x.decode("utf-8")
    except Exception:
        return None

# ---------------- Sentiment & persona ----------------
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    norm = score / max(1, len(toks))
    return norm

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
    item = {"id": hashlib.sha1((title+content+ts_now()).encode("utf-8")).hexdigest(),
            "title": title or "Memory", "content": content, "timestamp": ts_now()}
    data["timeline"].append(item)
    save_data(data)
    return item

# ---------------- JS / HTML helpers ----------------
def escape_js(s: str):
    # escape for use inside a JS template literal
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${").replace("\n", "\\n").replace("\r", "")

def escape_html(s: str):
    return html.escape(s).replace("\n", "<br>")

def browser_speak_js():
    return """
    <script>
    function speakText(text) {
      if (!('speechSynthesis' in window)) return;
      const u = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
    </script>
    """

# ---------------- OpenAI reply (safe wrappers) ----------------
def generate_reply_openai(data, user_msg):
    s = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, s)
    tone = data["profile"]["persona"].get("tone", "friendly")
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-6:]]
    mem_block = "\n".join(memories) if memories else "No memories available."

    system_prompt = f"""You are EchoSoul, a helpful, empathetic digital companion.
Personality tone: {tone}.
Use the user's timeline as context when replying.
Timeline:
{mem_block}

If the user asks "act like me", roleplay as the user.
Keep replies conversational and concise.
"""
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": user_msg}
    ]
    try:
        model = "gpt-4o-mini"
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512)
        c0 = resp.choices[0]
        if hasattr(c0, 'message') and isinstance(c0.message, dict):
            reply = c0.message.get('content','')
        elif hasattr(c0, 'message') and hasattr(c0.message, 'content'):
            reply = c0.message.content
        else:
            reply = getattr(c0, 'text', '') or str(resp)
    except Exception:
        # graceful fallback
        reply = f"(AI unavailable) I heard: {user_msg}"
    data['conversations'].append({'user': user_msg, 'bot': reply, 'ts': ts_now()})
    save_data(data)
    return reply

# ---------------- Neon CSS ----------------
NEON_CSS = """<style>
/* neon styling for chat bubbles + subtle sidebar style */
.neon-user {
  background: linear-gradient(90deg, rgba(255,0,150,0.12), rgba(255,140,0,0.06));
  box-shadow: 0 0 18px rgba(255,0,150,0.08);
  padding:12px; border-radius:12px; margin:8px 0; color:#fff; max-width:85%;
}
.neon-bot {
  background: linear-gradient(90deg, rgba(0,200,255,0.10), rgba(100,0,255,0.04));
  box-shadow: 0 0 22px rgba(100,0,255,0.10);
  padding:12px; border-radius:12px; margin:8px 0; color:#fff; max-width:85%;
}
.neon-header { color:#aef; text-shadow: 0 0 8px rgba(160,230,255,0.12); }
</style>"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- Load data & apply wallpaper ----------------
data = load_data()

def apply_wallpaper_css(data):
    if data.get('wallpaper') and os.path.exists(data['wallpaper']):
        try:
            with open(data['wallpaper'], 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            css = f"""<style>
            .stApp {{ background-image: url('data:image/png;base64,{b64}'); background-size: cover; background-position: center; }}
            </style>"""
            st.markdown(css, unsafe_allow_html=True)
        except Exception:
            pass

apply_wallpaper_css(data)

# ---------------- Sidebar: gallery + settings ----------------
with st.sidebar:
    st.markdown("<div style='padding:8px'>", unsafe_allow_html=True)
    st.header("EchoSoul Settings")
    name = st.text_input("Your name", value=data["profile"].get("name","User"))
    if st.button("Save name"):
        data["profile"]["name"] = name.strip() or "User"
        save_data(data)
        st.success("Name saved.")
        st.rerun()

    st.markdown("---")
    st.subheader("Voice Settings")
    st.radio("Voice", ["alloy","verse","amber"], index=0, key="voice_choice")
    st.markdown("---")
    st.subheader("Background Wallpaper (Gallery)")
    st.write("Upload wallpaper from your gallery, or pick an existing image below.")
    uploaded = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg","jpeg","png"])
    if st.button("Upload & Add to gallery"):
        if uploaded:
            fname = f"gallery_{hashlib.sha1((uploaded.name+ts_now()).encode()).hexdigest()}.{uploaded.name.split('.')[-1]}"
            with open(fname, "wb") as out:
                out.write(uploaded.getbuffer())
            st.success("Uploaded to gallery.")
            st.rerun()
        else:
            st.warning("Please choose an image first.")
    # list images
    imgs = [f for f in os.listdir('.') if f.lower().endswith(('.jpg','.jpeg','.png'))]
    imgs_sorted = sorted(imgs)
    sel = st.selectbox("Choose from gallery", options=['(none)'] + imgs_sorted)
    if st.button("Set selected as wallpaper"):
        if sel and sel != '(none)':
            data['wallpaper'] = sel
            save_data(data)
            st.success("Wallpaper applied.")
            st.rerun()
    if st.button("Clear wallpaper"):
        data['wallpaper'] = None
        save_data(data)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Main layout & nav ----------------
st.markdown("<div class='neon-header'><h1>💬 EchoSoul — Neon</h1></div>", unsafe_allow_html=True)
nav = st.radio("", ["Home","Chat","Call","Life Timeline","Vault","Export","About"], index=1, horizontal=True)

# If browser recorder uses URL param ?call_msg=<text>, handle it:
params = st.experimental_get_query_params()
if 'call_msg' in params:
    call_msg = params.get('call_msg')[0]
    # ensure call session exists
    if 'in_call' not in st.session_state:
        st.session_state['in_call'] = True
        st.session_state['call_history'] = []
    # add user's spoken message
    st.session_state['call_history'].append({'role': 'user', 'text': call_msg})
    # generate a reply
    if OPENAI_AVAILABLE:
        reply = generate_reply_openai(data, call_msg)
    else:
        reply = f"EchoSoul heard: {call_msg}"
        data['conversations'].append({'user': call_msg, 'bot': reply, 'ts': ts_now()})
        save_data(data)
    st.session_state['call_history'].append({'role': 'bot', 'text': reply})
    # remove param from URL so it doesn't resend on reload
    components.html("""<script>
        const u = new URL(window.location);
        u.searchParams.delete('call_msg');
        window.history.replaceState({}, document.title, u.pathname + u.search);
        </script>""", height=0)
    # auto speak the reply
    components.html(browser_speak_js() + f"<script>speakText(`{escape_js(reply)}`)</script>", height=0)

# ---------------- Home ----------------
if nav == "Home":
    st.subheader("Welcome to EchoSoul")
    st.write("EchoSoul remembers details you share and can hold a near-real 'call' using your browser's speech recognition + TTS.")

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
        st.info("No conversation yet. Say hi!")

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Say something to EchoSoul...", key="chat_input")
        col1, col2 = st.columns(2)
        with col1:
            send = st.form_submit_button("Send")
        with col2:
            save_mem = st.form_submit_button("Save as memory")
    if send:
        if user_msg.strip() == "":
            st.warning("Please type something.")
        else:
            if OPENAI_AVAILABLE:
                _ = generate_reply_openai(data, user_msg)
            else:
                reply = f"I heard you: {user_msg}"
                data['conversations'].append({'user': user_msg, 'bot': reply, 'ts': ts_now()})
                save_data(data)
            st.rerun()
    if save_mem:
        if user_msg.strip() == "":
            st.warning("Cannot save empty memory.")
        else:
            add_memory(data, "User note", user_msg.strip())
            st.success("Saved to timeline.")
            st.rerun()

# ---------------- Call (improved) ----------------
if nav == "Call":
    st.subheader("Voice Call — speak naturally (browser mic required)")
    if 'in_call' not in st.session_state:
        st.session_state['in_call'] = False
        st.session_state['call_history'] = []

    c1, c2 = st.columns([1,1])
    with c1:
        if st.session_state.get('in_call'):
            if st.button("End Call"):
                st.session_state['in_call'] = False
                st.rerun()
        else:
            if st.button("Start Call"):
                st.session_state['in_call'] = True
                st.session_state['call_history'] = []
                st.rerun()

    with c2:
        if st.button("Clear call history"):
            st.session_state['call_history'] = []
            st.rerun()

    if st.session_state.get('in_call'):
        st.info("Call active — press Start recording below (microphone permission required). Use Stop to send what you said.")
        # show call history
        for msg in st.session_state.get('call_history', []):
            if msg['role'] == 'user':
                st.markdown(f"<div class='neon-user'><b>You:</b> {escape_html(msg['text'])}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='neon-bot'><b>EchoSoul:</b> {escape_html(msg['text'])}</div>", unsafe_allow_html=True)

        # recorder UI: uses browser SpeechRecognition and navigates to ?call_msg=<text>
        recorder_html = """
        <div style="margin-top:8px;">
          <button id="startRec">🎤 Start recording</button>
          <button id="stopRec">⏹ Stop recording</button>
          <div id="status" style="margin-top:8px;color:lightgray"></div>
        </div>
        <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          document.getElementById('status').innerText = 'Speech recognition not supported in this browser.';
          document.getElementById('startRec').disabled = true;
          document.getElementById('stopRec').disabled = true;
        } else {
          const recognition = new SpeechRecognition();
          recognition.lang = 'en-US';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
          recognition.onstart = () => { document.getElementById('status').innerText = 'Listening...'; };
          recognition.onend = () => { document.getElementById('status').innerText = 'Stopped.'; };
          recognition.onerror = (e) => { document.getElementById('status').innerText = 'Error: ' + e.error; };
          recognition.onresult = (e) => {
            const text = Array.from(e.results).map(r => r[0].transcript).join('');
            // send the recognized text back to Streamlit via URL param
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
        items = data["timeline"]
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
            if content.strip() == "":
                st.warning("Content cannot be empty.")
            else:
                add_memory(data, title or "Memory", content.strip())
                st.success("Saved.")
                st.rerun()

# ---------------- Vault ----------------
if nav == "Vault":
    st.subheader("Private Vault")
    st.write("Store encrypted notes here (demo XOR encryption). Use a session password to encrypt/decrypt.")
    vault_pw = st.text_input("Vault password (session-only)", type="password")
    if data.get("vault"):
        for idx, v in enumerate(data["vault"]):
            st.markdown(f"**{v['title']}** — {v['timestamp']}")
            if vault_pw:
                dec = decrypt_text(vault_pw, v["cipher"])
                if dec is not None:
                    st.write(dec)
                else:
                    st.write("*Wrong password — cannot decrypt.*")
            else:
                st.write("*Supply vault password to view.*")
            if st.button("Delete vault item", key="vault_del_"+str(idx)):
                data["vault"].pop(idx)
                save_data(data)
                st.success("Deleted.")
                st.rerun()
            st.markdown("---")
    else:
        st.info("No vault items yet.")
    st.markdown("### Add vault item")
    vt = st.text_input("Title", key="vt_title")
    vc = st.text_area("Secret content", key="vt_content")
    if st.button("Save to vault"):
        if not vault_pw:
            st.warning("Set a vault password in the box above.")
        elif vc.strip() == "":
            st.warning("Secret content cannot be empty.")
        else:
            cipher = encrypt_text(vault_pw, vc.strip())
            data["vault"].append({"title": vt or "Vault item", "cipher": cipher, "timestamp": ts_now()})
            save_data(data)
            st.success("Saved to vault.")
            st.rerun()

# ---------------- Export ----------------
if nav == "Export":
    st.subheader("Legacy & Export")
    st.write("Download your EchoSoul data (timeline; vault entries remain encrypted).")
    dump = json.dumps(data, indent=2)
    st.download_button("Download full export (JSON)", dump, f"echosoul_export_{datetime.datetime.utcnow().date()}.json", "application/json")
    st.markdown("---")
    legacy = "\n\n".join([f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data['timeline']])
    st.text_area("Legacy snapshot", legacy, height=300)

# ---------------- About ----------------
if nav == "About":
    st.subheader("About EchoSoul — Neon")
    st.write("Personal companion prototype. Features: chat, call (browser speech), timeline, vault, wallpaper gallery, export.")
    if not OPENAI_AVAILABLE:
        st.info("OpenAI key not configured — EchoSoul responds with simple fallbacks. To enable richer replies, add OPENAI_API_KEY in Streamlit Secrets.")

# Save data at end
save_data(data)
