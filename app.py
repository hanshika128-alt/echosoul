# app.py
"""
EchoSoul — Personal AI companion (single-file Streamlit app)
Features:
- Chat with OpenAI gpt-4o-mini (if OPENAI_API_KEY is set in Streamlit Secrets)
- Persistent memory (echosoul_data.json)
- Life Timeline (add/view/search memories)
- Call: browser SpeechRecognition -> server AI reply -> browser SpeechSynthesis (real-feel)
- Wallpaper gallery: upload & choose wallpaper for chat area
- Private Vault (Fernet if cryptography installed; fallback XOR demo with clear warning)
- Export & Legacy snapshot
- Adaptive persona via simple sentiment heuristic
- Explainability visuals (sentiment + confidence heuristic)
- Header: "✨ EchoSoul — Hi <name>"
"""
import streamlit as st
import streamlit.components.v1 as components
import json, os, base64, datetime, hashlib, re, html, sys

# -------------- Config --------------
st.set_page_config(page_title="EchoSoul — Neon (Clean)", layout="wide")
DATA_FILE = "echosoul_data.json"

# -------------- Optional imports --------------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    # client will be created below using st.secrets if present
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

FERNET_AVAILABLE = False
try:
    from cryptography.fernet import Fernet, InvalidToken
    FERNET_AVAILABLE = True
except Exception:
    FERNET_AVAILABLE = False

# -------------- Utilities --------------
def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def short_id(s: str):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        default = {
            "profile": {"name": "User", "persona": {"tone": "friendly", "style": "casual"}, "created": now_iso()},
            "conversations": [],
            "timeline": [],
            "vault": [],
            "wallpaper": None,
            "voice_sample": None
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)

def load_data():
    ensure_datafile()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(d):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

def escape_html(s: str):
    return html.escape(s).replace("\n", "<br>")

def sanitize_js(s: str):
    # Make string safe for inclusion inside JS template literals
    if s is None:
        return ""
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n").replace("\r", "")

# -------------- Vault encryption helpers --------------
if FERNET_AVAILABLE:
    # If cryptography is installed, we will derive a key from the user's password using SHA256
    import hashlib
    def derive_fernet_key(password: str) -> bytes:
        # Derive 32 byte key from password (NOT PBKDF2 — simple; for production use PBKDF2 + salt)
        h = hashlib.sha256(password.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(h)

    def vault_encrypt(password: str, plaintext: str) -> str:
        key = derive_fernet_key(password)
        f = Fernet(key)
        return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def vault_decrypt(password: str, ciphertext: str):
        try:
            key = derive_fernet_key(password)
            f = Fernet(key)
            return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except Exception:
            return None
else:
    # Fallback XOR (DEMO only); show warning in UI
    import hashlib
    def _derive_xor_key(password: str, length: int):
        h = hashlib.sha256(password.encode("utf-8")).digest()
        return (h * (length // len(h) + 1))[:length]

    def vault_encrypt(password: str, plaintext: str) -> str:
        b = plaintext.encode("utf-8")
        key = _derive_xor_key(password, len(b))
        x = bytes([b[i] ^ key[i] for i in range(len(b))])
        return base64.b64encode(x).decode("utf-8")

    def vault_decrypt(password: str, ciphertext: str):
        try:
            b = base64.b64decode(ciphertext.encode("utf-8"))
            key = _derive_xor_key(password, len(b))
            x = bytes([b[i] ^ key[i] for i in range(len(b))])
            return x.decode("utf-8")
        except Exception:
            return None

# -------------- Sentiment & persona (simple heuristics) --------------
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_and_confidence(text: str):
    toks = re.findall(r"\w+", (text or "").lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = (pos - neg) / max(1, len(toks))
    if score > 0.06:
        label = "positive"
    elif score < -0.06:
        label = "negative"
    else:
        label = "neutral"
    # confidence heuristic: message length and presence of sentiment words
    conf = min(0.95, 0.2 + min(0.75, (abs(pos - neg) + len(toks) / 20)))
    return label, round(score, 3), round(conf, 2)

# -------------- OpenAI helper --------------
def make_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    try:
        # requires OPENAI_API_KEY in st.secrets
        key = st.secrets.get("OPENAI_API_KEY", None)
        if not key:
            return None
        client = OpenAI(api_key=key)
        return client
    except Exception:
        return None

OPENAI_CLIENT = make_openai_client()
OPENAI_ENABLED = OPENAI_CLIENT is not None

def ask_openai(data, user_text, max_tokens=512):
    # build system prompt with recent history + memories
    persona = data.get("profile", {}).get("persona", {})
    tone = persona.get("tone", "friendly")
    name = data.get("profile", {}).get("name", "User")
    recent_conv = data.get("conversations", [])[-12:]
    recent_mem = data.get("timeline", [])[-6:]
    mem_text = "\n".join([f"{m['timestamp']} | {m['title']}: {m['content']}" for m in recent_mem]) or "No memories."
    hist_text = "\n".join([f"User: {c['user']}\nEchoSoul: {c['bot']}" for c in recent_conv]) or "No recent conversation."
    system_prompt = (
        f"You are EchoSoul, a helpful personal AI for {name}.\n"
        f"Personality tone: {tone}.\n"
        "Use the timeline memories and recent conversation when helpful. Be honest about uncertainty and avoid inventing facts.\n\n"
        f"TIMELINE:\n{mem_text}\n\nRECENT_HISTORY:\n{hist_text}\n\n"
        "If the user asks 'act like me' roleplay as the user. Keep replies concise and kind."
    )
    if not OPENAI_ENABLED:
        return f"(Offline) EchoSoul heard: {user_text}"

    try:
        # robustly call client's chat completion
        resp = OPENAI_CLIENT.chat.completions.create(model="gpt-4o-mini", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ], max_tokens=max_tokens)
        choice = resp.choices[0]
        # different response shapes
        if hasattr(choice, "message") and isinstance(choice.message, dict):
            out = choice.message.get("content", "")
        elif hasattr(choice, "message") and hasattr(choice.message, "content"):
            out = choice.message.content
        else:
            out = getattr(choice, "text", "") or str(resp)
        return str(out)
    except Exception as e:
        # don't leak sensitive details of exceptions
        return f"(AI error) Unable to generate reply. {str(e)[:200]}"

# -------------- UI CSS (clean modern header + readable chat) --------------
CUSTOM_CSS = """
<style>
/* overall */
body, .stApp { background: linear-gradient(180deg, #0f1724 0%, #0b1220 100%); color: #e6f7ff; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
/* header */
.header { font-size: 32px; font-weight: 700; color: #bfefff; text-shadow: 0 4px 20px rgba(78,224,255,0.06); margin-bottom: 12px; }
/* main panels */
.panel { background: rgba(255,255,255,0.02); border-radius: 12px; padding: 14px; margin-bottom: 12px; }
/* chat area */
#chat-area { min-height: 50vh; max-height: 70vh; overflow-y: auto; padding: 12px; border-radius: 10px; background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent); }
/* chat bubbles */
.chat-bubble { padding: 12px 14px; border-radius: 12px; margin: 10px 0; max-width: 78%; line-height: 1.45; box-shadow: 0 6px 20px rgba(2,6,23,0.6); }
.user { background: linear-gradient(90deg,#8a2be2,#ff5fa2); color: #fff; margin-left: auto; }
.bot { background: linear-gradient(90deg,#0066ff,#00e5ff); color: #fff; margin-right: auto; }
/* sidebar glass */
.stSidebar { background: rgba(6,10,20,0.85); backdrop-filter: blur(6px); }
/* buttons */
button.stButton>button { background: linear-gradient(90deg,#4ee0ff,#8a6cff); color:#021425; border-radius:8px; padding:8px 12px; font-weight:600; border:none; }
.small-muted { color: #9fb6c8; font-size:13px; }
.note { color:#a6cbe0; font-size:13px; }
.xbox { background: rgba(255,255,255,0.01); border-left: 4px solid rgba(78,224,255,0.08); padding:8px; border-radius:8px; margin:8px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------- Load data --------------
ensure_datafile()
data = load_data()

# -------------- Apply chat wallpaper if set --------------
def apply_chat_wallpaper():
    wp = data.get("wallpaper")
    if wp and os.path.exists(wp):
        try:
            b64 = base64.b64encode(open(wp, "rb").read()).decode()
            css = f"""
            <style>
            #chat-area {{
                background-image: url('data:image/png;base64,{b64}');
                background-size: cover;
                background-position: center;
            }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
        except Exception:
            pass

apply_chat_wallpaper()

# -------------- Sidebar --------------
with st.sidebar:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#bfefff;'>EchoSoul — Settings</div>", unsafe_allow_html=True)
    name_in = st.text_input("Your name", value=data.get("profile", {}).get("name", "User"))
    if st.button("Save name"):
        data["profile"]["name"] = name_in.strip() or "User"
        save_data(data)
        st.success("Profile saved.")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("<div class='small-muted'>Voice & Call</div>", unsafe_allow_html=True)
    st.selectbox("Browser voice preference", ["default", "Google US English", "Alloy", "Verse", "Amber"], key="voice_choice")
    st.slider("Speech rate", 0.6, 1.6, 1.0, key="speech_rate")
    st.slider("Speech pitch", 0.6, 1.5, 1.0, key="speech_pitch")
    st.file_uploader("Upload short voice sample (mp3/wav) — demo", type=["mp3","wav"], key="voice_sample")
    if st.button("Save voice sample"):
        uf = st.session_state.get("voice_sample")
        if uf:
            fname = f"voice_{short_id(uf.name)}.{uf.name.split('.')[-1]}"
            with open(fname, "wb") as out:
                out.write(uf.getbuffer())
            data["voice_sample"] = fname
            save_data(data)
            st.success("Saved voice sample locally.")
        else:
            st.warning("Choose a file first.")

    st.markdown("---")
    st.markdown("<div class='small-muted'>Wallpaper (Gallery)</div>", unsafe_allow_html=True)
    up = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg","jpeg","png"], key="wp_upload")
    if st.button("Upload & Add to gallery"):
        uf = st.session_state.get("wp_upload")
        if uf:
            fname = f"wall_{short_id(uf.name)}.{uf.name.split('.')[-1]}"
            with open(fname, "wb") as out:
                out.write(uf.getbuffer())
            st.success("Added to gallery.")
            st.experimental_rerun()
        else:
            st.warning("Choose an image file.")

    # list images in working dir
    imgs = [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    imgs_sorted = ["(none)"] + sorted(imgs)
    sel = st.selectbox("Choose from gallery", imgs_sorted, key="gallery_select")
    if st.button("Set selected as chat wallpaper"):
        if sel and sel != "(none)":
            data["wallpaper"] = sel
            save_data(data)
            st.success("Wallpaper applied.")
            st.experimental_rerun()

    if st.button("Clear wallpaper"):
        data["wallpaper"] = None
        save_data(data)
        st.success("Cleared wallpaper.")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("<div class='small-muted'>Privacy & Vault</div>", unsafe_allow_html=True)
    if FERNET_AVAILABLE:
        st.markdown("<div class='note'>Vault encryption: Fernet (cryptography) available.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='note'>Vault encryption: cryptography not installed. Falling back to demo encryption (not secure). To enable secure vault, add 'cryptography' to requirements.txt.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------- Header --------------
st.markdown(f"<div class='header'>✨ EchoSoul — Hi {escape_html(data.get('profile',{}).get('name','User'))} ✨</div>", unsafe_allow_html=True)

# -------------- Navigation --------------
page = st.radio("", ["Chat", "Call", "Life Timeline", "Vault", "Export", "About"], index=0, horizontal=True)

# -------------- Chat Page --------------
if page == "Chat":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Explainability: below you can see which memories were considered and a sentiment heuristic for your input.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # show chat area
    st.markdown("<div id='chat-area' class='panel'>", unsafe_allow_html=True)
    convs = data.get("conversations", [])[-200:]
    for c in convs:
        st.markdown(f"<div class='chat-bubble user'>{escape_html(c.get('user',''))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble bot'>{escape_html(c.get('bot',''))}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # input form
    with st.form("chat_form", clear_on_submit=False):
        user_input = st.text_area("Say something to EchoSoul. You can ask it to 'act like me'.", height=100, key="user_input")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            send = st.form_submit_button("Send")
        with col2:
            suggest = st.form_submit_button("Suggest reply")
        with col3:
            save_mem = st.form_submit_button("Save memory")

    if suggest and user_input.strip():
        # generate suggestion (editable), show heuristics
        suggestion = ask_openai(data, user_input) if OPENAI_ENABLED else f"(Offline) Suggested reply: {user_input}"
        st.markdown("<div class='xbox'><b>Suggested reply (edit and send):</b></div>", unsafe_allow_html=True)
        edited = st.text_area("Edit suggested reply", value=suggestion, key="edited_reply")
        label, score, conf = sentiment_and_confidence(user_input)
        st.markdown(f"<div class='note'>Input sentiment: <b>{label}</b> (score={score}) • Confidence heuristic: {conf}</div>", unsafe_allow_html=True)
        if st.button("Send edited reply"):
            data.setdefault("conversations", []).append({"user": user_input, "bot": edited, "ts": now_iso()})
            save_data(data)
            st.success("Sent.")
            st.experimental_rerun()

    if send and user_input.strip():
        # update persona
        label, score, conf = sentiment_and_confidence(user_input)
        if score > 0.06:
            data["profile"]["persona"]["tone"] = "energetic"
        elif score < -0.06:
            data["profile"]["persona"]["tone"] = "empathetic"
        else:
            data["profile"]["persona"]["tone"] = "friendly"
        # ask AI
        reply = ask_openai(data, user_input) if OPENAI_ENABLED else f"(Offline) EchoSoul heard: {user_input}"
        data.setdefault("conversations", []).append({"user": user_input, "bot": reply, "ts": now_iso()})
        save_data(data)
        st.experimental_rerun()

    if save_mem and user_input.strip():
        item = {"id": short_id(user_input + now_iso()), "title": "Note", "content": user_input, "timestamp": now_iso()}
        data.setdefault("timeline", []).append(item)
        save_data(data)
        st.success("Saved to timeline.")
        st.experimental_rerun()

# -------------- Call Page (real-feel) --------------
elif page == "Call":
    st.markdown("<div class='panel'><b>Call EchoSoul — speak naturally; reply will be spoken by your device.</b></div>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Grant mic permission. Best in Chrome/Edge on Android. The app captures recognized text and sends it to the app, which replies and plays the reply with browser speech synthesis.</div>", unsafe_allow_html=True)

    if "in_call" not in st.session_state:
        st.session_state.in_call = False
        st.session_state.call_history = []

    c1, c2 = st.columns([1,1])
    with c1:
        if not st.session_state.in_call:
            if st.button("Start Call"):
                st.session_state.in_call = True
                st.session_state.call_history = []
                st.experimental_rerun()
        else:
            if st.button("End Call"):
                st.session_state.in_call = False
                st.experimental_rerun()
    with c2:
        if st.button("Clear history"):
            st.session_state.call_history = []
            st.experimental_rerun()

    if st.session_state.in_call:
        # render call history
        for m in st.session_state.call_history:
            if m["role"] == "user":
                st.markdown(f"<div class='chat-bubble user'>{escape_html(m['text'])}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble bot'>{escape_html(m['text'])}</div>", unsafe_allow_html=True)

        # JS recorder: uses SpeechRecognition, sets query param 'call_msg'
        recorder_js = """
        <div>
          <button id="startRec" style="padding:10px;border-radius:8px;background:#06f;color:#021425;border:none">🎙️ Start recording</button>
          <button id="stopRec" style="padding:10px;border-radius:8px;background:#444;color:#cfefff;border:none;margin-left:8px">⏹ Stop</button>
          <div id="rec_status" style="margin-top:8px;color:#9fb6c8"></div>
        </div>
        <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          document.getElementById('rec_status').innerText = 'Speech recognition not supported in this browser.';
          document.getElementById('startRec').disabled = true;
          document.getElementById('stopRec').disabled = true;
        } else {
          let recognition = new SpeechRecognition();
          recognition.lang = 'en-US';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
          recognition.onstart = () => { document.getElementById('rec_status').innerText = 'Listening...'; };
          recognition.onend = () => { document.getElementById('rec_status').innerText = 'Stopped.'; };
          recognition.onerror = (e) => { document.getElementById('rec_status').innerText = 'Error: ' + e.error; };
          recognition.onresult = (e) => {
            const text = Array.from(e.results).map(r => r[0].transcript).join('');
            const u = new URL(window.location);
            u.searchParams.set('call_msg', text);
            window.location = u.toString();
          };
          document.getElementById('startRec').onclick = (
