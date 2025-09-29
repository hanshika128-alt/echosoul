# app.py
"""
EchoSoul — Neon / Explainable AI companion (single-file Streamlit app)

Features implemented to match specification:
1) Conversational AI using OpenAI gpt-4o-mini (if OPENAI_API_KEY in Streamlit Secrets)
   - Uses full conversation history + selected memories as context
   - Can roleplay as the user if user asks ("act like me")
   - Exposes small explainability elements (sentiment, confidence heuristic)
2) Persistent memory stored in local file echosoul_data.json
   - Life Timeline: add/view/search/delete memories
   - Add memory from chat or timeline UI
3) Adaptive personality (sentiment-based): energetic / empathetic / friendly
4) Private Vault (prototype) with XOR demo encryption (session password)
5) Export: download JSON & legacy human-readable snapshot
6) Profile: set name in sidebar, used in replies
7) UI: Neon, dark theme similar to reference screenshot
8) Multimodal: text input, voice call (browser SpeechRecognition), voice playback (SpeechSynthesis)
9) Transparency: show sentiment, confidence heuristic, allow editing AI reply before sending/saving
10) Gallery wallpaper: upload & apply to chat container

Notes:
- This app tries to be transparent in UI: shows what the AI used as context,
  allows the user to edit generated text, and displays simple heuristics/metrics.
- Vault is NOT secure for real secrets.
"""

import streamlit as st
import streamlit.components.v1 as components
import json, os, base64, datetime, hashlib, re, html

# -------------- Config --------------
st.set_page_config(page_title="EchoSoul — Neon", layout="wide")
DATA_FILE = "echosoul_data.json"

# -------------- OpenAI client (optional) --------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except Exception:
    client = None
    OPENAI_AVAILABLE = False

# -------------- Utilities --------------
def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        default = {
            "profile": {"name": "User", "persona": {"tone": "friendly", "style":"casual"}, "created": now_iso()},
            "timeline": [],
            "vault": [],
            "conversations": [],
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

def short_id(s):
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def sanitize_js(s: str):
    return s.replace("\\","\\\\").replace("`","\\`").replace("\n","\\n").replace("\r","")

def escape_html(s: str):
    return html.escape(s).replace("\n", "<br>")

# -------------- Simple XOR (demo vault) --------------
import hashlib as _hashlib
def _derive_xor_key(password, length):
    h = _hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def xor_encrypt(password, text):
    b = text.encode("utf-8")
    key = _derive_xor_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def xor_decrypt(password, cipher_b64):
    try:
        b = base64.b64decode(cipher_b64.encode("utf-8"))
        key = _derive_xor_key(password, len(b))
        x = bytes([b[i] ^ key[i] for i in range(len(b))])
        return x.decode("utf-8")
    except Exception:
        return None

# -------------- Sentiment heuristic & confidence --------------
POS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic good".split())
NEG = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_and_confidence(text):
    toks = re.findall(r"\w+", (text or "").lower())
    pos = sum(1 for t in toks if t in POS)
    neg = sum(1 for t in toks if t in NEG)
    score = (pos - neg) / max(1, len(toks))
    # map to label
    label = "neutral"
    if score > 0.06:
        label = "positive"
    elif score < -0.06:
        label = "negative"
    # confidence heuristic: if message short, lower confidence; length and presence of sentiment words increases confidence
    conf = min(0.99, 0.3 + min(0.7, (abs(pos - neg) + len(toks)/20)))
    return label, round(float(score), 3), round(float(conf), 2)

# -------------- Compose system prompt (context) --------------
def build_system_prompt(data):
    persona = data["profile"].get("persona", {})
    tone = persona.get("tone", "friendly")
    name = data["profile"].get("name", "User")
    # include last N messages and top K memories
    conv = data.get("conversations", [])[-12:]  # recent history
    timeline_snippets = data.get("timeline", [])[-6:]
    tl_block = "\n".join([f"{m['timestamp']} | {m['title']}: {m['content']}" for m in timeline_snippets]) or "No memories."
    history_block = "\n".join([f"User: {c['user']}\nEchoSoul: {c['bot']}" for c in conv]) or "No conversation history."
    system = (
        f"You are EchoSoul, a personal companion for {name}. Personality tone: {tone}.\n"
        "Use the conversation history and timeline memories below as context when it's relevant.\n"
        "Be honest about uncertainty and avoid making up facts.\n\n"
        f"TIMELINE:\n{tl_block}\n\nRECENT HISTORY:\n{history_block}\n\n"
        "If the user asks 'act like me' or requests roleplay, respond in first-person as the user (use their name).\n"
        "Keep replies concise, helpful, and sensitive to the user's emotional state."
    )
    return system

# -------------- OpenAI call wrapper --------------
def ask_openai(data, user_text, max_tokens=512):
    system = build_system_prompt(data)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text}
    ]
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=max_tokens)
        # parse different client shapes
        c0 = resp.choices[0]
        if hasattr(c0, "message") and isinstance(c0.message, dict):
            out = c0.message.get("content", "")
        elif hasattr(c0, "message") and hasattr(c0.message, "content"):
            out = c0.message.content
        else:
            out = getattr(c0, "text", "") or str(resp)
        return str(out)
    except Exception as e:
        # don't reveal secrets — show safe message
        return f"(AI unavailable) {str(e)}" if OPENAI_AVAILABLE else f"(Offline mode) I heard: {user_text}"

# -------------- Neon CSS (UI similar to requested design) --------------
NEON_STYLE = """
<style>
:root{
  --bg1: #061220;
  --panel:#0b1722;
  --muted:#9fb6c8;
  --accent:#4ee0ff;
  --accent2:#8a6cff;
}
body, .stApp { background: radial-gradient(900px 400px at 10% 10%, rgba(10,20,40,0.5), transparent), var(--bg1); color: #dff7ff; }
/* Header */
.app-header { font-family: Inter, system-ui, sans-serif; font-size:34px; color: var(--accent); text-shadow: 0 6px 30px rgba(78,224,255,0.12); margin-bottom:6px; }
/* Sidebar */
.stSidebar { background: linear-gradient(180deg, rgba(6,14,25,0.9), rgba(10,18,30,0.95)); color:var(--muted); padding-top:10px; }
/* Panels */
.panel { background: rgba(255,255,255,0.02); border-radius:12px; padding:14px; margin-bottom:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
/* Chat */
#chat-area { min-height:60vh; max-height:75vh; overflow:auto; padding:18px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent); }
.chat-bubble { padding:12px 16px; border-radius:12px; margin:8px 0; display:inline-block; max-width:84%; line-height:1.45; }
/* user vs bot style */
.user-bubble { background: linear-gradient(90deg,#5b2a86, #bb1b67); color:#fff; box-shadow: 0 8px 40px rgba(187,27,103,0.06); }
.bot-bubble { background: linear-gradient(90deg,#023b5a,#07224a); color:#e8fbff; box-shadow: 0 10px 44px rgba(0,70,120,0.08); }
/* Controls */
.btn-neon { background: linear-gradient(90deg,var(--accent),var(--accent2)); border:none; padding:10px 14px; border-radius:10px; color:#021425; font-weight:600; box-shadow: 0 12px 42px rgba(78,224,255,0.08); }
.small { font-size:13px; color:var(--muted); }
.kv { color:#bfefff; font-weight:600; }
/* Explainable boxes */
.xbox { background: rgba(255,255,255,0.02); border-left:4px solid rgba(78,224,255,0.12); padding:8px; border-radius:8px; margin:8px 0; color:#cfeeff; font-size:13px; }
.note { font-size:12px; color:var(--muted); }
/* wallpaper overlay for chat area */
#chat-area .wallpaper-mask { border-radius:12px; padding:10px; }
</style>
"""
st.markdown(NEON_STYLE, unsafe_allow_html=True)

# -------------- Load data --------------
ensure_datafile()
data = load_data()

# -------------- Apply wallpaper to chat container --------------
def apply_chat_wallpaper():
    wp = data.get("wallpaper")
    if wp and os.path.exists(wp):
        try:
            b64 = base64.b64encode(open(wp, "rb").read()).decode()
            css = f"""
            <style>
            #chat-area {{ background-image: url('data:image/png;base64,{b64}'); background-size: cover; background-position: center; }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
        except Exception:
            pass

apply_chat_wallpaper()

# -------------- Sidebar (controls + profile + upload) --------------
with st.sidebar:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='kv'>EchoSoul</div>", unsafe_allow_html=True)
    st.text_input("Your name", value=data["profile"].get("name", "User"), key="sidebar_name")
    if st.button("Save profile"):
        data["profile"]["name"] = st.session_state.sidebar_name.strip() or "User"
        save_data(data)
        st.success("Profile saved.")

    st.markdown("---")
    st.markdown("<div class='kv'>Voice / Call</div>", unsafe_allow_html=True)
    st.selectbox("Browser voice (attempt; actual available voices depend on device)", ["default", "Google US English", "Alloy", "Verse", "Amber"], key="voice_choice")
    st.slider("AI speech rate", 0.6, 1.6, 1.0, key="speech_rate")
    st.slider("AI speech pitch", 0.7, 1.5, 1.0, key="speech_pitch")
    st.file_uploader("Upload short voice sample (mp3/wav) — demo only (stored local)", type=["mp3","wav"], key="voice_sample_u")
    if st.button("Save voice sample"):
        uf = st.session_state.get("voice_sample_u")
        if uf:
            fname = f"voice_sample_{short_id(uf.name)}.{uf.name.split('.')[-1]}"
            with open(fname, "wb") as f:
                f.write(uf.getbuffer())
            data["voice_sample"] = fname
            save_data(data)
            st.success("Voice sample saved (local).")
        else:
            st.warning("Choose a file first.")

    st.markdown("---")
    st.markdown("<div class='kv'>Wallpaper (Gallery)</div>", unsafe_allow_html=True)
    up = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg","jpeg","png"], key="wp_file")
    if st.button("Upload & Add to gallery"):
        uf = st.session_state.get("wp_file")
        if uf:
            fname = f"wall_{short_id(uf.name)}.{uf.name.split('.')[-1]}"
            with open(fname, "wb") as f:
                f.write(uf.getbuffer())
            st.success("Added to gallery.")
            st.experimental_rerun()
        else:
            st.warning("Select an image.")

    # list gallery images in repo root
    imgs = [f for f in os.listdir(".") if f.lower().endswith((".jpg",".jpeg",".png"))]
    imgs_sorted = ["(none)"] + sorted(imgs)
    choice = st.selectbox("Choose from gallery", imgs_sorted, key="gallery_choice")
    if st.button("Set selected as chat wallpaper"):
        if choice != "(none)":
            data["wallpaper"] = choice
            save_data(data)
            st.success("Wallpaper applied.")
            st.experimental_rerun()
    if st.button("Clear wallpaper"):
        data["wallpaper"] = None
        save_data(data)
        st.success("Wallpaper cleared.")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("<div class='kv'>Privacy & Export</div>", unsafe_allow_html=True)
    st.checkbox("Enable adaptive learning (local)", value=True, key="adaptive_toggle")
    if st.button("Download export (JSON)"):
        st.download_button("Download JSON", json.dumps(data, indent=2), "echosoul_export.json", "application/json")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------- Main header + nav --------------
st.markdown(f"<div class='app-header'>✨ EchoSoul — {escape_html(data['profile'].get('name','User'))}</div>", unsafe_allow_html=True)
page = st.radio("", ["Chat","Call","Life Timeline","Vault","Export","About"], index=0, horizontal=True)

# -------------- Chat Page --------------
if page == "Chat":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Explainable AI: the boxes below show what context EchoSoul used and simple heuristics for its reply.</div>", unsafe_allow_html=True)

    # show context/what the AI will see
    st.markdown("<div class='xbox'><b>Context (used for replies):</b><br>", unsafe_allow_html=True)
    mems = data.get("timeline", [])[-6:]
    st.markdown("<div class='note'>Recent timeline memories (used as context):</div>", unsafe_allow_html=True)
    if mems:
        for m in mems:
            st.markdown(f"<div class='note'>• {escape_html(m['title'])} — {escape_html(m['content'])} <span style='font-size:11px;color:#9fb6c8'>({m['timestamp']})</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='note'>(no memories yet)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # conversation area
    st.markdown("<div id='chat-area' class='panel'>", unsafe_allow_html=True)
    # render conversation
    for conv in data.get("conversations", [])[-80:]:
        st.markdown(f"<div class='chat-bubble user-bubble'><b>You:</b> {escape_html(conv['user'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble bot-bubble'><b>EchoSoul:</b> {escape_html(conv['bot'])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # input area with options
    with st.form("chat_input_form", clear_on_submit=False):
        user_text = st.text_area("Say something to EchoSoul (press Send). You can also ask EchoSoul to 'act like me'.", height=80, key="chat_input")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            send = st.form_submit_button("Send", use_container_width=True)
        with col2:
            save_mem = st.form_submit_button("Save to timeline")
        with col3:
            gen_reply = st.form_submit_button("Generate suggested reply (editable)")

    # generate suggested reply (editable) and XAI info
    if gen_reply and user_text.strip():
        # create suggested reply via OpenAI if available, else simple fallback
        if OPENAI_AVAILABLE:
            suggestion = ask_openai(data, user_text)
        else:
            suggestion = f"(Offline) EchoSoul suggests: {user_text}"
        st.markdown("<div class='xbox'><b>Suggested reply (edit before sending):</b></div>", unsafe_allow_html=True)
        edited = st.text_area("Edit suggested reply", value=suggestion, key="edited_reply")
        # show heuristics
        label, score, conf = sentiment_and_confidence(user_text)
        st.markdown(f"<div class='note'>Sentiment: <b>{label}</b> (score={score}) • Confidence heuristic: <b>{conf}</b></div>", unsafe_allow_html=True)
        if st.button("Send edited reply"):
            # store conversation: user->edited reply
            data.setdefault("conversations", []).append({"user": user_text, "bot": edited, "ts": now_iso()})
            save_data(data)
            st.success("Sent and saved to conversation.")
            st.experimental_rerun()

    # direct send (without suggestion)
    if send and user_text.strip():
        # update persona heuristics
        label, score, conf = sentiment_and_confidence(user_text)
        # adapt tone
        if score < -0.06:
            data["profile"]["persona"]["tone"] = "empathetic"
        elif score > 0.06:
            data["profile"]["persona"]["tone"] = "energetic"
        else:
            data["profile"]["persona"]["tone"] = "friendly"
        # get bot reply
        if OPENAI_AVAILABLE:
            bot_reply = ask_openai(data, user_text)
        else:
            bot_reply = f"(Offline) EchoSoul heard: {user_text}"
        data.setdefault("conversations", []).append({"user": user_text, "bot": bot_reply, "ts": now_iso()})
        save_data(data)
        st.experimental_rerun()

    if save_mem and user_text.strip():
        # save to timeline
        item = {"id": short_id(user_text + now_iso()), "title": "Note", "content": user_text, "timestamp": now_iso()}
        data.setdefault("timeline", []).append(item)
        save_data(data)
        st.success("Saved to timeline.")
        st.experimental_rerun()

# -------------- Call Page --------------
elif page == "Call":
    st.markdown("<div class='panel'><b>Call EchoSoul — speak and EchoSoul will reply using browser voice.</b></div>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Reminder: Grant microphone permission. Works best in Chrome/Edge on Android.</div>", unsafe_allow_html=True)

    # call session controls
    if "call_active" not in st.session_state:
        st.session_state.call_active = False
        st.session_state.call_history = []

    c1, c2 = st.columns([1,1])
    with c1:
        if not st.session_state.call_active:
            if st.button("Start Call", key="start_call"):
                st.session_state.call_active = True
                st.session_state.call_history = []
                st.experimental_rerun()
        else:
            if st.button("End Call", key="end_call"):
                st.session_state.call_active = False
                st.experimental_rerun()
    with c2:
        if st.button("Clear call history"):
            st.session_state.call_history = []
            st.experimental_rerun()

    if st.session_state.call_active:
        st.markdown("<div class='xbox'><b>Live call in progress</b></div>", unsafe_allow_html=True)
        # render call history
        for m in st.session_state.call_history:
            if m["role"] == "user":
                st.markdown(f"<div class='chat-bubble user-bubble'><b>You:</b> {escape_html(m['text'])}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble bot-bubble'><b>EchoSoul:</b> {escape_html(m['text'])}</div>", unsafe_allow_html=True)

        # recorder widget (JS SpeechRecognition) -> sets ?call_msg= recognized text
        recorder = """
        <div>
          <button id="startRec" style="padding:10px;border-radius:8px;background:#06f;color:#021425;border:none">🎙️ Start recording</button>
          <button id="stopRec" style="padding:10px;border-radius:8px;background:#444;color:#cfefff;border:none;margin-left:8px">⏹ Stop</button>
          <div id="rec_status" style="margin-top:8px;color:#9fb6c8"></div>
        </div>
        <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          document.getElementById('rec_status').innerText = 'Speech recognition not supported on this browser.';
          document.getElementById('startRec').disabled = true;
          document.getElementById('stopRec').disabled = true;
        } else {
          let recognition = new SpeechRecognition();
          recognition.lang = 'en-US';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
  
