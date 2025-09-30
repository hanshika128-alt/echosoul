# app.py — EchoSoul (robust, dependency-safe single file)
"""
EchoSoul - Streamlit app
- GPT replies via `openai` package if available and OPENAI_API_KEY is set in st.secrets
- Persistent local storage: echosoul_data.json
- Vault: cryptography.Fernet if available, otherwise XOR fallback (DEMO only, not cryptographically secure)
- Call: Browser SpeechRecognition -> server receives text via URL param -> AI reply -> browser SpeechSynthesis speaks it
- Wallpaper gallery: upload & set as chat background
Designed to run on Streamlit Cloud or local Streamlit.
"""
import streamlit as st
import streamlit.components.v1 as components
import os, json, datetime, hashlib, base64, html, re, sys

st.set_page_config(page_title="EchoSoul", layout="wide")

# ---------------- Optional imports (gracefully handled) ----------------
OPENAI_INSTALLED = False
OPENAI_CLIENT = None
try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

FERNET_INSTALLED = False
try:
    from cryptography.fernet import Fernet, InvalidToken
    FERNET_INSTALLED = True
except Exception:
    FERNET_INSTALLED = False

# ---------------- Paths & data ----------------
DATA_FILE = "echosoul_data.json"

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def short_id(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        default = {
            "profile": {"name": "User", "created": now_iso()},
            "conversations": [],
            "timeline": [],
            "vault": [],
            "wallpapers": [],
            "fernet_key": None  # stored only if generated (optional)
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

def escape_html(s):
    return html.escape(s).replace("\n", "<br>")

data = load_data()

# ---------------- Vault helpers (Fernet optional, XOR fallback) ----------------
if FERNET_INSTALLED:
    # Use a stored key if present, else create one and save it.
    if not data.get("fernet_key"):
        data["fernet_key"] = Fernet.generate_key().decode()
        save_data(data)
    _fernet = Fernet(data["fernet_key"].encode())

    def vault_encrypt(password, plaintext):
        # Note: we still use single Fernet key; password is used only to gate UI access (not used to derive key).
        # For production, derive key from password with PBKDF2 + salt and store salt.
        return _fernet.encrypt(plaintext.encode()).decode()

    def vault_decrypt(password, ciphertext):
        try:
            return _fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken:
            return None
else:
    # XOR fallback (DEMO ONLY). Will show clear warning in UI.
    def _xor_bytes(b, key):
        kb = key.encode("utf-8")
        return bytes([b[i] ^ kb[i % len(kb)] for i in range(len(b))])

    def vault_encrypt(password, plaintext):
        b = plaintext.encode("utf-8")
        x = _xor_bytes(b, password or "demo")
        return base64.b64encode(x).decode()

    def vault_decrypt(password, ciphertext):
        try:
            b = base64.b64decode(ciphertext.encode())
            x = _xor_bytes(b, password or "demo")
            return x.decode("utf-8")
        except Exception:
            return None

# ---------------- OpenAI helper (if openai installed & key present) ----------------
OPENAI_ENABLED = False
if OPENAI_INSTALLED:
    openai_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
    if openai_key:
        openai.api_key = openai_key
        OPENAI_ENABLED = True

def ask_gpt(data_obj, user_text, max_tokens=256):
    """
    Build a safe prompt using profile, recent timeline and conversation and ask the model.
    If OpenAI is not available, return a deterministic offline fallback.
    """
    profile = data_obj.get("profile", {})
    persona = profile.get("persona", {"tone": "friendly"})
    name = profile.get("name", "User")

    # Build small context
    recent_timeline = data_obj.get("timeline", [])[-6:]
    recent_conv = data_obj.get("conversations", [])[-8:]
    mem_text = "\n".join([f"- {m['title']}: {m['content']}" for m in recent_timeline]) or "No memories."
    hist_text = "\n".join([f"User: {c['user']}\nEchoSoul: {c['bot']}" for c in recent_conv]) or "No recent history."

    system_prompt = (
        f"You are EchoSoul, a supportive personal AI companion for {name}.\n"
        f"Personality tone: {persona.get('tone','friendly')}.\n"
        "You may use the timeline and recent conversation for context. Be concise, honest about uncertainty, and keep replies friendly.\n"
        f"TIMELINE:\n{mem_text}\n\nRECENT_HISTORY:\n{hist_text}\n\n"
        "Respond as EchoSoul in natural, conversational language."
    )

    if not OPENAI_ENABLED:
        # deterministic offline reply for testing
        return f"(Offline demo) EchoSoul: I heard: '{user_text}'"

    try:
        # using OpenAI's Chat Completions via openai.ChatCompletion for compatibility
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        # get text safely
        if resp and "choices" in resp and len(resp["choices"]) > 0:
            return resp["choices"][0]["message"]["content"]
        return "(AI returned no content)"
    except Exception as e:
        # return sanitized error message (not leaking secrets)
        return f"(AI error) Could not generate a reply."

# ---------------- Sentiment heuristic & adaptive persona ----------------
POS = set("good great happy love awesome wonderful amazing nice calm".split())
NEG = set("sad bad angry tired upset frustrated anxious".split())

def sentiment_label(text):
    toks = re.findall(r"\w+", (text or "").lower())
    p = sum(1 for t in toks if t in POS)
    n = sum(1 for t in toks if t in NEG)
    if p > n and p > 0:
        return "energetic"
    if n > p and n > 0:
        return "empathetic"
    return "friendly"

# ---------------- UI CSS (clean, like requested) ----------------
CSS = r"""
<style>
/* overall */
.stApp { background: linear-gradient(180deg,#0b1220,#0d0d2e); color: #e9f7ff; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
/* header */
.header { font-size: 30px; font-weight: 700; text-align: center; margin: 12px 0 22px 0; color:#dff7ff; text-shadow: 0 6px 24px rgba(0,230,255,0.06); }
/* panels */
.panel { background: rgba(255,255,255,0.02); border-radius: 12px; padding: 12px; margin-bottom: 12px; }
/* chat area */
#chat_area { min-height: 48vh; max-height: 72vh; overflow-y: auto; padding: 14px; border-radius: 10px; background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent); }
/* bubbles */
.chat-bubble { padding: 12px 16px; border-radius: 14px; margin: 8px 0; max-width: 78%; line-height: 1.45; box-shadow: 0 6px 20px rgba(2,6,23,0.6); }
.user { background: linear-gradient(90deg,#7b33ff,#e84aa7); color:#fff; margin-left:auto; }
.bot { background: linear-gradient(90deg,#0b6bff,#00e5ff); color:#fff; margin-right:auto; }
/* sidebar */
.stSidebar { background: rgba(7,10,20,0.9); }
/* small */
.small { color:#9fb6c8; font-size:13px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.title("⚙️ EchoSoul Settings")
    st.subheader("Profile")
    name = st.text_input("Your name", value=data.get("profile", {}).get("name", "User"))
    if st.button("Save name"):
        data["profile"]["name"] = name or "User"
        save_data(data)
        st.success("Saved name.")
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Wallpaper (Gallery)")
    wpf = st.file_uploader("Upload wallpaper (jpg/png) to gallery", type=["jpg", "jpeg", "png"])
    if wpf:
        # save as base64 inline image (small demo)
        b64 = base64.b64encode(wpf := wpf.read()).decode()
        data.setdefault("wallpapers", []).append(f"data:image/{wpf.type.split('/')[-1]};base64,{b64}")
        save_data(data)
        st.success("Added wallpaper to gallery.")
        st.experimental_rerun()

    wallpapers = data.get("wallpapers", [])
    if wallpapers:
        sel_idx = st.selectbox("Choose wallpaper", options=list(range(len(wallpapers))), format_func=lambda i: f"Wallpaper {i+1}")
        if st.button("Set wallpaper"):
            data["profile"]["active_wallpaper"] = wallpapers[sel_idx]
            save_data(data)
            st.experimental_rerun()
        if st.button("Clear wallpaper"):
            data["profile"]["active_wallpaper"] = None
            save_data(data)
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Voice & Call (client-side)")
    st.markdown("Speech recognition runs in your browser (best on Chrome).")
    st.selectbox("Voice (browser TTS)", ["default", "Google US English", "Alloy", "Verse", "Amber"], key="voice_choice")
    st.slider("Rate", 0.6, 1.4, 1.0, key="speech_rate")
    st.slider("Pitch", 0.6, 1.4, 1.0, key="speech_pitch")

    st.markdown("---")
    st.subheader("Vault & Privacy")
    if FERNET_INSTALLED:
        st.markdown("<div class='small'>Vault encryption: cryptography (Fernet) available.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small'>Vault encryption: cryptography not installed — using demo XOR fallback (NOT SECURE). To enable secure vault, add 'cryptography' to requirements.txt.</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("OpenAI status:")
    if OPENAI_ENABLED:
        st.success("OpenAI available (GPT enabled).")
    else:
        st.warning("OpenAI not available or OPENAI_API_KEY not set. AI runs in offline/demo mode.")

# ---------------- Apply Wallpaper background if set ----------------
active_wp = data.get("profile", {}).get("active_wallpaper")
if active_wp:
    st.markdown(f"<style>.stApp{{background-image: url('{active_wp}'); background-size: cover; background-position:center;}}</style>", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(f"<div class='header'>✨ EchoSoul — Hi {escape_html(data.get('profile',{}).get('name','User'))} ✨</div>", unsafe_allow_html=True)

# ---------------- Navigation ----------------
page = st.radio("", ["Chat", "Call", "Life Timeline", "Vault", "Export", "About"], index=0, horizontal=True)

# ---------------- Chat Page ----------------
if page == "Chat":
    st.markdown("<div class='panel'><div id='chat_area'>", unsafe_allow_html=True)
    convs = data.get("conversations", [])[-200:]
    for c in convs:
        st.markdown(f"<div class='chat-bubble user'>You: {escape_html(c['user'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble bot'>EchoSoul: {escape_html(c['bot'])}</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # input
    with st.form("chat_form", clear_on_submit=False):
        user_input = st.text_area("Say something to EchoSoul (type 'act like me' to roleplay you):", height=100, key="chat_input")
        submit = st.form_submit_button("Send")
        save_as_memory = st.form_submit_button("Save as memory")

    if submit and user_input and user_input.strip():
        # persona adapt
        tone = sentiment_label(user_input)
        data.setdefault("profile", {}).setdefault("persona", {})["tone"] = tone

        reply = ask_gpt(data, user_input, max_tokens=300)
        data.setdefault("conversations", []).append({"user": user_input, "bot": reply, "ts": now_iso()})
        # occasional memory auto-add (simple heuristic)
        if len(user_input.split()) > 4:
            data.setdefault("timeline", []).append({"id": short_id(user_input + now_iso()), "title": "AutoNote", "content": user_input, "timestamp": now_iso()})
        save_data(data)
        st.experimental_rerun()

    if save_as_memory and user_input and user_input.strip():
        data.setdefault("timeline", []).append({"id": short_id(user_input + now_iso()), "title": "Memory", "content": user_input, "timestamp": now_iso()})
        save_data(data)
        st.success("Saved to timeline.")

# ---------------- Call Page (browser-based) ----------------
elif page == "Call":
    st.markdown("<div class='panel'><b>Call EchoSoul</b></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Start a live client-side recording (SpeechRecognition). When speech is recognized it is sent to the app via URL param 'call_msg'. The app will generate a reply (GPT if available) and speak it via browser TTS.</div>", unsafe_allow_html=True)

    if "in_call" not in st.session_state:
        st.session_state.in_call = False
        st.session_state.call_history = []

    col1, col2 = st.columns([1,1])
    with col1:
        if not st.session_state.in_call:
            if st.button("Start Call"):
                st.session_state.in_call = True
                st.session_state.call_history = []
                st.experimental_rerun()
        else:
            if st.button("End Call"):
                st.session_state.in_call = False
                st.experimental_rerun()
    with col2:
        if st.button("Clear Call History"):
            st.session_state.call_history = []
            st.experimental_rerun()

    if st.session_state.in_call:
        # show history
        for m in st.session_state.call_history:
            if m["role"] == "user":
                st.markdown(f"<div class='chat-bubble user'>{escape_html(m['text'])}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble bot'>{escape_html(m['text'])}</div>", unsafe_allow_html=True)

        # inject recorder JS (wrapped safely)
        recorder_js = r"""
        <div style="margin-top:8px;">
          <button id="startRec" style="padding:10px;border-radius:8px;background:#06f;color:#021425;border:none">🎙️ Start recording</button>
          <button id="stopRec" style="padding:10px;border-radius:8px;background:#444;color:#cfefff;border:none;margin-left:8px">⏹ Stop</button>
          <div id="rec_status" style="margin-top:8px;color:#9fb6c8"></div>
        </div>
        <script>
        (function(){
          const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (!SpeechRecognition) {
            document.getElementById('rec_status').innerText = 'Speech recognition not supported in this browser.';
            document.getElementById('startRec').disabled = true;
            document.getElementById('stopRec').disabled = true;
            return;
          }
          let recognition = new SpeechRecognition();
          recognition.lang = 'en-US';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
          recognition.onstart = () => { document.getElementById('rec_status').innerText = 'Listening...'; };
          recognition.onend = () => { document.getElementById('rec_status').innerText = 'Stopped.'; };
          recognition.onerror = (e) => { document.getElementById('rec_status').innerText = 'Error: ' + e.error; };
          recognition.onresult = (e) => {
            try {
              const text = Array.from(e.results).map(r => r[0].transcript).join('');
              const u = new URL(window.location);
              u.searchParams.set('call_msg', text);
              window.location = u.toString();
            } catch(err) {
              document.getElementById('rec_status').innerText = 'Result handling error';
            }
          };
          document.getElementById('startRec').onclick = () => { try { recognition.start(); } catch(e) { document.getElementById('rec_status').innerText = e.toString(); } };
          document.getElementById('stopRec').onclick = () => { recognition.stop(); };
        })();
        </script>
        """
        components.html(recorder_js, height=180)

    # If SpeechRecognition wrote call_msg to URL, process it
    params = st.experimental_get_query_params()
    if params and "call_msg" in params:
        call_text = params.get("call_msg")[0]
        if call_text:
            # append to call history
            st.session_state.call_history.append({"role": "user", "text": call_text})
            # generate reply
            reply = ask_gpt(data, call_text, max_tokens=220)
            st.session_state.call_history.append({"role": "bot", "text": reply})
            # persist to conversation memory
            data.setdefault("conversations", []).append({"user": call_text, "bot": reply, "ts": now_iso()})
            save_data(data)
            # speak reply client-side using TTS injection
            v_choice = st.session_state.get("voice_choice", "default")
            rate = float(st.session_state.get("speech_rate", 1.0))
            pitch = float(st.session_state.get("speech_pitch", 1.0))
            safe_reply = reply.replace("`", "'").replace("\n", " ")
            speak_js = f"""
            <script>
            (function(){{
              const utter = new SpeechSynthesisUtterance(`{safe_reply}`);
              utter.rate = {rate};
              utter.pitch = {pitch};
              // try to pick a voice that matches preference (best-effort)
              const pref = "{v_choice}".toLowerCase();
              const voices = window.speechSynthesis.getVoices();
              for (let i=0;i<voices.length;i++){{
                if (voices[i].name && voices[i].name.toLowerCase().includes(pref)){{ utter.voice = voices[i]; break; }}
              }}
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(utter);
              // remove call_msg from url
              const u = new URL(window.location);
              u.searchParams.delete('call_msg');
              window.history.replaceState({}, document.title, u.pathname + u.search);
            }})();</script>
            """
            components.html(speak_js, height=0)
            st.experimental_rerun()

# ---------------- Life Timeline ----------------
elif page == "Life Timeline":
    st.markdown("<div class='panel'><b>Life Timeline</b></div>", unsafe_allow_html=True)
    q = st.text_input("Search timeline (keywords)", key="tl_search")
    items = data.get("timeline", [])
    if q:
        items = [it for it in items if q.lower() in (it.get("title","") + " " + it.get("content","")).lower()]
    if items:
        for it in sorted(items, key=lambda x: x.get("timestamp",""), reverse=True):
            st.markdown(f"**{escape_html(it.get('title','Memory'))}**  <span class='small'>{it.get('timestamp')}</span>", unsafe_allow_html=True)
            st.write(it.get("content",""))
            if st.button(f"Delete {it.get('id')}", key="del_"+it.get("id","")):
                data["timeline"] = [x for x in data.get("timeline",[]) if x.get("id") != it.get("id")]
                save_data(data)
                st.experimental_rerun()
            st.markdown("---")
    else:
        st.info("No timeline entries yet. Save memories from chat or add new ones below.")

    st.markdown("### Add memory")
    title = st.text_input("Title", key="mem_title")
    content = st.text_area("Content", key="mem_content")
    if st.button("Add memory"):
        if not content.strip():
            st.warning("Content cannot be empty.")
        else:
            item = {"id": short_id(title + content + now_iso()), "title": title or "Memory", "content": content.strip(), "timestamp": now_iso()}
            data.s
