# app.py
"""
EchoSoul — Corrected and cleaned single-file Streamlit app
All previous occurrences of `st.experimental_rerun()` replaced with `st.rerun()`.
Wired features:
- Chat (OpenAI if secret provided, otherwise fallback)
- Simulated Voice Call (browser TTS via SpeechSynthesis)
- Life Timeline (add/search/delete)
- Vault (demo XOR encryption)
- Wallpaper upload & apply
- Export (download JSON)
Storage: echosoul_data.json (in app folder)
IMPORTANT: Add OPENAI_API_KEY in Streamlit Secrets if you want real AI replies.
"""

import streamlit as st
import streamlit.components.v1 as components
import os, json, hashlib, base64, datetime, re

# ----------------- Config -----------------
st.set_page_config(page_title="EchoSoul Chat", layout="wide")
DATA_FILE = "echosoul_data.json"
WALLPAPER_PREFIX = "echosoul_wallpaper"

# ----------------- OpenAI client init (from Streamlit secrets) -----------------
try:
    # Use new OpenAI client if available
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except Exception:
    client = None
    OPENAI_AVAILABLE = False

# ----------------- Utility helpers -----------------
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

# ----------------- Simple XOR 'encryption' (demo) -----------------
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
        data = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data))
        x = bytes([data[i] ^ key[i] for i in range(len(data))])
        return x.decode("utf-8")
    except Exception:
        return None

# ----------------- Sentiment heuristic & persona -----------------
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    norm = score / max(1, len(toks))
    return norm

def sentiment_label(score):
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"

def update_persona_based_on_sentiment(data, score):
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# ----------------- Memory helpers -----------------
def add_memory(data, title, content):
    item = {"id": hashlib.sha1((title+content+ts_now()).encode("utf-8")).hexdigest(),
            "title": title or "Memory", "content": content, "timestamp": ts_now()}
    data["timeline"].append(item)
    save_data(data)
    return item

def find_relevant_memories(data, text, limit=3):
    found = []
    txt = text.lower()
    for item in reversed(data["timeline"]):
        if (any(w in txt for w in re.findall(r"\w+", item["content"].lower()))
            or any(w in txt for w in re.findall(r"\w+", item["title"].lower()))):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# ----------------- HTML/JS helpers -----------------
def html_escape_js(s: str):
    # minimal escape for JS template literal
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$").replace("\n", "\\n").replace("\r","")

def browser_speak_js():
    return """
    <script>
    function speakText(text, voice=null, rate=1, pitch=1) {
      if (!("speechSynthesis" in window)) {
        return "no-speech";
      }
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = rate;
      utter.pitch = pitch;
      if (voice) {
        const v = window.speechSynthesis.getVoices().find(x => x.name===voice);
        if (v) utter.voice = v;
      }
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
      return "ok";
    }
    </script>
    """

# ----------------- OpenAI backed reply generator -----------------
def generate_reply_openai(data, user_msg):
    # sentiment + persona update
    s = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, s)
    tone = data["profile"]["persona"].get("tone", "friendly")

    # memory context (last 6)
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-6:]]
    mem_block = "\n".join(memories) if memories else "No memories available."

    system_prompt = f"""You are EchoSoul, a helpful, empathetic digital companion.
Personality tone: {tone}.
Use the user's timeline (below) as context when replying but keep responses concise and conversational.
Timeline:
{mem_block}

If the user explicitly asks "act like me" or "be me", roleplay as the user (use their name if provided) and answer in first person as them.
Always avoid inventing facts not supported by the timeline.
"""

    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": user_msg}
    ]

    try:
        model = "gpt-4o-mini"
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512)
        # handle response shape carefully
        c0 = resp.choices[0]
        # depending on client format
        if hasattr(c0, "message") and isinstance(c0.message, dict):
            reply = c0.message.get("content", "")
        elif hasattr(c0, "message") and hasattr(c0.message, "content"):
            reply = c0.message.content
        else:
            # fallback parsing
            reply = getattr(c0, "text", "") or str(resp)
    except Exception as e:
        reply = f"(AI unavailable) I heard: {user_msg}"
    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ----------------- Wallpaper apply -----------------
def apply_wallpaper_css(data):
    if data.get("wallpaper"):
        path = data["wallpaper"]
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
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
        except Exception:
            pass

# ----------------- Load and prepare data -----------------
data = load_data()
apply_wallpaper_css(data)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("EchoSoul Settings")
    name = st.text_input("Your name", value=data["profile"].get("name","User"))
    if st.button("Save name"):
        data["profile"]["name"] = name.strip() or "User"
        save_data(data)
        st.success("Name saved.")
        st.rerun()

    st.markdown("---")
    st.header("Voice Settings")
    st.write("Choose a voice (browser TTS will use available voices):")
    st.radio("Voice", ["alloy","verse","amber"], index=0, key="voice_choice")
    st.write("Upload short voice sample (optional)")
    uploaded_voice = st.file_uploader("MP3/WAV sample (stored locally)", type=["mp3","wav"])
    if uploaded_voice is not None:
        fpath = f"voice_sample_{hashlib.md5(uploaded_voice.name.encode()).hexdigest()}.bin"
        with open(fpath, "wb") as f:
            f.write(uploaded_voice.getbuffer())
        st.success("Voice sample uploaded.")

    st.markdown("---")
    st.header("Background Wallpaper")
    wallpaper = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg","jpeg","png"])
    if st.button("Set wallpaper"):
        if wallpaper is not None:
            ext = wallpaper.name.split(".")[-1]
            fname = f"{WALLPAPER_PREFIX}.{ext}"
            with open(fname, "wb") as out:
                out.write(wallpaper.getbuffer())
            data["wallpaper"] = fname
            save_data(data)
            st.success("Wallpaper saved.")
            st.rerun()
        else:
            st.warning("Choose an image first.")

    if st.button("Clear wallpaper"):
        if data.get("wallpaper") and os.path.exists(data["wallpaper"]):
            try:
                os.remove(data["wallpaper"])
            except Exception:
                pass
        data["wallpaper"] = None
        save_data(data)
        st.rerun()

    st.markdown("---")
    st.header("Privacy")
    st.checkbox("Enable adaptive learning", value=True, key="adaptive_toggle")
    st.write("Data stored inside app instance (Streamlit Cloud). Vault uses demo encryption; avoid real secrets.")

# ----------------- Main UI / Navigation -----------------
st.markdown("<h1 style='display:inline-block'>💬 EchoSoul Chat</h1>", unsafe_allow_html=True)
nav = st.radio("", ["Home","Chat","Voice Call","Life Timeline","Vault","Export","About"], index=1, horizontal=True)

# HOME
if nav == "Home":
    st.subheader("Welcome to EchoSoul")
    st.write("Use navigation to chat, call, add memories, and keep a private vault.")
    st.write("Persona tone:", data["profile"]["persona"].get("tone","friendly"))
    st.write("Memories stored:", len(data["timeline"]))
    if data.get("wallpaper"):
        st.image(data["wallpaper"], caption="Current wallpaper", use_column_width=True)

# CHAT
if nav == "Chat":
    st.subheader("Chat with EchoSoul")
    convs = data.get("conversations", [])
    if convs:
        for c in convs[-50:]:
            st.markdown(f"**You:** {c['user']}")
            st.markdown(f"**EchoSoul:** {c['bot']}")
            st.markdown("---")

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Say something to EchoSoul...", key="chat_input")
        col1, col2 = st.columns([1,1])
        with col1:
            send = st.form_submit_button("Send")
        with col2:
            save_mem = st.form_submit_button("Save as memory")

    if send:
        if user_msg.strip() == "":
            st.warning("Please type something.")
        else:
            if OPENAI_AVAILABLE:
                reply = generate_reply_openai(data, user_msg)
            else:
                reply = f"I heard you: {user_msg}"
                data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
                save_data(data)
            st.rerun()

    if save_mem:
        if user_msg.strip() == "":
            st.warning("Cannot save empty memory.")
        else:
            add_memory(data, "User note", user_msg.strip())
            st.success("Saved to timeline.")
            st.rerun()

# VOICE CALL (simulated)
if nav == "Voice Call":
    st.subheader("Voice Call (simulated) — EchoSoul speaks using your browser's TTS")
    st.write("Start a short call: send messages, EchoSoul will reply and you can press Speak to hear the reply.")

    if "in_call" not in st.session_state:
        st.session_state["in_call"] = False

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Start Call"):
            st.session_state["in_call"] = True
            st.session_state["call_history"] = []
            st.session_state["call_started_at"] = ts_now()
            st.rerun()
    with c2:
        if st.button("End Call"):
            st.session_state["in_call"] = False
            st.session_state.pop("call_history", None)
            st.rerun()

    if st.session_state.get("in_call"):
        st.info("Call in progress — type below and send to talk. Use Speak reply to hear audio.")
        call_hist = st.session_state.get("call_history", [])
        for msg in call_hist:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**EchoSoul:** {msg['text']}")
        new_msg = st.text_input("Say to EchoSoul (call)", key="call_input")
        if st.button("Send in call"):
            if new_msg.strip():
                call_hist.append({"role":"user","text":new_msg})
                st.session_state["call_history"] = call_hist
                if OPENAI_AVAILABLE:
                    reply = generate_reply_openai(data, new_msg)
                else:
                    reply = "EchoSoul (offline) heard: " + new_msg
                call_hist.append({"role":"bot","text":reply})
                st.session_state["call_history"] = call_hist
                st.rerun()
        if call_hist and call_hist[-1]["role"] == "bot":
            last_text = call_hist[-1]["text"]
            st.markdown("**Last reply (tap Speak to play through your device):**")
            st.write(last_text)
            # inject JS speak function
            components.html(browser_speak_js(), height=0)
            speak_button_html = f"""
            <button onclick='speakText(`{html_escape_js(last_text)}`)'>Speak reply</button>
            """
            components.html(speak_button_html, height=70)

# LIFE TIMELINE
if nav == "Life Timeline":
    st.subheader("Life Timeline — add, search, view memories")
    col1, col2 = st.columns([2,1])
    with col1:
        q = st.text_input("Search timeline (keyword)")
        results = []
        if q.strip() == "":
            results = sorted(data["timeline"], key=lambda x:x["timestamp"], reverse=True)
        else:
            for item in data["timeline"]:
                if q.lower() in (item["title"]+item["content"]).lower():
                    results.append(item)
            results = sorted(results, key=lambda x:x["timestamp"], reverse=True)
        if results:
            for item in results:
                st.markdown(f"**{item['title']}**  — {item['timestamp']}")
                st.write(item["content"])
                if st.button("Delete", key="del_"+item["id"]):
                    data["timeline"] = [it for it in data["timeline"] if it["id"] != item["id"]]
                    save_data(data)
                    st.success("Deleted.")
                    st.rerun()
                st.markdown("---")
        else:
            st.info("No memories found — add one below.")
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

# VAULT
if nav == "Vault":
    st.subheader("Private Vault")
    st.write("Store encrypted notes here (demo XOR). Provide a password to view / save items.")
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
                st.write("*Provide vault password to view.*")
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

# EXPORT
if nav == "Export":
    st.subheader("Legacy & Export")
    st.write("Download your EchoSoul data (timeline, vault entries remain encrypted).")
    dump = json.dumps(data, indent=2)
    st.download_button("Download full export (JSON)", dump, f"echosoul_export_{datetime.datetime.utcnow().date()}.json", "application/json")
    st.markdown("---")
    legacy = "\n\n".join([f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data['timeline']])
    st.text_area("Legacy snapshot (human readable)", legacy, height=300)

# ABOUT
if nav == "About":
    st.subheader("About EchoSoul (Corrected)")
    st.write("""
EchoSoul is a prototype personal companion:
- Chat backed by OpenAI (if API key provided)
- Simulated voice call using browser TTS
- Life timeline, private vault, wallpaper, export
""")
    st.write("If you want more features (real TTS, better encryption, audio recording), tell me and I will add them.")

# ----------------- End: ensure data saved -----------------
save_data(data)
