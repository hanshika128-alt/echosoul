# app.py - EchoSoul (complete, corrected)
# 2025-10-02
# Single-file Streamlit app. Save as app.py and deploy to Streamlit Cloud.
# Recommended requirements (optional features): openai, cryptography, gTTS

import streamlit as st
from datetime import datetime
from pathlib import Path
import json
import io
import os
import uuid
import base64
import typing

# Optional dependencies
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

# ---------------- App config ----------------
st.set_page_config(page_title="EchoSoul", layout="wide", initial_sidebar_state="expanded")
st.title("EchoSoul — Personal Reflective AI")

# ---------------- Data paths ----------------
DATA_DIR = Path("data")
VOICE_DIR = DATA_DIR / "voices"
DATA_DIR.mkdir(exist_ok=True)
VOICE_DIR.mkdir(exist_ok=True)

MEMORY_FILE = DATA_DIR / "memories.json"
CHAT_FILE = DATA_DIR / "chats.json"
TIMELINE_FILE = DATA_DIR / "timeline.json"
VAULT_FILE = DATA_DIR / "vault.bin"
VAULT_META = DATA_DIR / "vault_meta.json"

# ---------------- Persistence helpers ----------------
def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

memories = load_json(MEMORY_FILE, [])
chats = load_json(CHAT_FILE, [])
timeline = load_json(TIMELINE_FILE, [])

# ---------------- Session defaults ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # live display: list of {"role","text","ts"}
if "vault_unlocked" not in st.session_state:
    st.session_state.vault_unlocked = False
if "vault_key" not in st.session_state:
    st.session_state.vault_key = None
if "vault_contents" not in st.session_state:
    st.session_state.vault_contents = {}
if "call_active" not in st.session_state:
    st.session_state.call_active = False
if "custom_voices" not in st.session_state:
    # list of {"name","path"}
    st.session_state.custom_voices = []
# load existing voice files from folder
for p in VOICE_DIR.glob("*"):
    # avoid duplicates
    entry = {"name": p.name, "path": str(p)}
    if entry not in st.session_state.custom_voices:
        st.session_state.custom_voices.append(entry)
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "EchoSoul (Default)"
if "api_pin" not in st.session_state:
    st.session_state.api_pin = ""

# ---------------- Vault helpers ----------------
def _derive_key(password: str, salt: bytes):
    if not HAS_CRYPTO:
        raise RuntimeError("cryptography not installed")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

def create_vault(password: str):
    if not HAS_CRYPTO:
        st.error("Vault requires cryptography library.")
        return False
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    initial = {"notes": [], "created": datetime.utcnow().isoformat()}
    token = f.encrypt(json.dumps(initial).encode("utf-8"))
    VAULT_FILE.write_bytes(token)
    save_json(VAULT_META, {"salt": base64.b64encode(salt).decode()})
    st.session_state.vault_unlocked = True
    st.session_state.vault_key = key
    st.session_state.vault_contents = initial
    return True

def unlock_vault(password: str) -> bool:
    if not HAS_CRYPTO:
        st.error("Vault requires cryptography library.")
        return False
    if not VAULT_FILE.exists() or not VAULT_META.exists():
        st.warning("Vault not found.")
        return False
    meta = load_json(VAULT_META, {})
    try:
        salt = base64.b64decode(meta["salt"])
    except Exception:
        st.error("Vault metadata corrupted.")
        return False
    key = _derive_key(password, salt)
    f = Fernet(key)
    try:
        data = f.decrypt(VAULT_FILE.read_bytes())
        st.session_state.vault_unlocked = True
        st.session_state.vault_key = key
        st.session_state.vault_contents = json.loads(data.decode("utf-8"))
        return True
    except InvalidToken:
        return False

def save_vault():
    if not (HAS_CRYPTO and st.session_state.vault_unlocked and st.session_state.vault_key):
        st.error("Vault not ready to save.")
        return False
    f = Fernet(st.session_state.vault_key)
    token = f.encrypt(json.dumps(st.session_state.vault_contents).encode("utf-8"))
    VAULT_FILE.write_bytes(token)
    return True

# ---------------- Emotion detection (simple) ----------------
EMO_DICT = {
    "happy": ["happy","joy","glad","excited","great","yay"],
    "sad": ["sad","down","depressed","lonely","miserable"],
    "angry": ["angry","mad","furious","annoyed","hate"],
    "anxious": ["anxious","anxiety","nervous","worried","panic"],
    "tired": ["tired","sleepy","exhausted","drained"]
}
def detect_emotion(text: str) -> str:
    t = (text or "").lower()
    scores = {k:0 for k in EMO_DICT}
    for k, kws in EMO_DICT.items():
        for kw in kws:
            if kw in t:
                scores[k] += 1
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "neutral"

# ---------------- AI chat helpers ----------------
def build_system_prompt():
    snippet = ""
    if memories:
        snippet = "Known details: " + "; ".join(f"{m.get('k')}: {m.get('v')}" for m in memories[:8])
    return ("You are EchoSoul, an empathetic AI companion. Adapt tone to user's mood and mimic style "
            "when asked. Respect privacy and be caring. " + snippet)

def local_reply(user_text: str, mimic=False, style_examples: typing.List[str]=None) -> str:
    emo = detect_emotion(user_text)
    if mimic and style_examples:
        sample = style_examples[-1] if style_examples else ""
        if len(sample.split()) < 5:
            return f"{sample}... {user_text}"
        return f"{sample.split()[0].capitalize()}, {user_text}"
    if emo == "happy":
        return "That's wonderful to hear — tell me more."
    if emo == "sad":
        return "I'm sorry you're feeling that. Do you want to share more?"
    if emo == "tired":
        return "You sound tired — remember to rest. Want a short breathing exercise?"
    return f"I hear you: \"{user_text}\" — how should I support you?"

def chat_with_model(user_text: str, mimic=False, style_examples: typing.List[str]=None, api_key: str=None) -> str:
    # If OpenAI available and key exists, call; otherwise local fallback
    key = api_key or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
    if HAS_OPENAI and key:
        try:
            openai.api_key = key
            messages = [{"role":"system","content": build_system_prompt()}]
            if mimic and style_examples:
                messages.append({"role":"system", "content":"Mimic the user's style using examples:\n" + "\n".join(style_examples[-10:])})
            messages.append({"role":"user","content": user_text})
            # choose a reasonable model string; avoid calling Model.list()
            model_name = "gpt-4o" if "gpt-4o" in getattr(openai, "model", {}) else "gpt-4o"
            resp = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0.8, max_tokens=400)
            return resp.choices[0].message.content
        except Exception as e:
            return f"[API error: {e}] " + local_reply(user_text, mimic, style_examples)
    else:
        return local_reply(user_text, mimic, style_examples)

# ---------------- TTS helpers ----------------
def tts_audio_from_text(text: str, voice_name: str="default") -> typing.Optional[io.BytesIO]:
    # Try OpenAI TTS (if available), else gTTS fallback, else None
    key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else st.session_state.api_pin or None
    if HAS_OPENAI and key:
        try:
            openai.api_key = key
            # guarded attempt - SDK may differ, but try commonly used pattern
            audio_resp = openai.Audio.speech.create(model="gpt-4o-mini-tts", voice=voice_name, input=text)
            bio = io.BytesIO(audio_resp.read())
            bio.seek(0)
            return bio
        except Exception:
            pass
    if HAS_GTTS:
        try:
            tts = gTTS(text)
            bio = io.BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            return bio
        except Exception:
            pass
    return None

# ---------------- Voice upload ----------------
def save_voice_file(uploaded, name=None):
    if not uploaded:
        return None
    name = name or uploaded.name
    uid = uuid.uuid4().hex
    dest = VOICE_DIR / f"{uid}_{name}"
    data = uploaded.read()
    dest.write_bytes(data)
    entry = {"name": name, "path": str(dest)}
    st.session_state.custom_voices.append(entry)
    return entry

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.header("EchoSoul — Menu")
    page = st.radio("", ["Chat", "Chat History", "Life Timeline", "Vault", "Export", "Brain Mimic", "About"], index=0)
    st.markdown("---")
    st.subheader("Call & Voice")
    if st.button("Start Call (simulate)"):
        st.session_state.call_active = True
    if st.button("End Call"):
        st.session_state.call_active = False

    builtin = ["EchoSoul (Default)", "Warm", "Calm", "Bright"]
    custom_names = [v["name"] for v in st.session_state.custom_voices]
    voice_options = builtin + custom_names
    st.selectbox("Choose voice", options=voice_options, key="selected_voice")

    uploaded_voice = st.file_uploader("Upload voice (.mp3 / .wav) to use in calls", type=["mp3","wav"])
    if uploaded_voice is not None:
        try:
            ent = save_voice_file(uploaded_voice, uploaded_voice.name)
            if ent:
                st.success(f"Saved voice: {ent['name']}")
        except Exception as e:
            st.error(f"Failed to save voice: {e}")

    st.text_input("API PIN (optional)", key="api_pin", placeholder="Optional: OpenAI key or pin")
    st.markdown("---")
    if st.button("New Conversation"):
        st.session_state.messages = []
    st.caption("Left: Chat, History, Timeline, Vault, Export, Brain Mimic, About")

# ---------------- Call screen UI ----------------
def render_call_screen():
    st.markdown("---")
    st.subheader("Call (simulated)")
    c1, c2, c3 = st.columns([1,4,1])
    with c2:
        st.image("https://via.placeholder.com/180.png?text=EchoSoul", width=140)
        st.markdown("<div style='text-align:center'>Calling... EchoSoul</div>", unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            if st.button("Mute"):
                st.info("Muted (simulated).")
        with cc2:
            if st.button("End Call (screen)"):
                st.session_state.call_active = False
        with cc3:
            if st.button("Speaker"):
                st.info("Speaker toggled (simulated).")

    # Play the last AI message as call audio (either uploaded voice or TTS)
    last_ai = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "ai":
            last_ai = m
            break
    if last_ai:
        st.markdown(f"**EchoSoul says:** {last_ai['text']}")
        if st.button("Play as call audio"):
            # If user selected a custom uploaded voice file, play it. Otherwise generate TTS.
            sel = st.session_state.selected_voice
            custom = next((c for c in st.session_state.custom_voices if c["name"] == sel), None)
            if custom:
                try:
                    st.audio(custom["path"])
                except Exception as e:
                    st.warning(f"Failed to play custom voice: {e}")
            else:
                bio = tts_audio_from_text(last_ai["text"], voice_name=sel)
                if bio:
                    try:
                        st.audio(bio.read(), format="audio/mp3")
                    except Exception:
                        bio.seek(0)
                        st.audio(bio)
                else:
                    st.warning("No TTS available (no OpenAI key and gTTS not installed).")

# ---------------- Pages ----------------
def chat_page():
    st.header("Chat — text & voice")
    # small call icon
    header_col = st.columns([9,1])[1]
    if header_col.button("📞"):
        st.session_state.call_active = True

    # Load persisted chats into session messages if empty
    if not st.session_state.messages and chats:
        for c in chats[-200:]:
            role = "user" if c.get("role") == "user" else "ai"
            st.session_state.messages.append({"role": role, "text": c.get("text",""), "ts": c.get("ts","")})

    # Chat render area
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div style='text-align:right;background:#c6f6d5;padding:10px;border-radius:12px;margin:6px'>{msg['text']}<div style='font-size:10px;color:#444'>{msg.get('ts','')}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left;background:#1f2937;color:white;padding:10px;border-radius:12px;margin:6px'>{msg['text']}<div style='font-size:10px;color:#ddd'>{msg.get('ts','')}</div></div>", unsafe_allow_html=True)

    # Bottom texting bar (st.chat_input)
    user_text = st.chat_input("Message...")
    if user_text:
        ts = datetime.utcnow().isoformat()
        st.session_state.messages.append({"role":"user","text":user_text,"ts":ts})
        chats.append({"role":"user","text":user_text,"ts":ts})
        # memory heuristic
        low = user_text.lower()
        if low.startswith("i am ") or "my name is " in low or low.startswith("i'm "):
            memories.insert(0, {"k":"self_statement","v":user_text,"ts":ts})
            save_json(MEMORY_FILE, memories)
        # AI reply
        style_examples = [m['text'] for m in chats if m.get("role")=="user"]
        ai_reply = chat_with_model(user_text, mimic=False, style_examples=style_examples, api_key=(st.session_state.api_pin or None))
        rts = datetime.utcnow().isoformat()
        st.session_state.messages.append({"role":"ai","text":ai_reply,"ts":rts})
        chats.append({"role":"ai","text":ai_reply,"ts":rts})
        timeline.append({"type":"chat","text":user_text,"response":ai_reply,"ts":ts})
        save_json(CHAT_FILE, chats)
        save_json(TIMELINE_FILE, timeline)
        # st.chat_input auto-clears; rerun to refresh UI immediately
        st.experimental_rerun()

    # quick controls below chat
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Play last AI (TTS)"):
            last_ai = next((m for m in reversed(st.session_state.messages) if m["role"]=="ai"), None)
            if last_ai:
                # if custom voice selected, play file
                sel = st.session_state.selected_voice
                custom = next((c for c in st.session_state.custom_voices if c["name"]==sel), None)
                if custom:
                    st.audio(custom["path"])
                else:
                    bio = tts_audio_from_text(last_ai["text"], voice_name=sel)
                    if bio:
                        try:
                            st.audio(bio.read(), format="audio/mp3")
                        except Exception:
                            bio.seek(0)
                            st.audio(bio)
                    else:
                        st.warning("No TTS available.")
            else:
                st.info("No AI message yet.")
    with c2:
        if st.button("Save conversation (persist)"):
            save_json(CHAT_FILE, chats)
            st.success("Saved.")

def history_page():
    st.header("Chat History")
    st.write("Recent persisted messages:")
    for m in reversed(chats[-500:]):
        who = "You" if m.get("role")=="user" else "EchoSoul"
        st.markdown(f"**{who}** ({m.get('ts','')}): {m.get('text','')}")

def timeline_page():
    st.header("Life Timeline")
    st.write("Chronological events, transcripts, and important moments.")
    with st.expander("Add event"):
        ev_text = st.text_input("Event text", key="ev_text")
        ev_tags = st.text_input("Tags (comma separated)", key="ev_tags")
        if st.button("Add event"):
            ev = {"id":str(uuid.uuid4()), "text":ev_text, "tags":[t.strip() for t in ev_tags.split(",") if t.strip()], "ts":datetime.utcnow().isoformat(), "type":"manual"}
            timeline.append(ev)
            save_json(TIMELINE_FILE, timeline)
            st.success("Added.")
            st.experimental_rerun()
    for ev in sorted(timeline, key=lambda x:x.get("ts",""), reverse=True):
        st.markdown(f"**{ev.get('ts')}** — {ev.get('text')}  \nTags: {', '.join(ev.get('tags',[]))}")

def vault_page():
    st.header("Memory Vault")
    if not HAS_CRYPTO:
        st.warning("Vault functions require the `cryptography` package. Vault disabled.")
        return
    if not VAULT_FILE.exists():
        st.info("No vault found. Create one to store encrypted memories.")
        pwd = st.text_input("Create vault password", type="password", key="vault_create_pwd")
        if st.button("Create Vault"):
            try:
                create_vault(pwd)
                st.success("Vault created and unlocked.")
            except Exception as e:
                st.error(f"Failed creating vault: {e}")
    else:
        if not st.session_state.vault_unlocked:
            pwd = st.text_input("Enter vault password", type="password", key="vault_unlock_pwd")
            if st.button("Unlock Vault"):
                ok = unlock_vault(pwd)
                if ok:
                    st.success("Vault unlocked.")
                else:
                    st.error("Wrong password.")
        else:
            st.success("Vault unlocked.")
            st.json(st.session_state.vault_contents)
            note = st.text_area("Write secure note", key="vault_note")
            if st.button("Save secure note"):
                st.session_state.vault_contents.setdefault("notes",[]).append({"text":note,"ts":datetime.utcnow().isoformat()})
                save_vault()
                st.success("Saved.")
            if st.button("Lock Vault"):
                st.session_state.vault_unlocked = False
                st.session_state.vault_contents = {}
                st.session_state.vault_key = None
                st.success("Locked.")

def export_page():
    st.header("Export / Backup")
    if st.button("Export all JSON"):
        payload = {"memories":memories,"timeline":timeline,"chats":chats,"exported_at":datetime.utcnow().isoformat()}
        b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Download archive.json", data=b, file_name="echosoul_export.json", mime="application/json")
    if st.button("Export timeline CSV"):
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["ts","type","text","tags"])
        for ev in timeline:
            w.writerow([ev.get("ts",""), ev.get("type",""), ev.get("text",""), ",".join(ev.get("tags",[]))])
        st.download_button("Download timeline.csv", data=buf.getvalue(), file_name="timeline.csv", mime="text/csv")

def brain_mimic_page():
    st.header("Brain Mimic")
    st.write("EchoSoul will try to reply in your voice using examples from your past messages.")
    n = st.slider("Use how many recent user messages as examples?", 1, 20, 6)
    examples = [m['text'] for m in chats]
