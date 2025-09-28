# app.py
"""
EchoSoul — Neon UI personal AI companion (single-file)
- Works without optional audio libs (graceful fallbacks)
- Uses new OpenAI client if available (OpenAI >= 1.0)
Instructions:
- Put OPENAI_API_KEY in Streamlit secrets or env var.
- Optional: add gTTS to requirements for offline TTS fallback: pip install gTTS
"""

import streamlit as st
import os
import json
import base64
import time
from datetime import datetime
from io import BytesIO

# Optional imports: try them, but don't crash if absent
_has_gtts = False
try:
    from gtts import gTTS
    _has_gtts = True
except Exception:
    _has_gtts = False

# Try to import new OpenAI client
_has_openai = False
try:
    from openai import OpenAI
    _has_openai = True
except Exception:
    _has_openai = False

# -------------------------
# Config & constants
# -------------------------
st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="wide")
DATA_FILE = "echosoul_data.json"
MODEL = "gpt-4o-mini"   # chat
TTS_MODEL = "gpt-4o-mini-tts"  # optional OpenAI TTS model name; may not exist in your account
NEON = "#00FFF7"
BG_GRAD = "linear-gradient(135deg, #0a0f1f 0%, #1a1137 100%)"
DEFAULT_TEMP = 0.7

# Initialize OpenAI client if key present
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
openai_client = None
if _has_openai and OPENAI_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        openai_client = None

# -------------------------
# Persistence helpers
# -------------------------
def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        base = {
            "profile": {"name": "", "tone": "Adaptive", "voice_enabled": False, "created_at": datetime.utcnow().isoformat()},
            "memories": [],
            "vault": {"entries": []},
            "conversations": []
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2, ensure_ascii=False)

def read_data():
    ensure_datafile()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_data(obj):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# Simple XOR "vault" (prototype only)
def xor_encrypt(plaintext: str, password: str) -> str:
    b = plaintext.encode("utf-8")
    kp = password.encode("utf-8")
    out = bytearray()
    for i, c in enumerate(b):
        out.append(c ^ kp[i % len(kp)])
    return base64.b64encode(bytes(out)).decode("utf-8")

def xor_decrypt(b64: str, password: str) -> str:
    b = base64.b64decode(b64)
    kp = password.encode("utf-8")
    out = bytearray()
    for i, c in enumerate(b):
        out.append(c ^ kp[i % len(kp)])
    return out.decode("utf-8")

# -------------------------
# Small NLP heuristics
# -------------------------
def simple_sentiment(text: str) -> str:
    t = text.lower()
    pos = sum(t.count(w) for w in ["good","great","happy","love","awesome","nice","fantastic","helpful","yes"])
    neg = sum(t.count(w) for w in ["sad","bad","angry","hate","frustrat","upset","no","problem","terrible"])
    if pos > neg + 1:
        return "positive"
    if neg > pos + 1:
        return "negative"
    return "neutral"

def confidence_heuristic(text: str) -> float:
    hedges = ["might","could","maybe","possibly","it depends","i think","i believe"]
    score = 0.85
    lower = text.lower()
    for h in hedges:
        if h in lower:
            score -= 0.12
    q = lower.count("?")
    score -= min(0.15, 0.03 * q)
    if any(p in lower for p in ["i don't know","i'm not sure","i do not know"]):
        score -= 0.3
    return max(0.05, min(0.99, round(score,2)))

# -------------------------
# OpenAI wrappers (safe)
# -------------------------
def openai_chat(messages, temperature=DEFAULT_TEMP, max_tokens=800):
    """
    Generic chat call using new OpenAI client if available; returns {'text','error'}
    """
    if not openai_client:
        return {"text": "⚠️ OpenAI client not configured. Put your API key into Streamlit secrets (OPENAI_API_KEY).", "error": "no_client"}
    try:
        # new SDK: client.chat.completions.create(...)
        resp = openai_client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return {"text": resp.choices[0].message.content, "error": None}
    except Exception as e:
        return {"text": f"⚠️ AI call failed: {str(e)}", "error": str(e)}

def openai_tts(text: str, voice: str = "alloy"):
    """
    Attempt OpenAI TTS, then gTTS fallback if available.
    Returns bytes (MP3) or None on failure.
    """
    # Try OpenAI TTS if supported
    if openai_client:
        try:
            tts_resp = openai_client.audio.speech.create(model=TTS_MODEL, voice=voice, input=text)
            # Some SDKs return a streaming object; try to read bytes
            if hasattr(tts_resp, "read"):
                return tts_resp.read()
            # else attempt several fallbacks
            if isinstance(tts_resp, (bytes, bytearray)):
                return bytes(tts_resp)
            # else try attribute access:
            b = getattr(tts_resp, "binary", None) or getattr(tts_resp, "data", None)
            if b:
                return bytes(b)
        except Exception:
            # swallow and try gTTS
            pass

    # Fallback to gTTS if installed
    if _has_gtts:
        try:
            tts_obj = gTTS(text, lang="en")
            buf = BytesIO()
            tts_obj.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception:
            return None

    # No TTS available
    return None

def openai_transcribe(uploaded_file):
    """
    Attempt OpenAI transcription; returns {'text','error'}.
    uploaded_file may be a BytesIO or UploadedFile.
    """
    if not openai_client:
        return {"text": None, "error": "no_client"}
    try:
        # New SDK usually expects a file-like; pass uploaded_file (stream) directly
        resp = openai_client.audio.transcriptions.create(file=uploaded_file, model="gpt-4o-transcribe")  # example; may be 'whisper-1' depending on your account
        # try to return text attribute or dict
        text = getattr(resp, "text", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("text")
        return {"text": text, "error": None}
    except Exception as e:
        return {"text": None, "error": str(e)}

# -------------------------
# CSS neon/glass (matching the look you want)
# -------------------------
st.markdown(f"""
<style>
body {{
  background: {BG_GRAD};
  color: #e6faf9;
  font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}}
h1.neon {{
  color: {NEON};
  font-size: 38px;
  font-weight: 800;
  text-shadow: 0 0 12px {NEON}, 0 0 28px rgba(0,255,247,0.08);
}}
.neon-box {{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 12px;
  padding: 14px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 12px 40px rgba(2,6,23,0.6);
}}
textarea, input, select {{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  color: #e6faf9 !important;
  border-radius: 10px !important;
}}
button.stButton>button {{
  background: {NEON} !important;
  color: #001517 !important;
  font-weight:700 !important;
  border-radius: 10px !important;
  box-shadow: 0 8px 30px rgba(0,255,247,0.12) !important;
}}
.small-muted {{ color: #bcd; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session state defaults
# -------------------------
if "onboarded" not in st.session_state:
    st.session_state["onboarded"] = False
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "last_ai_trace" not in st.session_state:
    st.session_state["last_ai_trace"] = None
if "vault_pass" not in st.session_state:
    st.session_state["vault_pass"] = ""

# -------------------------
# Onboarding
# -------------------------
data = read_data()
profile = data.get("profile", {})

if not st.session_state["onboarded"]:
    st.markdown(f"<h1 class='neon'>✨ EchoSoul</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Quick setup — this only takes a moment.</div>")
    with st.form("onboard", clear_on_submit=False):
        name = st.text_input("What should I call you?", value=profile.get("name",""))
        tone = st.selectbox("Preferred default tone:", ["Adaptive","Friendly","Empathetic","Energetic","Formal"], index=0)
        enable_voice = st.checkbox("Enable voice features (TTS/STT) if available", value=profile.get("voice_enabled", False))
        sub = st.form_submit_button("Save & Continue")
    if sub:
        profile["name"] = name
        profile["tone"] = tone
        profile["voice_enabled"] = enable_voice
        profile["created_at"] = profile.get("created_at", datetime.utcnow().isoformat())
        data["profile"] = profile
        write_data(data)
        st.session_state["onboarded"] = True
        st.success("Saved! Welcome.")
    else:
        st.stop()

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown(f"<h3 style='color:{NEON}'>EchoSoul</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Your personal neon AI companion</div>")
    st.markdown("---")
    page = st.radio("Navigate", ["Chat","Life Timeline","Vault","Voice","Export","Settings"])
    st.markdown("---")
    st.markdown(f"<div class='small-muted'>Model: {MODEL}</div>")
    st.markdown("---")
    if st.button("Export data (JSON)"):
        cur = read_data()
        st.download_button("Download JSON", json.dumps(cur, indent=2, ensure_ascii=False), file_name="echosoul_export.json")

# Floating quick-call (visual only)
st.markdown("""
<div style="position:fixed;right:18px;top:18px;z-index:9999">
  <button onclick="window.location.hash='call-ai'">Quick chat</button>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Chat page
# -------------------------
if page == "Chat":
    st.markdown(f"<h1 class='neon'>Hello, {profile.get('name','Friend')}</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Ongoing conversation — input auto-clears after send.</div>")

    # render messages
    for msg in st.session_state["conversation"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='neon-box'><strong>{profile.get('name','You')}</strong> • <span class='small-muted'>{msg.get('ts','')}</span><div style='margin-top:8px'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            conf = msg.get("confidence")
            conf_html = f"<div class='small-muted' style='margin-top:8px'>Confidence: <strong style='color:{NEON}'>{int(conf*100)}%</strong></div>" if conf is not None else ""
            st.markdown(f"<div class='neon-box'><strong style='color:{NEON}'>EchoSoul</strong> • <span class='small-muted'>{msg.get('ts','')}</span><div style='margin-top:8px'>{msg['content']}</div>{conf_html}</div>", unsafe_allow_html=True)

    # chat form (clears)
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_area("Type to EchoSoul", height=120, placeholder="Ask a question, add a memory, or say 'remember that...'")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            roleplay = st.checkbox("Roleplay as you", value=False)
        with c2:
            tone_choice = st.selectbox("Tone", ["Adaptive","Friendly","Empathetic","Energetic","Formal"], index=0)
        with c3:
            temp = st.slider("Creativity", 0.0, 1.0, DEFAULT_TEMP, step=0.05)
        submitted = st.form_submit_button("Send")

    if submitted and user_text.strip():
        ts = datetime.utcnow().isoformat()
        st.session_state["conversation"].append({"role":"user","content":user_text,"ts":ts})
        # Build system prompt
        if tone_choice == "Adaptive":
            sentiment = simple_sentiment(user_text)
            if sentiment == "positive":
                tone_inst = "Respond in an energetic and uplifting tone."
            elif sentiment == "negative":
                tone_inst = "Respond in an empathetic, calming tone."
            else:
                tone_inst = "Respond in a friendly, neutral tone."
        else:
            tone_inst = f"Respond in a {tone_choice.lower()} tone."

        sys_prompt = f"You are EchoSoul, a helpful companion. {tone_inst} Address the user by name {profile.get('name','Friend')}. Be transparent and concise."

        # create messages from recent conversation
        msgs = [{"role":"system","content":sys_prompt}]
        for m in st.session_state["conversation"][-12:]:
            # ensure format: role/content
            msgs.append({"role": m["role"], "content": m["content"]})

        # Call OpenAI
        res = openai_chat(msgs, temperature=temp)
        ai_text = res["text"]
        conf = None if res.get("error") else confidence_heuristic(ai_text)
        st.session_state["conversation"].append({"role":"assistant","content":ai_text,"ts":datetime.utcnow().isoformat(),"confidence":conf})

        # persist a short conversation snapshot
        data = read_data()
        data.setdefault("conversations", []).append({"time": datetime.utcnow().isoformat(), "user": user_text, "ai": ai_text})
        write_data(data)
        # explainability trace
        st.session_state["last_ai_trace"] = {"prompt": user_text, "memories_used": [m["text"] for m in data.get("memories", [])[-5:]], "model": MODEL, "temp": temp, "confidence": conf}

# -------------------------
# Voice page (TTS/STT)
# -------------------------
elif page == "Voice":
    st.header("🔊 Voice (TTS & STT) — Prototype")
    st.markdown("Use OpenAI TTS if available, otherwise gTTS fallback (if installed). You can also upload audio to transcribe (OpenAI STT).")

    voice_choice = st.selectbox("Voice (if supported)", ["alloy", "verse", "amber"], index=0)

    # last AI reply
    last_ai = None
    for m in reversed(st.session_state["conversation"]):
        if m["role"] == "assistant":
            last_ai = m["content"]
            break

    if last_ai:
        st.markdown("**Last AI reply**")
        st.write(last_ai)
        if st.button("Play last reply (TTS)"):
            audio_bytes = openai_tts(last_ai, voice=voice_choice)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            else:
                if _has_openai and not openai_client:
                    st.error("OpenAI client not configured. Put OPENAI_API_KEY in Streamlit secrets.")
                elif not _has_gtts:
                    st.error("TTS unavailable: OpenAI TTS failed and gTTS not installed. Add gTTS to your requirements or enable OpenAI TTS.")
                else:
                    st.error("TTS generation failed.")

    st.markdown("---")
    st.subheader("Upload audio to transcribe (STT)")
    uploaded = st.file_uploader("Upload audio file (mp3/wav/m4a)", type=["mp3","wav","m4a","webm"])
    if uploaded:
        st.info("Attempting transcription...")
        transcription = openai_transcribe(uploaded)
        if transcription.get("error"):
            st.error(f"Transcription failed: {transcription['error']}")
        else:
            st.success("Transcription complete")
            st.write(transcription["text"])

# -------------------------
# Life Timeline page
# -------------------------
elif page == "Life Timeline":
    st.header("📅 Life Timeline")
    data = read_data()
    memories = data.get("memories", [])
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Saved memories")
        if memories:
            for m in sorted(memories, key=lambda x: x.get("created_at",""), reverse=True):
                st.markdown(f"- **{m.get('created_at','')[:10]}** — {m.get('text')[:220]}")
        else:
            st.info("No memories yet.")
    with colB:
        st.subheader("Add memory")
        new_mem = st.text_area("Memory text", key="timeline_new")
        if st.button("Save memory"):
            if new_mem.strip():
                mem = {"id": str(int(time.time()*1000)), "text": new_mem.strip(), "tags": [], "created_at": datetime.utcnow().isoformat()}
                data.setdefault("memories", []).append(mem)
                write_data(data)
                st.success("Saved.")
            else:
                st.warning("Cannot save empty memory.")

# -------------------------
# Vault page
# -------------------------
elif page == "Vault":
    st.header("🔐 Vault (Prototype)")
    data = read_data()
    vp = st.text_input("Vault password (session only)", type="password", key="vault_pw")
    if vp:
        st.session_state["vault_pass"] = vp
        st.success("Vault password stored in session (prototype).")
    title = st.text_input("Title", key="vault_title")
    note = st.text_area("Private note", key="vault_note", height=140)
    if st.button("Save to vault"):
        if not st.session_state.get("vault_pass"):
            st.warning("Set a vault password first.")
        elif not note.strip():
            st.warning("Cannot save empty.")
        else:
            enc = xor_encrypt(note.strip(), st.session_state["vault_pass"])
            entry = {"id": str(int(time.time()*1000)), "title": title or "Untitled", "encrypted": enc, "created_at": datetime.utcnow().isoformat()}
            data.setdefault("vault", {}).setdefault("entries", []).append(entry)
            write_data(data)
            st.success("Saved to vault (prototype XOR).")
    st.markdown("---")
    st.subheader("Entries")
    entries = data.get("vault", {}).get("entries", [])
    if entries:
        for e in entries:
            if st.session_state.get("vault_pass"):
                try:
                    dec = xor_decrypt(e["encrypted"], st.session_state["vault_pass"])
                    st.markdown(f"- **{e.get('title')}** ({e.get('created_at')[:10]}): {dec[:160]}")
                except Exception:
                    st.markdown(f"- **{e.get('title')}** ({e.get('created_at')[:10]}): _locked (wrong password)_")
            else:
                st.markdown(f"- **{e.get('title')}** ({e.get('created_at')[:10]}): _locked (no password in session)_")
    else:
        st.info("No entries yet.")

# -------------------------
# Export page
# -------------------------
elif page == "Export":
    st.header("Export")
    cur = read_data()
    st.download_button("Download JSON", json.dumps(cur, indent=2, ensure_ascii=False), file_name="echosoul_export.json")

# -------------------------
# Settings page
# -------------------------
elif page == "Settings":
    st.header("Settings")
    data = read_data()
    prof = data.get("profile", {})
    new_name = st.text_input("Display name", value=prof.get("name",""))
    new_tone = st.selectbox("Default tone", ["Adaptive","Friendly","Empathetic","Energetic","Formal"], index=0)
    enable_voice = st.checkbox("Enable voice features (prototype)", value=prof.get("voice_enabled", False))
    if st.button("Save settings"):
        prof["name"] = new_name
        prof["tone"] = new_tone
        prof["voice_enabled"] = enable_voice
        data["profile"] = prof
        write_data(data)
        st.success("Saved.")

# -------------------------
# Explainability & footer
# -------------------------
st.markdown("---")
trace = st.session_state.get("last_ai_trace")
if trace:
    st.markdown("**Explainability (last reply)**")
    st.write(f"Prompt: {trace['prompt'][:300]}")
    st.write(f"Model: {trace.get('model','—')}  Temperature: {trace.get('temp', '—')}")
    if trace.get("confidence") is not None:
        st.progress(trace["confidence"])
        st.write(f"Confidence (heuristic): {int(trace['confidence']*100)}%")
else:
    st.markdown("<div class='small-muted'>EchoSoul ready — start a chat above.</d
