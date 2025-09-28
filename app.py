# app.py
import streamlit as st
import openai
import json
import os
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(page_title="EchoSoul", page_icon="⚡", layout="wide")
DATA_FILE = "echosoul_data.json"
MODEL_NAME = "gpt-4o-mini"  # change if needed
DEFAULT_TEMP = 0.7
NEON = "#3EE7A8"
DARK_BG = "#0D0D2E"
LIGHT_BG = "#F6F7FB"

# Load OpenAI key from Streamlit secrets or env var
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

# ---------------------------
# Utilities
# ---------------------------
def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        initial = {
            "profile": {"name": "", "preferred_tone": "Adaptive", "voice_enabled": False, "created_at": datetime.utcnow().isoformat()},
            "memories": [],
            "vault": {"entries": []},
            "conversations": []
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)

def read_data() -> Dict:
    ensure_datafile()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_data(obj: Dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def xor_encrypt(plaintext: str, password: str) -> str:
    if password is None or password == "":
        raise ValueError("Password required for encryption.")
    btxt = plaintext.encode("utf-8")
    bpass = password.encode("utf-8")
    out = bytearray()
    for i, c in enumerate(btxt):
        out.append(c ^ bpass[i % len(bpass)])
    return base64.b64encode(bytes(out)).decode("utf-8")

def xor_decrypt(b64cipher: str, password: str) -> str:
    b = base64.b64decode(b64cipher)
    bpass = password.encode("utf-8")
    out = bytearray()
    for i, c in enumerate(b):
        out.append(c ^ bpass[i % len(bpass)])
    return out.decode("utf-8")

def simple_sentiment(text: str) -> str:
    t = text.lower()
    pos = sum(t.count(w) for w in ["good","great","happy","love","awesome","nice","fantastic","helpful","yes"])
    neg = sum(t.count(w) for w in ["sad","bad","angry","hate","frustrat","upset","no","problem","terrible"])
    if pos > neg + 1:
        return "positive"
    if neg > pos + 1:
        return "negative"
    return "neutral"

def confidence_heuristic(ai_text: str) -> float:
    hedges = ["might","could","maybe","possibly","it depends","i think","i believe"]
    score = 0.85
    lower = ai_text.lower()
    for h in hedges:
        if h in lower:
            score -= 0.12
    q = lower.count("?")
    score -= min(0.15, 0.03 * q)
    if any(p in lower for p in ["i don't know","i'm not sure","i do not know"]):
        score -= 0.3
    return max(0.05, min(0.99, round(score,2)))

def call_openai_chat(prompt: str, system_role: str, memory_snippets: List[str], temp: float) -> Dict:
    # Prepare messages
    messages = [{"role":"system", "content": system_role}]
    if memory_snippets:
        mem_block = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memory_snippets[-6:])
        messages.append({"role":"system","content": mem_block})
    messages.append({"role":"user","content": prompt})

    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temp,
            max_tokens=800,
        )
        text = resp.choices[0].message["content"]
        return {"text": text, "raw": resp, "error": None}
    except Exception as e:
        return {"text": f"⚠️ AI call failed: {str(e)}", "raw": None, "error": str(e)}

# ---------------------------
# CSS / UI helpers
# ---------------------------
def inject_css():
    st.markdown(f"""
    <style>
    :root {{
      --neon: {NEON};
      --bg-dark: {DARK_BG};
      --bg-light: {LIGHT_BG};
    }}
    .top-right-call {{
      position: fixed;
      top: 18px;
      right: 18px;
      z-index: 9999;
      background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.04);
      padding: 10px 14px;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
      color: white;
      display:flex;
      gap:8px;
      align-items:center;
      cursor:pointer;
    }}
    .neon-text {{ color: var(--neon); text-shadow: 0 0 10px rgba(62,231,168,0.15); }}
    .glass {{
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.04);
      backdrop-filter: blur(6px);
    }}
    .tiny-muted {{ color: #B8C0D0; font-size:12px; }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------------------
# Session defaults
# ---------------------------
if "onboarded" not in st.session_state:
    st.session_state["onboarded"] = False
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "last_ai_trace" not in st.session_state:
    st.session_state["last_ai_trace"] = None
if "vault_pass" not in st.session_state:
    st.session_state["vault_pass"] = ""

# ---------------------------
# Onboarding (ask a few questions at start)
# ---------------------------
data = read_data()
profile = data.get("profile", {})

if not st.session_state["onboarded"]:
    st.markdown("<div style='display:flex;gap:12px;align-items:center'><h1 class='neon-text'>⚡ Welcome to EchoSoul</h1><div class='tiny-muted'>Your personal AI companion</div></div>", unsafe_allow_html=True)
    st.markdown("Let's get you set up — this only takes a moment.")
    with st.form("onboard", clear_on_submit=False):
        name = st.text_input("What should I call you?", value=profile.get("name",""))
        tone = st.selectbox("Preferred default tone:", ["Adaptive", "Friendly", "Empathetic", "Energetic", "Formal"], index=["Adaptive","Friendly","Empathetic","Energetic","Formal"].index(profile.get("preferred_tone","Adaptive")) if profile.get("preferred_tone") else 0)
        voice_enabled = st.checkbox("Enable voice responses (prototype)?", value=profile.get("voice_enabled", False))
        submit_onboard = st.form_submit_button("Save & Continue")
    if submit_onboard:
        profile["name"] = name
        profile["preferred_tone"] = tone
        profile["voice_enabled"] = voice_enabled
        profile["created_at"] = profile.get("created_at", datetime.utcnow().isoformat())
        data["profile"] = profile
        write_data(data)
        st.session_state["onboarded"] = True
        st.experimental_rerun()
    else:
        st.stop()

# ---------------------------
# Layout: sidebar + main area
# ---------------------------
with st.sidebar:
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px'><div style='width:44px;height:44px;border-radius:10px;background:{NEON};box-shadow:0 6px 14px rgba(62,231,168,0.12)'></div><div><h3 style='margin:0'>EchoSoul</h3><div class='tiny-muted'>Personal AI companion</div></div></div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["Chat", "Life Timeline", "Vault", "Legacy & Export", "Settings"], index=0)
    st.markdown("---")
    theme = st.radio("Theme Mode", ["Dark", "Light"], index=0)
    if theme == "Dark":
        st.markdown(f"<style>body{{background:{DARK_BG};color:#F0F0F0}}</style>", unsafe_allow_html=True)
    else:
        st.markdown(f"<style>body{{background:{LIGHT_BG};color:#0D0D2E}}</style>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Prototype Vault** — not secure encryption (XOR demo).")
    st.text("OpenAI model: " + MODEL_NAME)
    st.markdown("---")
    if st.button("Export Data (JSON)"):
        current = read_data()
        st.download_button("Download JSON", json.dumps(current, indent=2, ensure_ascii=False), file_name="echosoul_export.json")

# floating call AI button (pure UI)
st.markdown("""
<div class="top-right-call" onclick="window.location.hash='call-ai'">
  <div style="font-weight:700">Call AI</div>
  <div class='tiny-muted' style='font-size:12px'>Quick chat</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Chat Page
# ---------------------------
if page == "Chat":
    st.markdown(f"<h2 class='neon-text'>Hello, {profile.get('name','Friend')} 👋</h2>", unsafe_allow_html=True)
    st.markdown("<div class='tiny-muted'>This conversation is ongoing — EchoSoul keeps context and suggests memories to save.</div>", unsafe_allow_html=True)
    st.markdown("")

    # display conversation
    for m in st.session_state["conversation"]:
        if m["role"] == "user":
            st.markdown(f"<div class='glass' style='margin-bottom:8px'><strong>{profile.get('name','You')}</strong> <span class='tiny-muted'>• {m.get('ts','')}</span><div style='margin-top:6px'>{m['content']}</div></div>", unsafe_allow_html=True)
        else:
            conf = m.get("confidence")
            conf_html = ""
            if conf:
                conf_html = f"<div class='tiny-muted' style='margin-top:8px'>Confidence: <strong style='color:{NEON}'>{int(conf*100)}%</strong></div>"
            st.markdown(f"<div class='glass' style='margin-bottom:8px'><strong class='neon-text'>EchoSoul</strong> <span class='tiny-muted'>• {m.get('ts','')}</span><div style='margin-top:6px'>{m['content']}</div>{conf_html}</div>", unsafe_allow_html=True)

    # form for sending (clears on submit)
    with st.form(key="chat_form", clear_on_submit=True):
        user_text = st.text_area("Type to EchoSoul", key="user_text", placeholder="Ask a question, tell a memory, or say 'roleplay'...", height=120)
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            roleplay = st.checkbox("Roleplay as you", value=False)
        with col2:
            tone_select = st.selectbox("Tone", ["Adaptive", "Friendly", "Empathetic", "Energetic", "Formal"], index=0)
        with col3:
            temp = st.slider("Creativity", 0.0, 1.0, DEFAULT_TEMP, step=0.05)
        send_btn = st.form_submit_button("Send")

    if send_btn and user_text.strip():
        ts = datetime.utcnow().isoformat()
        st.session_state["conversation"].append({"role":"user","content":user_text,"ts":ts})
        # choose tone instruction
        if tone_select == "Adaptive":
            sentiment = simple_sentiment(user_text)
            if sentiment == "positive":
                tone_instruction = "Respond in an energetic and uplifting tone."
            elif sentiment == "negative":
                tone_instruction = "Respond in an empathetic, calming tone."
            else:
                tone_instruction = "Respond in a friendly, neutral tone."
        else:
            tone_instruction = f"Respond in a {tone_select.lower()} tone."

        system_role = f"You are EchoSoul, a helpful assistant. {tone_instruction} Be transparent, show brief reasoning and a confidence estimate."

        # prepare memories (naive retrieval: last 5)
        data = read_data()
        mems = [m["text"] for m in data.get("memories", [])[-5:]]

        ai_res = call_openai_chat(user_text, system_role, mems, temp)
        ai_text = ai_res["text"]
        conf = None
        if not ai_res.get("error"):
            conf = confidence_heuristic(ai_text)

        st.session_state["conversation"].append({"role":"ai","content":ai_text,"ts":datetime.utcnow().isoformat(),"confidence":conf})
        # Save conversation snapshot to data
        data.setdefault("conversations", []).append({
            "time": datetime.utcnow().isoformat(),
            "user": user_text,
            "ai": ai_text,
            "meta": {"model": MODEL_NAME, "temp": temp}
        })
        write_data(data)
        # store last ai trace for explainability pane
        st.session_state["last_ai_trace"] = {
            "prompt": user_text,
            "memories_used": mems,
            "model": MODEL_NAME,
            "temperature": temp,
            "confidence": conf
        }
        # don't call any session_state direct editing of widget values; form cleared automatically.

    # Explainability / trace
    st.markdown("---")
    st.header("Explainable AI (XAI) — Trace")
    last = st.session_state.get("last_ai_trace")
    if last:
        st.markdown(f"**Model**: `{last['model']}`  ")
        st.markdown(f"**Temperature**: {last['temperature']}")
        st.markdown("**Prompt you sent**")
        st.code(last["prompt"][:400])
        st.markdown("**Memories included**")
        if last["memories_used"]:
            for mm in last["memories_used"]:
                st.write(f"- {mm[:120]}")
        else:
            st.write("_No recent memories used._")
        st.markdown("**Confidence (heuristic)**")
        if last["confidence"] is not None:
            st.progress(last["confidence"])
            st.write(f"{int(last['confidence']*100)}% (heuristic)")
    else:
        st.write("Your next AI reply will show a short explainability trace here.")

# ---------------------------
# Life Timeline Page
# ---------------------------
elif page == "Life Timeline":
    st.header("📅 Life Timeline")
    data = read_data()
    memories = data.get("memories", [])
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Saved memories")
        if memories:
            for m in sorted(memories, key=lambda x: x.get("created_at",""), reverse=True):
                st.markdown(f"- **{m.get('created_at','')[:10]}** — {m.get('text')[:200]}")
        else:
            st.info("No memories yet. Add memories from chat or here.")
    with colB:
        st.subheader("Add memory")
        new_mem = st.text_area("Memory text", key="new_mem_text", height=120)
        if st.button("Save memory"):
            if new_mem.strip():
                mem = {"id": str(int(time.time()*1000)), "text": new_mem.strip(), "tags": [], "created_at": datetime.utcnow().isoformat()}
                data.setdefault("memories", []).append(mem)
                write_data(data)
                st.success("Memory saved.")
            else:
                st.warning("Cannot save empty memory.")

# ---------------------------
# Vault Page
# ---------------------------
elif page == "Vault":
    st.header("🔐 Private Vault (Prototype)")
    data = read_data()
    vp = st.text_input("Enter vault password (session-only)", type="password", key="vault_pass_input")
    if vp:
        st.session_state["vault_pass"] = vp
        st.success("Vault loaded in session (prototype).")
    st.markdown("Add private note:")
    title = st.text_input("Title", key="vault_title")
    note = st.text_area("Content", key="vault_content", height=120)
    if st.button("Save to Vault"):
        if not st.session_state.get("vault_pass"):
            st.warning("Set a vault password first in the field above.")
        elif not note.strip():
            st.warning("Cannot save empty note.")
        else:
            enc = xor_encrypt(note.strip(), st.session_state["vault_pass"])
            entry = {"id": str(int(time.time()*1000)), "title": title or "Untitled", "encrypted": enc, "created_at": datetime.utcnow().isoformat()}
            data.setdefault("vault", {}).setdefault("entries", []).append(entry)
            write_data(data)
            st.success("Saved to vault (prototype XOR).")
    st.markdown("---")
    st.subheader("Vault entries (decrypted if password set)")
    entries = data.get("vault", {}).get("entries", [])
    if entries:
        for e in entries:
            if st.session_state.get("vault_pass"):
                try:
                    dec = xor_decrypt(e["encrypted"], st.session_state["vault_pass"])
                    st.markdown(f"- **{e.get('title')}** ({e.get('created_at')[:10]}): {dec[:180]}")
                except Exception:
                    st.markdown(f"- **{e.get('title')}** ({e.get('created_at')[:10]}): _locked (wrong password)_")
            else:
                st.markdown(f"- **{e.get('title')}** ({e.get('created_at')[:10]}): _locked (no password in session)_")
    else:
        st.info("No vault entries yet.")

# ---------------------------
# Legacy & Export
# ---------------------------
elif page == "Legacy & Export":
    st.header("📜 Legacy & Export")
    data = read_data()
    st.markdown("Download your full data (memories, vault metadata, conversations)")
    st.download_button("Download data (JSON)", json.dumps(data, indent=2, ensure_ascii=False), file_name="echosoul_export.json")
    st.markdown("---")
    st.subheader("Legacy Snapshot (human-readable)")
    st.markdown("**Profile**")
    st.write(data.get("profile", {}))
    st.markdown("**Recent memories**")
    for m in data.get("memories", [])[-10:]:
        st.markdown(f"- {m.get('created_at','')} — {m.get('text')[:200]}")

# ---------------------------
# Settings
# ---------------------------
elif page == "Settings":
    st.header("⚙️ Settings")
    data = read_data()
    prof = data.get("profile", {})
    new_name = st.text_input("Display name", value=prof.get("name",""))
    new_tone = st.selectbox("Default tone", ["Adaptive","Friendly","Empathetic","Energetic","Formal"], index=["Adaptive","Friendly","Empathetic","Energetic","Formal"].index(prof.get("preferred_tone","Adaptive")))
    voice_toggle = st.checkbox("Enable voice responses (prototype)", value=prof.get("voice_enabled", False))
    if st.button("Save settings"):
        prof["name"] = new_name
        prof["preferred_tone"] = new_tone
        prof["voice_enabled"] = voice_toggle
        data["profile"] = prof
        write_data(data)
        st.success("Settings saved.")

# ---------------------------
# Footer / privacy note
# ---------------------------
st.markdown("---")
st.markdown("<div class='tiny-muted'>Privacy: EchoSoul stores data locally in <code>echosoul_data.json</code>. Vault encryption is a demo (XOR) — not secure. Do not store production secrets unless you replace with real encryption. The AI may be fallible; use the explainability trace to review answers.</div>", unsafe_allow_html=True)
