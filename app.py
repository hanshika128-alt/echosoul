# app.py
"""
EchoSoul — Personal AI companion (single-file Streamlit app)

Instructions:
- Put your OpenAI key in Streamlit secrets as OPENAI_API_KEY, or set the env var OPENAI_API_KEY.
- `pip install streamlit openai` (requirements: streamlit, openai)
- Deploy to Streamlit Cloud or run locally: `streamlit run app.py`
"""

import streamlit as st
import openai
import json
import os
import base64
import time
from datetime import datetime
from typing import List, Dict

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="EchoSoul", page_icon="⚡", layout="wide")
DATA_FILE = "echosoul_data.json"
MODEL_NAME = "gpt-4o-mini"        # change if needed
DEFAULT_TEMP = 0.7
NEON = "#3EE7A8"
DARK_BG = "#0D0D2E"
LIGHT_BG = "#F6F7FB"

# Load OpenAI key
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

# -------------------------
# Utilities: data + crypto (XOR prototype)
# -------------------------
def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        base = {
            "profile": {"name": "", "preferred_tone": "Adaptive", "voice_enabled": False, "created_at": datetime.utcnow().isoformat()},
            "memories": [],
            "vault": {"entries": []},
            "conversations": []
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2, ensure_ascii=False)

def read_data() -> Dict:
    ensure_datafile()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_data(obj: Dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def xor_encrypt(plaintext: str, password: str) -> str:
    if not password:
        raise ValueError("Password required.")
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
# Heuristics
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
# OpenAI wrapper
# -------------------------
def call_openai_chat(prompt: str, system_role: str, memory_snippets: List[str], temperature: float = DEFAULT_TEMP) -> Dict:
    messages = [{"role":"system", "content": system_role}]
    if memory_snippets:
        mem_block = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memory_snippets[-6:])
        messages.append({"role":"system","content": mem_block})
    messages.append({"role":"user","content": prompt})
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=800
        )
        text = resp.choices[0].message["content"]
        return {"text": text, "raw": resp, "error": None}
    except Exception as e:
        return {"text": f"⚠️ AI call failed: {str(e)}", "raw": None, "error": str(e)}

# -------------------------
# Inject CSS for neon / glass
# -------------------------
def inject_css():
    st.markdown(f"""
    <style>
    :root{{ --neon: {NEON}; --bg-dark: {DARK_BG}; --bg-light: {LIGHT_BG}; }}
    .neon-text {{ color: var(--neon); text-shadow: 0 0 12px rgba(62,231,168,0.15); font-weight:700; }}
    .glass {{ background: rgba(255,255,255,0.02); border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,0.04); backdrop-filter: blur(6px); }}
    .topcall {{ position: fixed; top:18px; right:18px; z-index:9999; padding:10px 14px; border-radius:12px; cursor:pointer; border:1px solid rgba(255,255,255,0.04); background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); }}
    .tiny-muted {{ color:#B8C0D0; font-size:12px; }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

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
# Onboarding (asks a few intro questions)
# -------------------------
data = read_data()
profile = data.get("profile", {})

if not st.session_state["onboarded"]:
    st.markdown(f"<h1 class='neon-text'>⚡ Welcome to EchoSoul</h1>", unsafe_allow_html=True)
    st.markdown("Let's get you set up — this only takes a moment.")
    with st.form("onboard_form", clear_on_submit=False):
        name = st.text_input("What should I call you?", value=profile.get("name",""))
        tone = st.selectbox("Preferred default tone:", ["Adaptive", "Friendly", "Empathetic", "Energetic", "Formal"], index=0)
        voice_enabled = st.checkbox("Enable voice responses (prototype)?", value=profile.get("voice_enabled", False))
        submitted = st.form_submit_button("Save & Continue")
    if submitted:
        profile["name"] = name
        profile["preferred_tone"] = tone
        profile["voice_enabled"] = voice_enabled
        profile["created_at"] = profile.get("created_at", datetime.utcnow().isoformat())
        data["profile"] = profile
        write_data(data)
        st.session_state["onboarded"] = True
        st.success("Onboarding saved. Welcome — you can continue using EchoSoul below.")
        # No st.experimental_rerun() — continue in same run
    else:
        st.stop()

# -------------------------
# Sidebar layout
# -------------------------
with st.sidebar:
    st.markdown(f"<div style='display:flex;gap:10px;align-items:center'><div style='width:44px;height:44px;border-radius:10px;background:{NEON};box-shadow:0 6px 14px rgba(62,231,168,0.12)'></div><div><h3 style='margin:0'>EchoSoul</h3><div class='tiny-muted'>Your personal AI companion</div></div></div>", unsafe_allow_html=True)
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
    st.markdown(f"Model: `{MODEL_NAME}`")
    st.markdown("---")
    if st.button("Export Data (JSON)"):
        cur = read_data()
        st.download_button("Download JSON", json.dumps(cur, indent=2, ensure_ascii=False), file_name="echosoul_export.json")

# Floating Call AI button (UI only)
st.markdown("""
<div class="topcall glass" onclick="window.location.hash='call-ai'">
  <div style="font-weight:700">Call AI</div>
  <div class="tiny-muted" style="font-size:12px">Quick chat</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Pages
# -------------------------
if page == "Chat":
    # Header
    st.markdown(f"<h2 class='neon-text'>Hello, {profile.get('name','Friend')} 👋</h2>", unsafe_allow_html=True)
    st.markdown("<div class='tiny-muted'>Ongoing conversation — input auto-clears after send.</div>", unsafe_allow_html=True)
    st.markdown("")

    # Show conversation messages
    for m in st.session_state["conversation"]:
        if m["role"] == "user":
            st.markdown(f"<div class='glass' style='margin-bottom:8px'><strong>{profile.get('name','You')}</strong> <span class='tiny-muted'>• {m.get('ts','')}</span><div style='margin-top:6px'>{m['content']}</div></div>", unsafe_allow_html=True)
        else:
            conf = m.get("confidence")
            conf_html = f"<div class='tiny-muted' style='margin-top:8px'>Confidence: <strong style='color:{NEON}'>{int(conf*100)}%</strong></div>" if conf is not None else ""
            st.markdown(f"<div class='glass' style='margin-bottom:8px'><strong class='neon-text'>EchoSoul</strong> <span class='tiny-muted'>• {m.get('ts','')}</span><div style='margin-top:6px'>{m['content']}</div>{conf_html}</div>", unsafe_allow_html=True)

    # Input form — clears on submit
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_area("Type to EchoSoul", height=120, placeholder="Ask a question, add a memory, or say 'roleplay'...")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            roleplay = st.checkbox("Roleplay as you", value=False)
        with col2:
            tone_choice = st.selectbox("Tone", ["Adaptive", "Friendly", "Empathetic", "Energetic", "Formal"], index=0)
        with col3:
            temp = st.slider("Creativity", 0.0, 1.0, DEFAULT_TEMP, step=0.05)
        sent = st.form_submit_button("Send")

    if sent and user_text.strip():
        ts = datetime.utcnow().isoformat()
        st.session_state["conversation"].append({"role":"user","content":user_text,"ts":ts})

        # Build system role based on tone and sentiment
        tone_instruction = ""
        if tone_choice == "Adaptive":
            sent_label = simple_sentiment(user_text)
            tone_instruction = "Respond in an energetic tone." if sent_label=="positive" else ("Respond in an empathetic, calming tone." if sent_label=="negative" else "Respond in a friendly, neutral tone.")
        else:
            tone_instruction = f"Respond in a {tone_choice.lower()} tone."

        system_role = f"You are EchoSoul, a helpful assistant. {tone_instruction} Be transparent about limitations and show a short confidence estimate."

        # Memory retrieval (naive: last 5 memories)
        data = read_data()
        memories = [m["text"] for m in data.get("memories", [])[-5:]]

        # Call OpenAI
        ai_result = call_openai_chat(user_text, system_role, memories, temp)
        ai_text = ai_result["text"]
        conf = None if ai_result.get("error") else confidence_heuristic(ai_text)

        st.session_state["conversation"].append({"role":"ai","content":ai_text,"ts":datetime.utcnow().isoformat(),"confidence":conf})

        # Save short conversation snapshot
        data.setdefault("conversations", []).append({
            "time": datetime.utcnow().isoformat(),
            "user": user_text,
            "ai": ai_text,
            "meta": {"model": MODEL_NAME, "temp": temp}
        })
        write_data(data)

        # Set explainability trace
        st.session_state["last_ai_trace"] = {"prompt": user_text, "memories_used": memories, "model": MODEL_NAME, "temperature": temp, "confidence": conf}

    # Explainability pane
    st.markdown("---")
    st.header("Explainable AI (XAI) — Trace")
    trace = st.session_state.get("last_ai_trace")
    if trace:
        st.markdown(f"**Model:** `{trace['model']}`  ")
        st.markdown(f"**Temperature:** {trace['temperature']}")
        st.markdown("**Prompt you sent**")
        st.code(trace["prompt"][:400])
        st.markdown("**Memories included**")
        if trace["memories_used"]:
            for mm in trace["memories_used"]:
                st.write(f"- {mm[:140]}")
        else:
            st.write("_No recent memories used._")
        st.markdown("**Confidence (heuristic)**")
        if trace["confidence"] is not None:
            st.progress(trace["confidence"])
            st.write(f"{int(trace['confidence']*100)}% (heuristic)")
    else:
        st.write("After the next reply you'll see a short explainability trace.")

# -------------------------
# Life Timeline
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
        new_mem = st.text_area("Memory text", key="timeline_new_mem", height=120)
        if st.button("Save memory"):
            if new_mem.strip():
                mem = {"id": str(int(time.time()*1000)), "text": new_mem.strip(), "tags": [], "created_at": datetime.utcnow().isoformat()}
                data.setdefault("memories", []).append(mem)
                write_data(data)
                st.success("Memory saved.")
            else:
                st.warning("Cannot save empty memory.")

# -------------------------
# Vault
# -------------------------
elif page == "Vault":
    st.header("🔐 Private Vault (Prototype)")
    data = read_data()
    vp = st.text_input("Vault password (session only)", type="password", key="vault_pw_input")
    if vp:
        st.session_state["vault_pass"] = vp
        st.success("Vault password stored in session (prototype).")
    title = st.text_input("Title for note", key="vault_title")
    note = st.text_area("Private note", key="vault_note", height=140)
    if st.button("Save to vault"):
        if not st.session_state.get("vault_pass"):
            st.warning("Set a vault password in the field above.")
        elif not note.strip():
            st.warning("Cannot save empty note.")
        else:
            enc = xor_encrypt(note.strip(), st.session_state["vault_pass"])
            entry = {"id": str(int(time.time()*1000)), "title": title or "Untitled", "encrypted": enc, "created_at": datetime.utcnow().isoformat()}
            data.setdefault("vault", {}).setdefault("entries", []).append(entry)
            write_data(data)
            st.success("Saved to vault (XOR prototype).")
    st.markdown("---")
    st.subheader("Entries")
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

# -------------------------
# Legacy & Export
# -------------------------
elif page == "Legacy & Export":
    st.header("📜 Legacy & Export")
    data = read_data()
    st.download_button("⬇️ Download Data (JSON)", json.dumps(data, indent=2, ensure_ascii=False), file_name="echosoul_export.json")
    st.markdown("---")
    st.subheader("Legacy Snapshot")
    st.write("**Profile:**")
    st.write(data.get("profile", {}))
    st.write("**Recent memories:**")
    for m in data.get("memories", [])[-10:]:
        st.markdown(f"- {m.get('created_at','')} — {m.get('text')[:200]}")

# -------------------------
# Settings
# -------------------------
elif page == "Settings":
    st.header("⚙️ Settings")
    data = read_data()
    prof = data.get("profile", {})
    new_name = st.text_input("Display name", value=prof.get("name",""))
    new_tone = st.selectbox("Default tone", ["Adaptive","Friendly","Empathetic","Energetic","Formal"], index=0)
    voice_toggle = st.checkbox("Enable voice responses (prototype)", value=prof.get("voice_enabled", False))
    if st.button("Save settings"):
        prof["name"] = new_name
        prof["preferred_tone"] = new_tone
        prof["voice_enabled"] = voice_toggle
        data["profile"] = prof
        write_data(data)
        st.success("Settings saved.")

# -------------------------
# Footer / privacy note
# -------------------------
st.markdown("---")
st.markdown("<div class='tiny-muted'>Privacy: EchoSoul stores data locally in <code>echosoul_data.json</code>. Vault encryption is a demo (XOR) — not secure. Do not store production secrets unless you replace with proper encryption. The AI may be fallible; use the explainability trace to review answers.</div>", unsafe_allow_html=True)
