# app.py
import streamlit as st
import openai
import json
import os
import threading
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# ---------------------------
# Configuration & Constants
# ---------------------------
DATA_FILE = "echosoul_data.json"
VAULT_KEY_NAME = "vault_password"
MODEL_NAME = "gpt-4o-mini"  # per your spec (adjust if your account uses a different identifier)
DEFAULT_TEMP = 0.7

# UI color scheme (used with CSS variables)
LIGHT_BG = "#F6F7FB"
DARK_BG = "#0D0D2E"
NEON = "#3EE7A8"  # accent neon

# Ensure the OpenAI key is available from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    # If not deployed to Streamlit Cloud, try environment var (graceful fallback)
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------
# Utilities: File persistence
# ---------------------------
lock = threading.Lock()

def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        initial = {
            "profile": {"name": "You", "created_at": datetime.utcnow().isoformat()},
            "memories": [],  # list of {"id", "text", "tags", "created_at", "source"}
            "vault": {"entries": []},  # list of {"id","title","encrypted","created_at"}
            "conversations": [],  # historical chats
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2)

def read_data() -> Dict[str, Any]:
    ensure_datafile()
    with lock:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

def write_data(data: Dict[str, Any]):
    with lock:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------
# Vault (simple XOR prototype)
# ---------------------------
def xor_encrypt(text: str, password: str) -> str:
    if password is None or password == "":
        raise ValueError("Password required for vault encryption.")
    btxt = text.encode("utf-8")
    bpass = password.encode("utf-8")
    out = bytearray()
    for i, c in enumerate(btxt):
        out.append(c ^ bpass[i % len(bpass)])
    # store base64 for safe json
    return base64.b64encode(bytes(out)).decode("utf-8")

def xor_decrypt(b64text: str, password: str) -> str:
    b = base64.b64decode(b64text)
    bpass = password.encode("utf-8")
    out = bytearray()
    for i, c in enumerate(b):
        out.append(c ^ bpass[i % len(bpass)])
    return out.decode("utf-8")

# ---------------------------
# Simple sentiment heuristic
# ---------------------------
POS_WORDS = {"good", "great", "happy", "love", "awesome", "nice", "fantastic", "helpful", "yes"}
NEG_WORDS = {"sad", "bad", "angry", "hate", "frustrat", "upset", "no", "problem", "terrible"}

def simple_sentiment(text: str) -> str:
    txt = text.lower()
    pos = sum(txt.count(w) for w in POS_WORDS)
    neg = sum(txt.count(w) for w in NEG_WORDS)
    if pos > neg + 1:
        return "positive"
    if neg > pos + 1:
        return "negative"
    return "neutral"

# ---------------------------
# Explainability heuristic: "confidence"
# ---------------------------
def confidence_heuristic(ai_text: str) -> float:
    # Very simple heuristic. Real confidence needs model-level probabilities.
    # Lower confidence if hedging words appear or many question marks.
    hedges = ["might", "could", "maybe", "possibly", "it depends", "sometimes", "I think", "I believe"]
    score = 0.8
    lower = ai_text.lower()
    for h in hedges:
        if h in lower:
            score -= 0.12
    qmarks = lower.count("?")
    score -= min(0.15, 0.03 * qmarks)
    # penalty for "I don't know" style
    for phrase in ["i don't know", "i'm not sure", "i do not know"]:
        if phrase in lower:
            score -= 0.3
    # clamp
    return max(0.05, min(0.99, round(score, 2)))

# ---------------------------
# OpenAI call wrapper
# ---------------------------
def call_ai(prompt: str, system_role: str = "You are EchoSoul, a helpful AI.", memory_snippets: List[str] = [], roleplay_as_user: bool = False, temperature: float = DEFAULT_TEMP) -> Dict[str, Any]:
    """
    Sends prompt + minimal context to OpenAI and returns {'text','raw','usage','elapsed_seconds'}
    """
    start = time.time()
    # Build messages with memory trace for explainability
    messages = [{"role": "system", "content": system_role}]
    # Add memory context - keep it short (last N)
    if memory_snippets:
        mem_content = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memory_snippets[-6:])
        messages.append({"role": "system", "content": mem_content})
    # Roleplay: if true, instruct the model to speak as the user-provided persona
    if roleplay_as_user:
        messages.append({"role": "system", "content": "Roleplay: reply in the voice/persona of the user. Be clear when you are roleplaying."})
    messages.append({"role": "user", "content": prompt})

    # Try to call OpenAI; if fails, return friendly error object
    try:
        # Using standard OpenAI library interface
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=800,
            top_p=1,
        )
        text = response.choices[0].message["content"]
        raw = response.to_dict() if hasattr(response, "to_dict") else response
        elapsed = time.time() - start
        return {"text": text, "raw": raw, "usage": getattr(response, "usage", {}), "elapsed": elapsed}
    except Exception as e:
        # Helpful fallback message
        elapsed = time.time() - start
        return {"text": f"⚠️ AI call failed: {str(e)}", "raw": None, "usage": {}, "elapsed": elapsed, "error": True}

# ---------------------------
# UI Helpers
# ---------------------------

def inject_css():
    # CSS for neon accent, floating call button, glassmorphism cards
    st.markdown(
        f"""
    <style>
    :root {{
      --bg: {DARK_BG};
      --fg: #F0F0F0;
      --muted: #B8C0D0;
      --accent: {NEON};
    }}
    /* Floating call button top-right */
    .call-ai-btn {{
      position: fixed;
      top: 18px;
      right: 18px;
      z-index: 9999;
      background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      border: 1px solid rgba(255,255,255,0.06);
      padding: 10px 14px;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
      color: var(--fg);
      cursor: pointer;
      font-weight: 600;
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .neon {{
      color: var(--accent);
      text-shadow: 0 0 10px rgba(62,231,168,0.18);
    }}
    .glass {{
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      padding: 14px;
      border: 1px solid rgba(255,255,255,0.04);
      backdrop-filter: blur(6px);
    }}
    .tiny-muted {{ color: var(--muted); font-size: 12px; }}
    /* tooltip style for explainability icons */
    .tooltip {{
      border-bottom: 1px dotted var(--muted);
      cursor: help;
    }}
    /* Make dark and light theme controlled via body class */
    </style>
    """,
        unsafe_allow_html=True,
    )

def floating_call_button():
    # A clickable HTML button that triggers a Streamlit callback (via query params hack)
    st.markdown(
        """
        <div class="call-ai-btn glass" onclick="window.dispatchEvent(new CustomEvent('call-ai-click'))">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" style="margin-right:4px">
            <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.8 19.8 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.8 19.8 0 0 1-3.07-8.63A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.12 1.21.45 2.39.97 3.49a2 2 0 0 1-.45 2.11L9.91 10.09a16 16 0 0 0 6 6l.76-1.76a2 2 0 0 1 2.11-.45c1.1.52 2.28.85 3.49.97A2 2 0 0 1 22 16.92z"></path>
          </svg>
          <div>
            <div style="font-size:13px">Call AI</div>
            <div class="tiny-muted" style="font-size:11px">Quick chat</div>
          </div>
        </div>
        <script>
          // forward the call-ai-click event into Streamlit by changing location.hash (simple hack)
          window.addEventListener('call-ai-click', () => {
            const timestamp = Date.now()
            window.location.hash = 'call-ai-' + timestamp
          })
        </script>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="EchoSoul — Personal AI Companion", layout="wide", initial_sidebar_state="expanded")
inject_css()

# Theme toggle - top-level
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Apply theme CSS changes
if st.session_state["theme"] == "dark":
    st.markdown(f"<style>body{{background: {DARK_BG}; color: #F0F0F0}}</style>", unsafe_allow_html=True)
else:
    st.markdown(f"<style>body{{background: {LIGHT_BG}; color: #0D0D2E}}</style>", unsafe_allow_html=True)

# Sidebar Layout
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px'>"
                f"<div style='width:48px;height:48px;border-radius:12px;background:{NEON};box-shadow:0 6px 14px rgba(62,231,168,0.12)'></div>"
                "<div><h3 style='margin:0'>EchoSoul</h3><div class='tiny-muted'>Personal AI companion</div></div>"
                "</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Profile settings
    st.header("Profile")
    data = read_data()
    profile = data.get("profile", {"name":"You"})
    name = st.text_input("Display name", value=profile.get("name","You"), key="profile_name")
    if st.button("Save profile"):
        data["profile"]["name"] = name
        write_data(data)
        st.success("Saved profile.")

    st.markdown("---")
    st.header("Quick Actions")
    if st.button("New Conversation (+)"):
        st.session_state["conversation"] = []
        st.success("New conversation started.")
    st.write("Search memories")
    mem_query = st.text_input("Find memory...", key="mem_search")
    if mem_query:
        matches = [m for m in data.get("memories", []) if mem_query.lower() in m.get("text","").lower()][:10]
        for m in matches:
            st.markdown(f"- **{m.get('created_at','')}** — {m.get('text')[:120]}")
    st.markdown("---")

    st.header("Mode")
    mode = st.radio("AI Mode", ["All", "Forums", "Short videos", "Images", "More"], index=0)
    st.markdown("---")
    st.header("Appearance")
    theme_toggle = st.radio("Theme", ["dark", "light"], index=0 if st.session_state["theme"]=="dark" else 1)
    if theme_toggle != st.session_state["theme"]:
        st.session_state["theme"] = theme_toggle
        st.experimental_rerun()

    st.markdown("---")
    st.header("Vault (Prototype)")
    vault_pass = st.text_input("Set/enter vault password", type="password", key="vault_pass")
    if st.button("Save vault password"):
        # we won't store password plaintext; we keep in session for prototype
        st.session_state["vault_pass"] = vault_pass
        st.success("Vault password saved to session (for this demo only).")

    st.markdown("---")
    st.header("Deployment")
    st.write("This app is intended for Streamlit Cloud. Keep your OpenAI key in Streamlit Secrets.")
    st.markdown("**Export / Legacy**")
    if st.button("Export all data (JSON)"):
        data = read_data()
        b = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("Download JSON", b, file_name="echosoul_export.json")
    st.markdown("---")
    st.markdown("Need help? Click the question mark at the top right.")

# Top-right floating call button
floating_call_button()

# Main columns
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("<div style='display:flex;align-items:center;gap:12px'>"
                f"<h2 style='margin:0'>Hello, {st.session_state.get('profile_name', profile.get('name','You'))} 👋</h2>"
                f"<div class='tiny-muted'>EchoSoul — your personal AI</div>"
                "</div>", unsafe_allow_html=True)

    st.markdown("### Chat / Conversation")
    # Conversation area (scrollable)
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    # Show conversation messages
    for msg in st.session_state["conversation"]:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("ts", "")
        if role == "user":
            st.markdown(f"<div class='glass' style='margin-bottom:8px'><strong>{st.session_state.get('profile_name','You')}</strong> <span class='tiny-muted'>• {ts}</span><div style='margin-top:6px'>{content}</div></div>", unsafe_allow_html=True)
        else:
            # AI reply card with confidence bar
            conf = msg.get("confidence", None)
            conf_html = ""
            if conf is not None:
                conf_html = f"<div class='tiny-muted' style='margin-top:6px'>Confidence: <span style='color:{NEON}; font-weight:700'>{int(conf*100)}%</span></div>"
            st.markdown(f"<div class='glass' style='margin-bottom:8px;'><strong class='neon'>EchoSoul</strong> <span class='tiny-muted'>• {ts}</span><div style='margin-top:6px'>{content}</div>{conf_html}</div>", unsafe_allow_html=True)

    # Input form (clears automatically after send)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message", key="user_input", placeholder="Type to EchoSoul... (press Send)", height=120)
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            roleplay = st.checkbox("Roleplay as me", value=False, help="EchoSoul will answer in your voice/persona.")
        with col2:
            persona = st.selectbox("Tone", ["Adaptive (auto)", "Friendly", "Empathetic", "Energetic", "Formal"], index=0)
        with col3:
            temp = st.slider("Creativity", min_value=0.0, max_value=1.0, value=DEFAULT_TEMP, step=0.05)
        submitted = st.form_submit_button("Send")

    # After sending
    if submitted and user_input.strip() != "":
        # Save the user's message to conversation
        tiso = datetime.utcnow().isoformat()
        st.session_state["conversation"].append({"role":"user", "content":user_input, "ts":tiso})
        # Basic sentiment
        sentiment = simple_sentiment(user_input)
        # Compose system role based on persona
        if persona == "Adaptive (auto)":
            if sentiment == "positive":
                tone_instruction = "Respond in an energetic and uplifting tone."
            elif sentiment == "negative":
                tone_instruction = "Respond in an empathetic, calming tone."
            else:
                tone_instruction = "Respond in a friendly, neutral tone."
        else:
            tone_instruction = f"Respond in a {persona.lower()} tone."

        system_role = f"You are EchoSoul, a helpful AI companion. {tone_instruction} Be transparent about limitations, and show a short confidence estimate and trace of which memories you used."

        # Load memories and decide which to pass (simple relevance)
        data = read_data()
        memories = data.get("memories", [])
        # naive relevance: contains any overlapping word
        user_words = set([w.lower().strip(".,!?") for w in user_input.split() if len(w)>2])
        relevant = []
        for m in memories:
            text = m.get("text","")
            if user_words & set(text.lower().split()):
                relevant.append(text)
        # Always attach last 3 memories for context if any
        memory_snippets = relevant[-3:] if relevant else [m['text'] for m in memories[-3:]]

        # Call AI
        ai_result = call_ai(user_input, system_role=system_role, memory_snippets=memory_snippets, roleplay_as_user=roleplay, temperature=temp)
        ai_text = ai_result.get("text", "")
        is_error = ai_result.get("error", False)

        # Compute confidence heuristic
        conf = None
        if not is_error:
            conf = confidence_heuristic(ai_text)

        # Append AI reply to conversation
        st.session_state["conversation"].append({
            "role": "ai",
            "content": ai_text,
            "ts": datetime.utcnow().isoformat(),
            "confidence": conf
        })

        # Save conversation and memory suggestion
        # Auto-suggest saving as memory if user marked it or the AI suggests it
        data = read_data()
        data.setdefault("conversations", []).append({
            "started_at": datetime.utcnow().isoformat(),
            "messages": st.session_state["conversation"][-6:],  # last few
            "ai_meta": {"model": MODEL_NAME, "temp": temp, "elapsed": ai_result.get("elapsed",0)}
        })
        write_data(data)

        # Show XAI trace on the right column by updating session_state
        st.session_state["last_ai"] = {
            "prompt_used": user_input,
            "memory_snippets": memory_snippets,
            "model": MODEL_NAME,
            "temp": temp,
            "confidence": conf,
            "raw": ai_result.get("raw")
        }

# Right column: explainability, timeline, vault quick view
with right_col:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("Explainable AI (XAI)")
    lai = st.session_state.get("last_ai", None)
    if lai:
        st.markdown(f"**Model**: `{lai['model']}`  ")
        st.markdown(f"**Temperature**: {lai['temp']}  ")
        st.markdown("**Prompt (what you sent)**")
        st.code(lai["prompt_used"][:400], language=None)
        st.markdown("**Memories included in context**")
        if lai["memory_snippets"]:
            for mem in lai["memory_snippets"]:
                st.write(f"- {mem[:140]}")
        else:
            st.write("_No memories used for this reply._")
        st.markdown("**Confidence (heuristic)**")
        conf = lai.get("confidence")
        if conf:
            st.progress(conf)
            st.write(f"{int(conf*100)}% (heuristic estimate)")
        else:
            st.write("—")
        st.markdown("**Why the model answered this way**")
        st.write("We include the prompt, a short memory trace, and the temperature to help you understand the AI's reasoning. You can edit the AI response below to keep or refine it.")
        st.markdown("---")
        # Show raw metadata toggle
        if st.checkbox("Show raw AI metadata (debug)"):
            st.write(lai.get("raw"))

        st.markdown("---")
    else:
        st.write("After you send a message, EchoSoul will show a short explainability trace here.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Life Timeline")
    data = read_data()
    memories = data.get("memories", [])
    if st.button("Add memory from last chat"):
        # Add last user message as a memory (if any)
        if st.session_state["conversation"]:
            last_user = next((m for m in reversed(st.session_state["conversation"]) if m["role"]=="user"), None)
            if last_user:
                mem = {"id": str(int(time.time()*1000)), "text": last_user["content"], "tags": [], "created_at": datetime.utcnow().isoformat(), "source":"chat"}
                data.setdefault("memories", []).append(mem)
                write_data(data)
                st.success("Memory added.")
            else:
                st.info("No user message found in the conversation.")
        else:
            st.info("No conversation in memory.")
    # List memories
    for m in sorted(memories, key=lambda x: x.get("created_at",""), reverse=True)[:8]:
        st.markdown(f"- **{m.get('created_at','')[:10]}** — {m.get('text')[:120]}")
        if st.button(f"View / Edit {m.get('id')}", key=f"viewmem-{m.get('id')}"):
            # Editing modal: show text_area and save
           
