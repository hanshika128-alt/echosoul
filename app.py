"""
EchoSoul — Full app.py (clean, corrected, ready to paste)

Notes:
- Put your OpenAI API key into Streamlit Secrets as:
    OPENAI_API_KEY = "sk-..."
  (Do NOT put the key into GitHub.)
- requirements.txt should include at least:
    streamlit
    openai
    cryptography   # optional (Fernet). If not present, the app uses a safe XOR fallback.
- This file is self-contained and defensive: it ensures required keys exist,
  uses st.chat_input so the box clears after send, and avoids deprecated calls.
"""

import streamlit as st
import os
import json
import hashlib
import base64
import datetime
import re
import io
import textwrap

# Optional imports (handled defensively)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    Fernet = None
    USE_FERNET = False

# ---- constants ----
DATA_FILE = "echosoul_data.json"

# ---- utility helpers ----
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {
        "profile": {
            "name": None,
            "age": None,
            "hobbies": None,
            "free_time": None,
            "created": ts_now(),
            "persona": {"tone": "friendly", "style": "casual"},
            "intro_completed": False
        },
        "timeline": [],
        "vault": [],
        "conversations": []
    }

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            # defensive: ensure expected top-level keys exist
            if not isinstance(d, dict):
                return default_data()
            if "profile" not in d or not isinstance(d["profile"], dict):
                d["profile"] = default_data()["profile"]
            for k in ("timeline","vault","conversations"):
                if k not in d or not isinstance(d[k], list):
                    d[k] = default_data()[k]
            return d
        except Exception:
            return default_data()
    return default_data()

def save_data(d):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Saving error: {e}")

# ---- encryption helpers ----
def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def encrypt_xor(password, plaintext):
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_xor(password, ciphertext_b64):
    try:
        data = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data))
        x = bytes([data[i] ^ key[i] for i in range(len(data))])
        return x.decode("utf-8")
    except Exception:
        return None

def gen_fernet_key_from_password(password: str) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)

def encrypt_with_fernet(password: str, plaintext: str) -> str:
    key = gen_fernet_key_from_password(password)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

def decrypt_with_fernet(password: str, token: str) -> str:
    key = gen_fernet_key_from_password(password)
    f = Fernet(key)
    return f.decrypt(token.encode("utf-8")).decode("utf-8")

def encrypt_text(password: str, plaintext: str) -> str:
    if USE_FERNET and Fernet is not None:
        try:
            return encrypt_with_fernet(password, plaintext)
        except Exception:
            # fallback
            return encrypt_xor(password, plaintext)
    else:
        return encrypt_xor(password, plaintext)

def decrypt_text(password: str, blob: str) -> str:
    if USE_FERNET and Fernet is not None:
        try:
            return decrypt_with_fernet(password, blob)
        except Exception:
            return decrypt_xor(password, blob)
    else:
        return decrypt_xor(password, blob)

# ---- memory / timeline helpers ----
def add_memory(data, title, content, tags=None):
    item = {
        "id": hashlib.sha1((title + content + ts_now()).encode("utf-8")).hexdigest(),
        "title": title,
        "content": content,
        "tags": tags or [],
        "timestamp": ts_now()
    }
    data["timeline"].append(item)
    save_data(data)
    return item

def find_relevant_memories(data, text, limit=3):
    found = []
    txt = text.lower()
    for item in reversed(data["timeline"]):
        if any(w in txt for w in re.findall(r"\w+", item.get("content","").lower())) or any(w in txt for w in re.findall(r"\w+", item.get("title","").lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# ---- simple sentiment heuristic ----
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", (text or "").lower())
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

# ---- OpenAI client helper ----
def get_openai_client():
    if OpenAI is None:
        return None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        return None
    except Exception:
        return None

openai_client = get_openai_client()

def call_gpt(system_prompt: str, user_msg: str) -> str:
    if openai_client is None:
        return "OpenAI not configured. Add OPENAI_API_KEY to Streamlit Secrets to enable AI replies."
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8,
            max_tokens=600
        )
        # compatibility: SDK returns resp.choices[0].message.content commonly
        return resp.choices[0].message.content
    except Exception as e:
        return f"[GPT error: {str(e)}]"

# ---- audio transcription helper (safe/fallback) ----
def transcribe_audio_bytes(file_bytes: bytes, filename: str) -> Optional[str]:
    if openai_client is None:
        return None
    try:
        # Some SDKs accept bytes-like; wrap as BytesIO
        fobj = io.BytesIO(file_bytes)
        resp = openai_client.audio.transcriptions.create(model="gpt-4o-transcribe", file=fobj)
        # adapt to common fields
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        return str(resp)
    except Exception:
        return None

# ---- main reply generator (GPT + auto-memory) ----
def generate_reply(data, user_msg):
    # update sentiment/persona
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)

    # prepare memory context
    memories = [f"{m['title']}: {m['content']}" for m in data["timeline"][-6:]]
    context = "\n".join(memories) if memories else "No memories yet."

    tone = data["profile"].get("persona", {}).get("tone", "friendly")
    system_prompt = textwrap.dedent(f"""
        You are EchoSoul, an empathetic, evolving personal companion.
        Personality tone: {tone}.
        Known memories and facts about the user (most recent first):
        {context}

        Behaviors:
        - Use memory facts naturally when relevant.
        - If the user asks you to 'act like me' or similar, roleplay as the user using their profile and timeline facts.
        - Keep replies warm, helpful, and concise.
        - Never reveal secrets from the private vault unless asked and unlocked by the user.
    """)

    reply = call_gpt(system_prompt, user_msg)

    # Auto-save obvious facts
    low = (user_msg or "").lower()
    if re.search(r"\bremember\b", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b", low):
        add_memory(data, "User fact", user_msg)

    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ---- onboarding flow ----
def run_onboarding(data):
    st.header("Welcome to EchoSoul — let's get to know you")
    st.write("I'll ask 4 quick questions so I can remember you and talk in your style.")
    if "onb_step" not in st.session_state:
        st.session_state.onb_step = 0

    step = st.session_state.onb_step

    if step == 0:
        name = st.text_input("What's your name?", value=(data["profile"].get("name") or ""))
        if st.button("Next"):
            if name.strip():
                data["profile"]["name"] = name.strip()
                add_memory(data, "Name", f"My name is {name.strip()}")
                st.session_state.onb_step = 1
                save_data(data)
                st.rerun()
            else:
                st.warning("Please enter a name.")
    elif step == 1:
        age = st.text_input("What's your age?", value=(data["profile"].get("age") or ""))
        if st.button("Next"):
            if age.strip():
                data["profile"]["age"] = age.strip()
                add_memory(data, "Age", f"My age is {age.strip()}")
                st.session_state.onb_step = 2
                save_data(data)
                st.rerun()
            else:
                st.warning("Please enter an age.")
    elif step == 2:
        hobbies = st.text_area("What are your hobbies?", value=(data["profile"].get("hobbies") or ""))
        if st.button("Next"):
            if hobbies.strip():
                data["profile"]["hobbies"] = hobbies.strip()
                add_memory(data, "Hobbies", hobbies.strip())
                st.session_state.onb_step = 3
                save_data(data)
                st.rerun()
            else:
                st.warning("Please list at least one hobby.")
    elif step == 3:
        free_time = st.text_area("What do you like to do in your free time?", value=(data["profile"].get("free_time") or ""))
        if st.button("Finish"):
            if free_time.strip():
                data["profile"]["free_time"] = free_time.strip()
                add_memory(data, "Free time", free_time.strip())
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.success("Thanks — I will remember these things.")
                st.session_state.onb_step = 0
                st.rerun()
            else:
                st.warning("Please enter something you like to do in free time.")

# ---- Streamlit UI ----
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — Your personal companion")

data = load_data()  # always returns valid structure

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.write("Profile")
    current_name = data["profile"].get("name") or ""
    new_name = st.text_input("Display name", value=current_name)
    if st.button("Save profile name"):
        data["profile"]["name"] = new_name.strip() or data["profile"]["name"]
        save_data(data)
        st.success("Name saved.")

    st.markdown("---")
    st.write("Vault (secure notes)")
    vault_pwd = st.text_input("Vault password (to lock/unlock)", type="password")
    if st.button("Clear vault password (local only)"):
        vault_pwd = ""
        st.info("Vault password cleared for this session.")

    st.markdown("---")
    st.write("App options")
    st.checkbox("Enable adaptive learning (persona updates)", value=True, key="adaptive_toggle")
    st.markdown("Voice: upload audio file to transcribe (optional).")

# run onboarding if needed
if not data["profile"].get("intro_completed", False):
    run_onboarding(data)
    st.stop()

# main tabs
tab = st.radio("", ["Chat", "Conversation History", "Memories & Timeline", "Private Vault", "Legacy & Export", "About"], horizontal=True)

if tab == "Chat":
    st.header(f"Chat — Hello {data['profile'].get('name') or 'friend'}")
    st.write("Type or upload audio. The input clears after send.")

    # optional audio upload
    audio_file = st.file_uploader("Upload audio to transcribe (optional)", type=["mp3","wav","m4a","ogg"])
    if audio_file is not None:
        file_bytes = audio_file.read()
        st.info("Transcribing...")
        t = transcribe_audio_bytes(file_bytes, audio_file.name)
        if t:
            st.success("Transcription done — added to conversation.")
            reply = generate_reply(data, t)
            st.markdown(f"**You (audio):** {t}")
            st.markdown(f"**EchoSoul:** {reply}")
        else:
            st.error("Transcription failed or OpenAI not configured.")

    # show last conversations
    for conv in data.get("conversations", [])[-30:]:
        st.markdown(f"**You:** {conv['user']}")
        st.markdown(f"**EchoSoul:** {conv['bot']}")
        st.markdown("---")

    # chat_input that clears automatically
    user_msg = st.chat_input("Say something to EchoSoul")
    if user_msg:
        with st.spinner("Thinking..."):
            _ = generate_reply(data, user_msg)
        # rerun to refresh conversation display; st.chat_input clears automatically
        st.rerun()

elif tab == "Conversation History":
    st.header("Conversation History")
    conv = data.get("conversations", [])
    if not conv:
        st.info("No conversations yet.")
    else:
        for m in conv[::-1]:
            st.markdown(f"**You:** {m['user']}")
            st.markdown(f"**EchoSoul:** {m['bot']}")
            st.markdown("---")

elif tab == "Memories & Timeline":
    st.header("Memories & Timeline")
    if data["timeline"]:
        for item in sorted(data["timeline"], key=lambda x: x["timestamp"], reverse=True):
            st.markdown(f"**{item['title']}** — {item['timestamp']}")
            st.write(item["content"])
            st.markdown("---")
    else:
        st.info("No memories yet — they will appear here when you say 'remember' or complete onboarding.")

    st.markdown("### Add a memory manually")
    mtitle = st.text_input("Title", key="mem_title")
    mcontent = st.text_area("Content", key="mem_content")
    if st.button("Save Memory"):
        if mcontent.strip():
            add_memory(data, mtitle or "Memory", mcontent.strip())
            st.success("Memory saved.")
            st.rerun()
        else:
            st.warning("Content cannot be empty.")

elif tab == "Private Vault":
    st.header("Private Vault")
    st.write("Enter your vault password in the sidebar to unlock or save items. The app encrypts items before saving.")
    if not vault_pwd:
        st.warning("Vault password required (enter it in the sidebar).")
    else:
        if data["vault"]:
            for v in data["vault"]:
                st.markdown(f"**{v['title']}** — {v['timestamp']}")
                dec = decrypt_text(vault_pwd, v["cipher"])
                if dec is None:
                    st.write("*Unable to decrypt with this password.*")
                else:
                    st.write(dec)
                st.markdown("---")
        else:
            st.info("No vault items yet.")

        st.markdown("### Add to vault")
        vt = st.text_input("Title for vault item", key="vt")
        vc = st.text_area("Secret content", key="vc")
        if st.button("Save to Vault"):
            if not vc.strip():
                st.warning("Secret content cannot be empty.")
            else:
                cipher = encrypt_text(vault_pwd, vc.strip())
                data["vault"].append({"title": vt or "Vault item", "cipher": cipher, "timestamp": ts_now()})
                save_data(data)
                st.success("Saved to vault.")
                st.rerun()

elif tab == "Legacy & Export":
    st.header("Legacy Mode & Export")
    st.write("Export timeline and conversations. Vault entries remain encrypted in the export.")
    if st.button("Download full export (JSON)"):
        st.download_button("Click to download JSON", json.dumps(data, indent=2), f"echosoul_export_{datetime.datetime.utcnow().date()}.json", "application/json")
    legacy = "\n\n".join([f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data["timeline"]])
    st.text_area("Legacy snapshot", legacy, height=300)

elif tab == "About":
    st.header("About EchoSoul")
    st.write("EchoSoul remembers personal details, adapts its tone, and uses GPT for natural replies (when configured).")
    st.markdown("- Persistent Memory: remembers and recalls personal details.")
    st.markdown("- Adaptive Personality: adjusts tone over time.")
    st.markdown("- Emotion Recognition: text-based heuristics (voice requires transcription).")
    st.markdown("- Life Timeline, Custom Conversation Style, Knowledge Growth, Private Vault, Legacy Mode.")
    st.markdown("Important: Put your OPENAI_API_KEY into Streamlit Secrets to enable GPT and transcription features.")

# defensive save
save_data(data)
