"""
EchoSoul — Full app.py (complete)

Features:
- Onboarding questions at first-run (name, age, hobbies, free-time)
- Persistent timeline memory (saved to echosoul_data.json)
- Adaptive persona tone
- GPT-based replies (OpenAI API key must be in Streamlit secrets)
- Chat input auto-clears after send (st.chat_input)
- Auto memory saving for "remember" or personal facts
- Private Vault with Fernet encryption if 'cryptography' installed (secure), fallback to XOR based demo
- Voice upload (audio file) transcription via OpenAI (if enabled)
- Legacy export and timeline viewer
"""

import streamlit as st
import os, json, hashlib, base64, datetime, re, io, textwrap
from typing import Optional

# ---- OpenAI client (reads key from Streamlit secrets) ----
# You must add OPENAI_API_KEY to Streamlit Secrets before running
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    client = None
    # We'll still allow offline testing, but GPT calls will fail with a clear message.

# ---- Optional strong encryption ----
USE_FERNET = False
try:
    from cryptography.fernet import Fernet
    USE_FERNET = True
except Exception:
    USE_FERNET = False

# ---- Storage file ----
DATA_FILE = "echosoul_data.json"

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
                return json.load(f)
        except Exception:
            return default_data()
    return default_data()

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---- Encryption helpers ----
def gen_fernet_key_from_password(password: str) -> bytes:
    # Derive a deterministic Fernet key from password (not PBKDF2 here to keep simple).
    # For production, use a proper KDF (PBKDF2HMAC with salt). This is adequate for prototype.
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)

def encrypt_with_fernet(password: str, plaintext: str) -> str:
    key = gen_fernet_key_from_password(password)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

def decrypt_with_fernet(password: str, token: str) -> Optional[str]:
    try:
        key = gen_fernet_key_from_password(password)
        f = Fernet(key)
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception:
        return None

# XOR fallback (demo only)
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

def encrypt_text(password, plaintext):
    if USE_FERNET:
        return encrypt_with_fernet(password, plaintext)
    return encrypt_xor(password, plaintext)

def decrypt_text(password, blob):
    if USE_FERNET:
        return decrypt_with_fernet(password, blob)
    return decrypt_xor(password, blob)

# ---- Memories timeline helpers ----
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
        if any(w in txt for w in re.findall(r"\w+", item["content"].lower())) or any(w in txt for w in re.findall(r"\w+", item["title"].lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# ---- Simple text-based emotion detection (heuristic) ----
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted calm optimistic".split())
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

# ---- GPT integration ----
def call_gpt(system_prompt: str, user_msg: str) -> str:
    if client is None:
        return "OpenAI client not configured. Add your OPENAI_API_KEY to Streamlit Secrets."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8,
            max_tokens=600
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[GPT error: {str(e)}]"

def transcribe_audio_file(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    Transcribe uploaded audio using OpenAI whisper endpoint via the SDK.
    If OpenAI client isn't available or transcription call fails, returns None.
    """
    if client is None:
        return None
    try:
        # The SDK expects a file-like object. We use bytes -> io.BytesIO
        file_obj = io.BytesIO(file_bytes)
        # sdk-style call (may vary across SDK versions). Wrap in try/except.
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_obj)
        # Many SDKs return a .text or .transcript field; try common ones safely:
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        # fallback: string conversion
        return str(resp)
    except Exception:
        return None

# ---- Main GPT reply generator with auto-memory ----
def generate_reply(data, user_msg):
    # sentiment and persona update
    score = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, score)

    # memory context
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-6:]]
    context = "\n".join(memories) if memories else "No memories yet."

    tone = data["profile"]["persona"].get("tone", "friendly")

    system_prompt = textwrap.dedent(f"""
    You are EchoSoul, an empathetic, evolving personal companion for the user.
    Personality tone: {tone}.
    Known memories and facts about the user:
    {context}

    Behavior:
    - Use the memory facts naturally in replies when relevant.
    - If the user asks "act like me" or similar, roleplay as the user using their profile and timeline facts.
    - Keep responses helpful, warm, and concise when asked direct questions.
    - Do not expose private data unless the user explicitly asks to see it.
    """)

    reply = call_gpt(system_prompt, user_msg)

    # Auto-save obvious facts:
    low = user_msg.lower()
    # explicit "remember X" or "remember that ..." or "my name is ..."
    if re.search(r"\bremember\b", low) or re.search(r"\bmy name is\b", low) or re.search(r"\bi am\b", low):
        # Save as a user fact memory
        add_memory(data, "User fact", user_msg)

    # Save the conversation
    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ---- Onboarding UI ----
def run_onboarding(data):
    st.header("Welcome to EchoSoul — let's get to know you")
    st.write("I'll ask a few quick questions so I can remember you and speak like you.")
    # use session_state to store answers temporarily
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
                st.experimental_rerun()
            else:
                st.warning("Please enter your name.")
    elif step == 1:
        age = st.text_input("What's your age?", value=(data["profile"].get("age") or ""))
        if st.button("Next"):
            if age.strip():
                data["profile"]["age"] = age.strip()
                add_memory(data, "Age", f"My age is {age.strip()}")
                st.session_state.onb_step = 2
                save_data(data)
                st.experimental_rerun()
            else:
                st.warning("Please enter your age.")
    elif step == 2:
        hobbies = st.text_area("What are your hobbies? (comma separated)", value=(data["profile"].get("hobbies") or ""))
        if st.button("Next"):
            if hobbies.strip():
                data["profile"]["hobbies"] = hobbies.strip()
                add_memory(data, "Hobbies", hobbies.strip())
                st.session_state.onb_step = 3
                save_data(data)
                st.experimental_rerun()
            else:
                st.warning("Please enter at least one hobby.")
    elif step == 3:
        free_time = st.text_area("What do you like to do in your free time?", value=(data["profile"].get("free_time") or ""))
        if st.button("Finish and continue"):
            if free_time.strip():
                data["profile"]["free_time"] = free_time.strip()
                add_memory(data, "Free time", free_time.strip())
                data["profile"]["intro_completed"] = True
                save_data(data)
                st.success("Thanks — I will remember these things.")
                # reset step for future onboarding calls
                st.session_state.onb_step = 0
                st.experimental_rerun()
            else:
                st.warning("Please tell me something you like doing in your free time.")

# ---- Streamlit layout ----
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — Your personal companion")

data = load_data()

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    st.write("Profile")
    pname = data["profile"].get("name") or ""
    new_name = st.text_input("Display name", value=pname)
    if st.button("Save profile name"):
        data["profile"]["name"] = new_name.strip() or data["profile"]["name"]
        save_data(data)
        st.success("Name saved.")

    st.markdown("---")
    st.write("Vault (secure notes)")
    vault_password = st.text_input("Vault password (to lock/unlock)", type="password")
    if st.button("Clear vault password (local only)"):
        vault_password = ""
        st.info("Vault password cleared (this only affects this session).")

    st.markdown("---")
    st.write("App options")
    st.checkbox("Enable adaptive learning (persona updates from mood)", value=True, key="adaptive_toggle")
    st.markdown("Voice & text: you can upload an audio file to transcribe (optional).")

# If onboarding not done, run onboarding flow
if not data["profile"].get("intro_completed", False):
    run_onboarding(data)
    st.stop()  # stop further UI until onboarding complete

# Main UI tabs
tabs = st.radio("", ["Chat", "Conversation History", "Memories & Timeline", "Private Vault", "Legacy & Export", "About"], horizontal=True)

if tabs == "Chat":
    st.header(f"Chat with EchoSoul — Hello {data['profile'].get('name','friend')}")
    st.write("You can type or upload an audio file. The chat input clears after sending.")

    # --- audio upload area (optional) ---
    audio_file = st.file_uploader("Upload audio (optional) — I will try to transcribe it", type=["mp3","wav","m4a","wav","ogg"])
    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.info("Transcribing audio...")
        transcription = transcribe_audio_file(audio_bytes, audio_file.name)
        if transcription:
            st.success("Transcription complete. It has been added to chat.")
            # add transcription as incoming user message
            reply = generate_reply(data, transcription)
            st.markdown(f"**You (transcribed):** {transcription}")
            st.markdown(f"**EchoSoul:** {reply}")
        else:
            st.error("Transcription failed or OpenAI not configured.")

    # Show last 20 messages
    conv = data.get("conversations", [])[-20:]
    for message in conv:
        st.markdown(f"**You:** {message['user']}")
        st.markdown(f"**EchoSoul:** {message['bot']}")
        st.markdown("---")

    # Chat input that clears after send
    user_input = st.chat_input("Say something to EchoSoul")
    if user_input:
        with st.spinner("EchoSoul is thinking..."):
            reply = generate_reply(data, user_input)
        # the chat_input auto-clears; show the reply immediately
        st.experimental_rerun()

elif tabs == "Conversation History":
    st.header("Conversation History")
    conv = data.get("conversations", [])
    if not conv:
        st.info("No conversations yet.")
    else:
        for m in conv[::-1]:
            st.markdown(f"**You:** {m['user']}")
            st.markdown(f"**EchoSoul:** {m['bot']}")
            st.markdown("---")

elif tabs == "Memories & Timeline":
    st.header("Memories & Timeline")
    if data["timeline"]:
        for item in sorted(data["timeline"], key=lambda x: x["timestamp"], reverse=True):
            st.markdown(f"**{item['title']}** — {item['timestamp']}")
            st.write(item["content"])
            st.markdown("---")
    else:
        st.info("No memories yet. They will appear here as you chat and when you say 'remember ...'.")

    st.markdown("### Add a memory manually")
    mt = st.text_input("Title", key="mem_title")
    mc = st.text_area("Content", key="mem_content")
    if st.button("Save Memory"):
        if mc.strip():
            add_memory(data, mt or "Memory", mc.strip())
            st.success("Memory saved.")
            st.experimental_rerun()
        else:
            st.warning("Please add some content.")

elif tabs == "Private Vault":
    st.header("Private Vault")
    st.write("Store sensitive notes. Use a strong vault password; data is encrypted before saving.")
    st.write("If cryptography is installed, Fernet is used (strong). Otherwise a demo XOR fallback is used (less secure).")
    if not vault_password:
        st.warning("Enter your vault password in the sidebar to unlock or save items.")
    else:
        # show items
        if data["vault"]:
            for v in data["vault"]:
                st.markdown(f"**{v['title']}** — {v['timestamp']}")
                dec = decrypt_text(vault_password, v["cipher"])
                if dec is None:
                    st.write("*Unable to decrypt with this password.*")
                else:
                    st.write(dec)
                st.markdown("---")
        else:
            st.info("No vault items yet.")

        st.markdown("### Add to vault")
        vtitle = st.text_input("Title for vault", key="vault_title")
        vcontent = st.text_area("Secret content", key="vault_content")
        if st.button("Save to Vault"):
            if not vcontent.strip():
                st.warning("Secret content cannot be empty.")
            else:
                cipher = encrypt_text(vault_password, vcontent.strip())
                data["vault"].append({"title": vtitle or "Vault item", "cipher": cipher, "timestamp": ts_now()})
                save_data(data)
                st.success("Saved to vault.")
                st.experimental_rerun()

elif tabs == "Legacy & Export":
    st.header("Legacy Mode & Export")
    st.write("Export your EchoSoul data (timeline, vault entries remain encrypted).")
    if st.button("Download full export (JSON)"):
        st.download_button("Click to download JSON", json.dumps(data, indent=2), f"echosoul_export_{datetime.datetime.utcnow().date()}.json", "application/json")
    legacy = "\n\n".join([f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data['timeline']])
    st.text_area("Legacy snapshot", legacy, height=300)

elif tabs == "About":
    st.header("About EchoSoul")
    st.write("EchoSoul stores personal memories, adapts tone, and uses OpenAI for natural conversation.")
    st.markdown("- Persistent Memory: remembers and recalls personal details.")
    st.markdown("- Adaptive Personality: adjusts tone over time.")
    st.markdown("- Emotion Recognition: text-based heuristics; voice requires transcription.")
    st.markdown("- Life Timeline, Custom Conversation Style, Knowledge Growth, Private Vault, Legacy Mode.")
    st.markdown("**Important:** Add OPENAI_API_KEY to Streamlit Secrets to enable GPT replies and audio transcription.")

# Save data at end of request cycle (defensive)
save_data(data)
