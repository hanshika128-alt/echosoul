
"""
EchoSoul — Prototype Streamlit app
This is a simplified prototype implementing the core features described in the uploaded EchoSoul.pdf:
- Persistent memory (JSON file)
- Adaptive personality (tone changes with sentiment)
- Simple sentiment recognition (heuristic lexicon)
- Life timeline (add/view memories)
- Private vault with password-based XOR 'encryption' (prototype - not cryptographically secure)
- Legacy export (downloadable JSON)
- Text chat interface (Streamlit)

Notes/Limitations:
- This is a prototype. It intentionally uses lightweight, dependency-free approaches so you can run it without heavy ML models.
- For production, replace the simple sentiment and 'encryption' with proper ML models and real cryptography.
"""

import streamlit as st
import os, json, hashlib, base64, datetime, re

DATA_FILE = "echosoul_data.json"

# ------------------------------
# Simple storage helpers
# ------------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data()
    else:
        return default_data()

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def default_data():
    return {
        "profile":{"name":"User","created":ts_now(),"persona":{"tone":"friendly","style":"casual"} },
        "timeline":[],
        "vault":[],
        "conversations":[]
    }

def ts_now():
    return datetime.datetime.utcnow().isoformat()+"Z"

# ------------------------------
# Simple (prototype) 'encryption' using XOR derived key
# ------------------------------
def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    key = (h * (length // len(h) + 1))[:length]
    return key

def encrypt_text(password, plaintext):
    if password == "":
        raise ValueError("Vault password cannot be empty for encryption.")
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_text(password, ciphertext_b64):
    try:
        if password == "":
            raise ValueError("Vault password cannot be empty for decryption.")
        data = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data))
        x = bytes([data[i] ^ key[i] for i in range(len(data))])
        return x.decode("utf-8")
    except Exception:
        return None

# ------------------------------
# Very small sentiment heuristic
# ------------------------------
POS_WORDS = set("""good great happy love excellent amazing wonderful nice
grateful gratefulness fun delighted excited calm optimistic""".split())
NEG_WORDS = set("""bad sad angry depressed unhappy terrible awful hate lonely anxious
stressed worried frustrated""".split())

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    # normalize by length
    norm = score / max(1, len(toks))
    return norm

def sentiment_label(score):
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"

# ------------------------------
# Memory & timeline helpers
# ------------------------------
def add_memory(data, title, content):
    item = {"id": hashlib.sha1((title+content+ts_now()).encode("utf-8")).hexdigest(),
            "title": title, "content": content, "timestamp": ts_now()}
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

# ------------------------------
# Adaptive persona
# ------------------------------
def update_persona_based_on_sentiment(data, score):
    tone = data["profile"].get("persona", {}).get("tone", "friendly")
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# ------------------------------
# Generate a reply (prototype)
# ------------------------------
GENERIC_QUESTIONS = ["how are you", "who are you", "what can you do", "tell me about yourself"]
def generate_reply(data, user_msg):
    s = sentiment_score(user_msg)
    label = sentiment_label(s)
    update_persona_based_on_sentiment(data, s)

    # try to recall memories
    rec = find_relevant_memories(data, user_msg)
    if rec:
        # weave memories in reply
        mem = rec[0]
        reply = f"I remember something you told me: \"{mem['title']}\" — {mem['content']}. Would you like to talk more about it?"
        data["conversations"].append({"user":user_msg,"bot":reply,"ts":ts_now()})
        save_data(data)
        return reply

    low = user_msg.lower().strip()
    for q in GENERIC_QUESTIONS:
        if q in low:
            if q == "how are you":
                persona = data["profile"]["persona"]["tone"]
                if persona == "empathetic":
                    return "I'm here for you. How are *you* feeling right now?"
                if persona == "energetic":
                    return "Feeling upbeat and ready to help you—what shall we do?"
                return "I'm doing well — I'm here to listen. How can I help?"
            if q == "who are you":
                return "I'm EchoSoul — your evolving digital companion."
            if q == "what can you do":
                return "I can listen, remember things you tell me, adapt my style, keep a private vault, and help you reflect on your life."
            if q == "tell me about yourself":
                return "I grow with you. Over time I remember details and adapt my voice to match you."

    # Default templated reply
    persona = data["profile"]["persona"]["tone"]
    if persona == "empathetic":
        tail = "I hear you — would you like to tell me more?"
    elif persona == "energetic":
        tail = "That's awesome! Want to explore more ideas?"
    else:
        tail = "Thanks for sharing. Anything else on your mind?"

    # Short "knowledge growth": if user writes "I am X" or "I like X", store as memory suggestion
    match = re.search(r"\bi (?:am|was|feel|like|love)\s+(.+)", user_msg.lower())
    if match:
        fact = match.group(1).strip().capitalize()
        add_memory(data, "Personal note", fact)
        reply = f"Noted: {fact}. {tail}"
        data["conversations"].append({"user":user_msg,"bot":reply,"ts":ts_now()})
        save_data(data)
        return reply

    reply = f"I heard you: \"{user_msg}\". {tail}"
    data["conversations"].append({"user":user_msg,"bot":reply,"ts":ts_now()})
    save_data(data)
    return reply

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="EchoSoul — Prototype", layout="centered")

st.title("EchoSoul — Prototype")
st.write("A lightweight prototype of the EchoSoul concept (based on the project's PDF). This demo stores data locally on the device and is intended for experimentation.")

# Load data
data = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Your name", value=data["profile"].get("name","User"))
    if st.button("Save profile name"):
        data["profile"]["name"] = name
        save_data(data)
        st.success("Name saved.")
    st.markdown("---")
    st.markdown("**Vault** (prototype)")
    vault_password = st.text_input("Vault password (used for encrypt/decrypt)", type="password")
    if st.button("Clear vault password (local only)"):
        vault_password = ""
        st.info("Vault password cleared in this session.")
    st.markdown("---")
    st.checkbox("Enable adaptive learning (persona update)", value=True, key="adaptive_toggle")

# Tabs for Chat / Timeline / Vault / Legacy
tab = st.radio("", ["Chat","Life Timeline","Private Vault","Legacy & Export","About"])

if tab == "Chat":
    st.subheader("Chat with your EchoSoul")
    # Show conversation
    for conv in data.get("conversations",[])[-50:]:
        st.markdown(f"**You:** {conv['user']}")
        st.markdown(f"**EchoSoul:** {conv['bot']}")
        st.markdown("---")
    user_input = st.text_input("Say something to EchoSoul", key="chat_input")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Send"):
            if user_input.strip() == "":
                st.warning("Please type something.")
            else:
                reply = generate_reply(data, user_input)
                st.experimental_rerun()
    with col2:
        if st.button("Add as memory"):
            if user_input.strip() == "":
                st.warning("Type the memory content in the box first.")
            else:
                item = add_memory(data, "User added memory", user_input.strip())
                st.success("Memory saved to timeline.")
                st.experimental_rerun()

if tab == "Life Timeline":
    st.subheader("Life Timeline")
    st.write("Add, view and search chronological memories.")
    # Show timeline
    if data["timeline"]:
        for item in sorted(data["timeline"], key=lambda x:x["timestamp"], reverse=True):
            st.markdown(f"**{item['title']}** — {item['timestamp']}")
            st.write(item["content"])
            st.markdown("---")
    else:
        st.info("No memories yet. Use Chat -> 'Add as memory' or the form below to add new items.")
    st.markdown("### Add new memory")
    ttitle = st.text_input("Title", key="mem_title")
    tcontent = st.text_area("Content", key="mem_content")
    if st.button("Save Memory"):
        if tcontent.strip() == "":
            st.warning("Memory content cannot be empty.")
        else:
            add_memory(data, ttitle or "Memory", tcontent.strip())
            st.success("Saved memory.")
            st.experimental_rerun()

if tab == "Private Vault":
    st.subheader("Private Vault (encrypted notes)")
    st.write("Store sensitive memories here. This prototype uses a password-based XOR-style encoding — not secure for real secrets. Use only for testing.")
    # show vault items (titles only)
    if data["vault"]:
        for v in data["vault"]:
            st.markdown(f"**{v['title']}** — {v['timestamp']}")
            if vault_password:
                dec = decrypt_text(vault_password, v["cipher"])
                if dec is not None:
                    st.write(dec)
                else:
                    st.write("*Unable to decrypt with the provided password.*")
            else:
                st.write("*Password not provided in sidebar.*")
            st.markdown("---")
    else:
        st.info("No vault items yet.")
    st.markdown("### Add vault item")
    vt = st.text_input("Title for vault item", key="vt")
    vc = st.text_area("Secret content", key="vc")
    if st.button("Save to Vault"):
        if not vault_password:
            st.warning("Set a vault password in the sidebar first.")
        elif vc.strip() == "":
            st.warning("Secret content cannot be empty.")
        else:
            cipher = encrypt_text(vault_password, vc.strip())
            data["vault"].append({"title": vt or "Vault item", "cipher": cipher, "timestamp": ts_now()})
            save_data(data)
            st.success("Saved to vault.")

if tab == "Legacy & Export":
    st.subheader("Legacy Mode & Export")
    st.write("Export your EchoSoul data (timeline, vault entries *still encrypted*, profile).")
    if st.button("Download full export (JSON)"):
        st.download_button("Click to download JSON", json.dumps(data, indent=2), f"echosoul_export_{datetime.datetime.utcnow().date()}.json", "application/json")
    st.markdown("---")
    st.write("Generate a simple 'Legacy snapshot' (human-readable timeline).")
    legacy = "\n\n".join([f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data['timeline']])
    st.text_area("Legacy snapshot", legacy, height=300)

if tab == "About":
    st.header("About this prototype")
    st.write("This demo implements a simplified EchoSoul inspired by the uploaded project document. It is a small, local prototype and is intentionally lightweight so you can run it on a typical machine.")
    st.markdown("**Key implemented features:**")
    st.markdown("- Persistent memory (timeline), simple adaptive personality, heuristic sentiment detection, private vault (prototype encryption), legacy export.")
    st.markdown("**Not included in the prototype:**")
    st.markdown("- Deep neural memory networks, advanced sentiment ML models, voice biometrics, real cryptography. Those require extra dependencies and model assets.")
    st.markdown("If you want I can help upgrade this to use real ML models (BERT / sentence-transformers) and proper encryption.")
