"""
EchoSoul — Prototype with GPT integration
Now upgraded to use OpenAI for real AI replies.
"""

import streamlit as st
import os, json, hashlib, base64, datetime, re
from openai import OpenAI

# Initialize OpenAI client with secret key from Streamlit settings
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------
# Data storage
# ------------------------------
DATA_FILE = "echosoul_data.json"

def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {
        "profile": {
            "name": "User",
            "created": ts_now(),
            "persona": {"tone": "friendly", "style": "casual"}
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

# ------------------------------
# Vault encryption (simple XOR, demo only)
# ------------------------------
def _derive_key(password, length):
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return (h * (length // len(h) + 1))[:length]

def encrypt_text(password, plaintext):
    b = plaintext.encode("utf-8")
    key = _derive_key(password, len(b))
    x = bytes([b[i] ^ key[i] for i in range(len(b))])
    return base64.b64encode(x).decode("utf-8")

def decrypt_text(password, ciphertext_b64):
    try:
        data = base64.b64decode(ciphertext_b64.encode("utf-8"))
        key = _derive_key(password, len(data))
        x = bytes([data[i] ^ key[i] for i in range(len(data))])
        return x.decode("utf-8")
    except Exception:
        return None

# ------------------------------
# Memories
# ------------------------------
def add_memory(data, title, content):
    item = {
        "id": hashlib.sha1((title + content + ts_now()).encode("utf-8")).hexdigest(),
        "title": title,
        "content": content,
        "timestamp": ts_now()
    }
    data["timeline"].append(item)
    save_data(data)
    return item

# ------------------------------
# GPT-based reply
# ------------------------------
def generate_reply(data, user_msg):
    # Collect memory context (last 5 memories)
    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-5:]]
    context = "\n".join(memories) if memories else "No memories yet."

    # Persona tone
    tone = data["profile"]["persona"].get("tone", "friendly")

    # Build system prompt
    system_prompt = f"""
    You are EchoSoul, Hanshika's evolving digital companion.
    Personality tone: {tone}.
    Known memories:\n{context}
    If the user asks you to act like them, then roleplay as Hanshika.
    Always reply with warmth, natural flow, and personal touch.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )

    reply = response.choices[0].message.content
    data["conversations"].append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="EchoSoul — Prototype", layout="centered")

st.title("EchoSoul — Prototype")
st.write("A lightweight prototype of the EchoSoul concept, now powered by OpenAI for realistic conversation.")

# Load data
data = load_data()

# Sidebar
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

# Tabs
tab = st.radio("", ["Chat","Life Timeline","Private Vault","Legacy & Export","About"])

if tab == "Chat":
    st.subheader("Chat with your EchoSoul")
    for conv in data.get("conversations",[])[-20:]:
        st.markdown(f"**You:** {conv['user']}")
        st.markdown(f"**EchoSoul:** {conv['bot']}")
        st.markdown("---")
    user_input = st.text_input("Say something to EchoSoul", key="chat_input")
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please type something.")
        else:
            reply = generate_reply(data, user_input)
            st.rerun()
    if st.button("Add as memory"):
        if user_input.strip() == "":
            st.warning("Type the memory content in the box first.")
        else:
            item = add_memory(data, "User added memory", user_input.strip())
            st.success("Memory saved to timeline.")
            st.rerun()

elif tab == "Life Timeline":
    st.subheader("Life Timeline")
    if data["timeline"]:
        for item in sorted(data["timeline"], key=lambda x:x["timestamp"], reverse=True):
            st.markdown(f"**{item['title']}** — {item['timestamp']}")
            st.write(item["content"])
            st.markdown("---")
    else:
        st.info("No memories yet.")
    st.markdown("### Add new memory")
    ttitle = st.text_input("Title", key="mem_title")
    tcontent = st.text_area("Content", key="mem_content")
    if st.button("Save Memory"):
        if tcontent.strip() == "":
            st.warning("Memory content cannot be empty.")
        else:
            add_memory(data, ttitle or "Memory", tcontent.strip())
            st.success("Saved memory.")
            st.rerun()

elif tab == "Private Vault":
    st.subheader("Private Vault (encrypted notes)")
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

elif tab == "Legacy & Export":
    st.subheader("Legacy Mode & Export")
    if st.button("Download full export (JSON)"):
        st.download_button("Click to download JSON",
                           json.dumps(data, indent=2),
                           f"echosoul_export_{datetime.datetime.utcnow().date()}.json",
                           "application/json")
    legacy = "\n\n".join([f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data['timeline']])
    st.text_area("Legacy snapshot", legacy, height=300)

elif tab == "About":
    st.header("About this prototype")
    st.write("This version uses OpenAI GPT for realistic replies, while keeping the timeline, vault, and export features.")
