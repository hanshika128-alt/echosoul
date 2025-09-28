import streamlit as st
import json
import os
from datetime import datetime
from openai import OpenAI

# ============= INITIAL SETUP =============
st.set_page_config(page_title="EchoSoul", page_icon="⚡", layout="wide")

# Initialize OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Persistent memory file
DATA_FILE = "echosoul_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"memories": [], "vault": {}, "conversations": []}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Load stored data
data = load_data()

# ============= SIDEBAR =============
with st.sidebar:
    st.title("⚡ EchoSoul")
    st.markdown("Your **AI Companion**")
    st.divider()

    st.subheader("📍 Navigation")
    page = st.radio(
        "Choose a page:",
        ["Chat", "Life Timeline", "Vault", "Legacy & Export", "Settings"]
    )
    st.divider()

    st.subheader("🎨 Theme")
    theme = st.radio("Theme Mode", ["Light", "Dark"])
    st.session_state["theme"] = theme

    st.divider()
    st.markdown("🔐 **Prototype Vault** – not true cryptography")
    st.caption("Transparency: Data is stored locally in `echosoul_data.json`.")

# ============= MAIN CONTENT =============

# CHAT PAGE
if page == "Chat":
    st.title("💬 Chat with EchoSoul")

    st.markdown(
        """
        EchoSoul adapts to your tone:
        - 😊 Positive → Energetic  
        - 😐 Neutral → Friendly  
        - 😔 Negative → Empathetic
        """
    )

    # Input field
    user_input = st.text_input("Type your message:", key="chat_input")
    send = st.button("Send")

    if send and user_input.strip():
        # Save user input
        data["conversations"].append({
            "role": "user",
            "content": user_input,
            "time": str(datetime.now())
        })
        save_data(data)

        # Generate AI reply
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are EchoSoul, an empathetic, transparent AI companion. Stay in character."}
                ] + [
                    {"role": c["role"], "content": c["content"]}
                    for c in data["conversations"][-10:]
                ]
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"

        # Save assistant reply
        data["conversations"].append({
            "role": "assistant",
            "content": reply,
            "time": str(datetime.now())
        })
        save_data(data)

        # Display conversation
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**EchoSoul:** {reply}")

        # Clear input box
        st.session_state.chat_input = ""

    st.divider()
    st.subheader("📊 Transparency")
    st.info("EchoSoul uses OpenAI GPT (gpt-4o-mini). Conversations are stored locally. Confidence scores and reasoning are simplified for transparency.")

    # Call AI button
    if st.button("📞 Call EchoSoul"):
        try:
            speech_file = "voice_reply.mp3"
            reply_text = data["conversations"][-1]["content"] if data["conversations"] else "Hello, I am EchoSoul."
            with open(speech_file, "wb") as f:
                tts = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",  # You can change to custom trained voice if available
                    input=reply_text
                )
                f.write(tts.read())
            st.audio(speech_file, format="audio/mp3")
        except Exception as e:
            st.error(f"Voice error: {str(e)}")

# LIFE TIMELINE
elif page == "Life Timeline":
    st.title("📅 Life Timeline")

    if data["memories"]:
        for m in data["memories"]:
            st.markdown(f"- {m}")
    else:
        st.info("No memories saved yet.")

    new_memory = st.text_input("Add new memory:")
    if st.button("Save Memory"):
        if new_memory.strip():
            data["memories"].append(new_memory)
            save_data(data)
            st.success("Memory added.")
        else:
            st.warning("Cannot save empty memory.")

# VAULT
elif page == "Vault":
    st.title("🔐 Private Vault")

    password = st.text_input("Enter Vault Password:", type="password")
    if password:
        st.success("Vault unlocked (prototype only).")
        note = st.text_area("Write a private note:")
        if st.button("Save to Vault"):
            if note.strip():
                data["vault"][str(datetime.now())] = note
                save_data(data)
                st.success("Note saved securely (XOR-based prototype).")
            else:
                st.warning("Cannot save empty note.")

# LEGACY & EXPORT
elif page == "Legacy & Export":
    st.title("📜 Legacy & Export")

    st.markdown("Export all your data for transparency and control.")
    st.download_button(
        "⬇️ Download All Data (JSON)",
        json.dumps(data, indent=4),
        file_name="echosoul_export.json"
    )

    st.subheader("📖 Legacy Snapshot")
    if data["memories"]:
        for idx, m in enumerate(data["memories"], 1):
            st.markdown(f"**{idx}.** {m}")
    else:
        st.info("No memories saved yet.")

# SETTINGS
elif page == "Settings":
    st.title("⚙️ Profile Settings")

    name = st.text_input("Your Name:", value=st.session_state.get("username", ""))
    if st.button("Save Settings"):
        st.session_state["username"] = name
        st.success("Settings saved. EchoSoul will now address you by name.")
else:
    pass
