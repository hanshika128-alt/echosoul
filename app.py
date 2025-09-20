import streamlit as st
import os, json, hashlib, base64, datetime, re
from openai import OpenAI

"""
EchoSoul — Prototype App
Author: Hanshika (Ishu)
Description:
An evolving digital companion that remembers, reflects, and adapts.
"""

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DATA_FILE = "data.json"

# ---------- Utility Functions ----------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {
            "profile": {"name": "Hanshika", "nickname": "Ishu", "persona": {"tone": "warm"}},
            "memories": [],
            "timeline": [],
            "conversations": []
        }

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def ts_now():
    return datetime.datetime.now().isoformat(timespec="seconds")

# ---------- AI Response ----------
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

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="EchoSoul", page_icon="✨")
    st.title("✨ EchoSoul — Your Digital Reflection")

    data = load_data()

    # Input box
    user_msg = st.text_input("Speak with EchoSoul:", "")

    if user_msg:
        reply = generate_reply(data, user_msg)
        st.markdown(f"**EchoSoul:** {reply}")

    # Show history
    st.subheader("🗨️ Conversation History")
    for c in data["conversations"][-5:]:
        st.markdown(f"- **You:** {c['user']}")
        st.markdown(f"  - **EchoSoul:** {c['bot']}")

    # Show memories
    st.subheader("📖 Memories & Timeline")
    if data["timeline"]:
        for m in data["timeline"][-5:]:
            st.markdown(f"- *{m['title']}* — {m['content']}")
    else:
        st.info("No memories yet. EchoSoul will build them as you chat.")

if __name__ == "__main__":
    main()
