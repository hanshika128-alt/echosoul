import streamlit as st
import sqlite3
import os
from datetime import datetime
from dateutil import parser as dateparser
import openai
import base64

# ==============================
# DB Setup
# ==============================
DB_FILE = "echosoul.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Chats
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  role TEXT,
                  content TEXT,
                  timestamp TEXT)''')
    # Timeline
    c.execute('''CREATE TABLE IF NOT EXISTS timeline
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event TEXT,
                  timestamp TEXT)''')
    # Vault
    c.execute('''CREATE TABLE IF NOT EXISTS vault
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  note TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ==============================
# DB Functions
# ==============================
def save_chat(role, content):
    ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)", (role, content, ts))
    conn.commit()
    conn.close()
    st.session_state["messages"].append({"role": role, "content": content, "timestamp": ts})

def load_chats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chats ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]

def save_event(event):
    ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO timeline (event, timestamp) VALUES (?, ?)", (event, ts))
    conn.commit()
    conn.close()

def load_events():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT event, timestamp FROM timeline ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"event": e, "timestamp": t} for e, t in rows]

def save_vault(note):
    ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO vault (note, timestamp) VALUES (?, ?)", (note, ts))
    conn.commit()
    conn.close()

def load_vault():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT note, timestamp FROM vault ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"note": n, "timestamp": t} for n, t in rows]

# ==============================
# LLM Chat
# ==============================
def chat_with_llm(user_input, system="You are EchoSoul, a kind, supportive AI companion."):
    try:
        client = openai.OpenAI(api_key=st.session_state.get("api_key"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(Error: {e})"

# ==============================
# Views
# ==============================
def chat_view():
    st.header("Chat — EchoSoul")

    # Show history
    for m in st.session_state["messages"]:
        role = "👤 You" if m["role"] == "user" else "🤖 EchoSoul"
        ts = dateparser.parse(m["timestamp"]).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"**{role}** ({ts}): {m['content']}")

    st.markdown("---")

    # Input
    user_input = st.text_input("Type your message:", key="chat_input")
    if st.button("Send"):
        if user_input.strip():
            txt = user_input.strip()
            save_chat("user", txt)
            reply = chat_with_llm(txt)
            save_chat("assistant", reply)

def history_view():
    st.header("Chat History")
    rows = load_chats()
    for r in rows:
        role = "👤 You" if r["role"] == "user" else "🤖 EchoSoul"
        st.markdown(f"**{role}** ({r['timestamp']}): {r['content']}")

def timeline_view():
    st.header("Life Timeline")
    event = st.text_input("Add a life event:")
    if st.button("Save Event"):
        if event.strip():
            save_event(event.strip())
            st.success("Event saved.")

    st.subheader("Your Timeline")
    for e in load_events():
        ts = dateparser.parse(e["timestamp"]).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"📌 **{ts}** — {e['event']}")

def vault_view():
    st.header("Vault 🔒")
    pw = st.text_input("Enter password:", type="password")
    if pw == "1234":  # change this
        st.success("Vault unlocked ✅")
        note = st.text_area("Write a private note:")
        if st.button("Save Note"):
            if note.strip():
                save_vault(note.strip())
                st.success("Note saved.")
        st.subheader("Your Vault Notes")
        for v in load_vault():
            ts = dateparser.parse(v["timestamp"]).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"📝 **{ts}** — {v['note']}")
    elif pw:
        st.error("Wrong password!")

def export_view():
    st.header("Export Data")
    rows = load_chats()
    content = "\n".join([f"{r['role']} ({r['timestamp']}): {r['content']}" for r in rows])
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="chats.txt">📥 Download Chats</a>'
    st.markdown(href, unsafe_allow_html=True)

def brain_mimic_view():
    st.header("Brain Mimic 🧠")
    history = load_chats()
    if not history:
        st.info("No chat history yet.")
        return
    combined = "\n".join([f"{r['role']}: {r['content']}" for r in history])
    user_input = st.text_input("Ask something in your style:")
    if st.button("Mimic Reply"):
        if user_input.strip():
            sys = f"You are EchoSoul. Reply as if you were the user, based on their past messages:\n\n{combined}"
            reply = chat_with_llm(user_input, system=sys)
            st.markdown(f"🧠 Mimic: {reply}")

def call_view():
    st.header("📞 Call Simulation")
    st.info("This is a simple voice simulation using text-to-speech.")

    user_input = st.text_input("Say something (simulate voice):", key="call_input")
    if st.button("Send to Call"):
        if user_input.strip():
            reply = chat_with_llm(user_input)
            st.success(f"EchoSoul: {reply}")

def about_view():
    st.header("About EchoSoul")
    st.write("EchoSoul is your personal AI companion with memory, timeline, vault, brain mimic, and call simulation.")

# ==============================
# Main
# ==============================
def main():
    st.set_page_config(page_title="EchoSoul", layout="wide")

    # API Key
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.getenv("OPENAI_API_KEY", "")

    with st.sidebar:
        st.title("EchoSoul")
        st.caption("Personal AI companion — persistent memory, timeline, vault, and voice calls.")

        choice = st.radio("Navigate", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"])

        api_key = st.text_input("Enter OpenAI API Key:", type="password", value=st.session_state["api_key"])
        if api_key:
            st.session_state["api_key"] = api_key
            st.success("API key set.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = load_chats()

    if choice == "Chat":
        chat_view()
    elif choice == "Chat history":
        history_view()
    elif choice == "Life timeline":
        timeline_view()
    elif choice == "Vault":
        vault_view()
    elif choice == "Export":
        export_view()
    elif choice == "Brain mimic":
        brain_mimic_view()
    elif choice == "Call":
        call_view()
    elif choice == "About":
        about_view()

if __name__ == "__main__":
    main()
