import streamlit as st
import sqlite3
import os
from datetime import datetime
import openai
from dateutil import parser as dateparser

# ==============================
# DB Setup
# ==============================
DB_FILE = "echosoul.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  role TEXT,
                  content TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_chat(role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)",
              (role, content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    st.session_state["messages"].append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})

def load_chats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chats ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]

# ==============================
# LLM Chat Function
# ==============================
def chat_with_llm(user_input, system="You are EchoSoul, a supportive AI companion."):
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
# UI Pages
# ==============================
def chat_view():
    st.header("Chat — EchoSoul")

    # Show history
    for m in st.session_state["messages"]:
        role = "👤 You" if m["role"] == "user" else "🤖 EchoSoul"
        ts = dateparser.parse(m["timestamp"]).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"**{role}** ({ts}): {m['content']}")

    st.markdown("---")

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message:")
        submitted = st.form_submit_button("Send")
        if submitted and user_input.strip():
            txt = user_input.strip()
            save_chat("user", txt)

            reply = chat_with_llm(txt)
            save_chat("assistant", reply)

            st.experimental_rerun()

def history_view():
    st.header("Chat History")
    rows = load_chats()
    for r in rows:
        role = "👤 You" if r["role"] == "user" else "🤖 EchoSoul"
        st.markdown(f"**{role}** ({r['timestamp']}): {r['content']}")

def timeline_view():
    st.header("Life Timeline")
    st.info("⏳ Timeline feature under development...")

def vault_view():
    st.header("Vault (Password Protected)")
    pw = st.text_input("Enter password:", type="password")
    if pw == "1234":  # 🔑 change this in production
        st.success("Vault unlocked ✅")
        st.write("Your private memories go here...")
    elif pw:
        st.error("Wrong password!")

def export_view():
    st.header("Export Data")
    if st.button("Export Chat History"):
        rows = load_chats()
        with open("exported_chats.txt", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['role']} ({r['timestamp']}): {r['content']}\n")
        st.success("Exported to exported_chats.txt")

def brain_mimic_view():
    st.header("Brain Mimic 🧠")
    st.info("AI will try to respond in your own style (based on your past chats).")

def call_view():
    st.header("📞 Call Simulation")
    st.info("Voice call simulation feature coming soon.")

def about_view():
    st.header("About EchoSoul")
    st.write("Personal AI companion — persistent memory, timeline, vault, and voice calls.")

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

    # Ensure session messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_chats()

    # Routing
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
