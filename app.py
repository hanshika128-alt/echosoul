import streamlit as st
import json, os, uuid, datetime
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DATA_FILE = "echosoul_data.json"

# ------------------ Persistent Storage ------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"memories": [], "vault": {}, "profile": {"name": "User"}}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

data = load_data()

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Profile")

name = st.sidebar.text_input("Your name", value=data["profile"].get("name", "User"))
if st.sidebar.button("Save name"):
    data["profile"]["name"] = name
    save_data(data)
    st.success("Name saved!")

# Wallpaper upload
st.sidebar.subheader("🖼️ Wallpaper")
uploaded_wall = st.sidebar.file_uploader("Upload wallpaper", type=["jpg", "jpeg", "png"])
if uploaded_wall:
    wall_path = f"wallpaper_{uploaded_wall.name}"
    with open(wall_path, "wb") as f:
        f.write(uploaded_wall.read())
    st.session_state["wallpaper"] = wall_path
    st.success("Wallpaper updated!")

# Background CSS
if "wallpaper" in st.session_state:
    st.markdown(f"""
    <style>
    .stApp {{
        background: url('{st.session_state["wallpaper"]}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------ Navigation ------------------
st.markdown(f"<h1 style='text-align:center;color:#9df;'>EchoSoul — Hi {name}</h1>", unsafe_allow_html=True)

choice = st.radio("Navigate", ["💬 Chat", "📞 Call", "🧠 Life Timeline", "🔐 Vault", "📜 Export", "ℹ️ About"])

# ------------------ Chat ------------------
if choice == "💬 Chat":
    st.subheader("Chat with EchoSoul")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_input("Say something...", key="chat_input")
    if st.button("Send"):
        if user_input:
            st.session_state["history"].append({"role": "user", "content": user_input})

            # Persistent memory context
            memory_context = " ".join([m["text"] for m in data["memories"][-5:]])
            convo = [{"role": "system", "content": f"You are EchoSoul, adaptive, remembering. Memory: {memory_context}"}]
            convo += st.session_state["history"]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=convo
            )
            reply = response.choices[0].message.content
            st.session_state["history"].append({"role": "assistant", "content": reply})

    for h in st.session_state["history"]:
        if h["role"] == "user":
            st.markdown(f"<div style='background:#6a0dad;padding:8px;border-radius:8px;margin:4px;color:white;'>You: {h['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#1e90ff;padding:8px;border-radius:8px;margin:4px;color:white;'>EchoSoul: {h['content']}</div>", unsafe_allow_html=True)

# ------------------ Call ------------------
elif choice == "📞 Call":
    st.subheader("Talk with EchoSoul")

    call_input = st.text_input("What do you want to say? (simulating call)")
    if st.button("Speak"):
        if call_input:
            # Generate AI reply
            convo = [{"role": "system", "content": "You are EchoSoul, adaptive voice persona."}]
            convo.append({"role": "user", "content": call_input})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=convo
            )
            reply = response.choices[0].message.content

            # Inject browser speech synthesis JS
            speak_js = f"""
            <script>
            (function(){{
              const utter = new SpeechSynthesisUtterance(`{reply}`);
              utter.rate = 1;
              utter.pitch = 1;
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(utter);
              const u = new URL(window.location);
              u.searchParams.delete('call_msg');
              window.history.replaceState({{}}, document.title, u.pathname + u.search);
            }})();
            </script>
            """
            st.markdown(speak_js, unsafe_allow_html=True)
            st.success(f"EchoSoul says: {reply}")

# ------------------ Life Timeline ------------------
elif choice == "🧠 Life Timeline":
    st.subheader("Life Timeline")
    new_memory = st.text_area("Add new memory")
    if st.button("Save Memory"):
        if new_memory:
            data["memories"].append({"id": str(uuid.uuid4()), "text": new_memory, "time": str(datetime.datetime.now())})
            save_data(data)
            st.success("Memory saved!")

    for m in data["memories"]:
        st.markdown(f"📌 **{m['time']}** — {m['text']}")

# ------------------ Vault ------------------
elif choice == "🔐 Vault":
    st.subheader("Private Vault")
    password = st.text_input("Enter password", type="password")
    note = st.text_area("Secret note")
    if st.button("Save to Vault"):
        if password and note:
            data["vault"][str(uuid.uuid4())] = {"note": note, "time": str(datetime.datetime.now())}
            save_data(data)
            st.success("Saved securely!")

    st.write("🔒 Stored Vault Notes:")
    for v in data["vault"].values():
        st.markdown(f"- {v['time']}: {v['note']}")

# ------------------ Export ------------------
elif choice == "📜 Export":
    st.subheader("Export Your Data")
    st.download_button("Download JSON", json.dumps(data, indent=2), "echosoul_export.json")

# ------------------ About ------------------
elif choice == "ℹ️ About":
    st.subheader("About EchoSoul")
    st.write("""
    EchoSoul is your evolving AI companion:
    - ✅ Remembers your memories
    - ✅ Talks in adaptive style
    - ✅ Can be called with voice
    - ✅ Preserves your legacy
    """)
