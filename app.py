import streamlit as st
import os, json, hashlib, base64, datetime, re, textwrap
from typing import Optional

# ------------------------------
# OpenAI setup
# ------------------------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None

# ------------------------------
# Storage
# ------------------------------
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

data = load_data()

# ------------------------------
# Sentiment & Persona
# ------------------------------
POS_WORDS = {"good","great","happy","love","excellent","amazing","wonderful","nice","grateful","fun"}
NEG_WORDS = {"bad","sad","angry","depressed","terrible","awful","hate","lonely","anxious","stressed"}

def sentiment_score(text):
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    return (pos - neg) / max(1,len(toks))

def update_persona(data, score):
    if score < -0.05:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.05:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# ------------------------------
# Memories
# ------------------------------
def add_memory(data, title, content):
    mem = {
        "id": hashlib.sha1((title+content+ts_now()).encode()).hexdigest(),
        "title": title,
        "content": content,
        "timestamp": ts_now()
    }
    data["timeline"].append(mem)
    save_data(data)
    return mem

# ------------------------------
# GPT-based Reply
# ------------------------------
def generate_reply(data, user_msg):
    score = sentiment_score(user_msg)
    update_persona(data, score)

    memories = [f"{m['title']}: {m['content']}" for m in data['timeline'][-5:]]
    context = "\n".join(memories) if memories else "No memories yet."
    tone = data["profile"]["persona"].get("tone","friendly")

    system_prompt = f"""
    You are EchoSoul, a warm and evolving companion.
    Tone: {tone}.
    User profile: {data['profile']}
    Known memories: {context}
    If asked to act like the user, roleplay them naturally.
    """

    if client is None:
        reply = "⚠️ OpenAI not configured. Add your API key in Streamlit → Secrets."
    else:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_msg}
                ]
            )
            reply = resp.choices[0].message.content
        except Exception as e:
            reply = f"[GPT error: {e}]"

    # auto memory save
    if "remember" in user_msg.lower() or "my name is" in user_msg.lower():
        add_memory(data,"User fact",user_msg)

    data["conversations"].append({"user":user_msg,"bot":reply,"ts":ts_now()})
    save_data(data)
    return reply

# ------------------------------
# Onboarding
# ------------------------------
def run_onboarding(data):
    st.header("Welcome to EchoSoul — let's get to know you")
    step = st.session_state.get("onb_step",0)

    if step == 0:
        name = st.text_input("What's your name?")
        if st.button("Next"):
            if name.strip():
                data["profile"]["name"]=name.strip()
                add_memory(data,"Name",f"My name is {name.strip()}")
                st.session_state.onb_step=1
                save_data(data)
                st.rerun()
    elif step == 1:
        age = st.text_input("What's your age?")
        if st.button("Next"):
            if age.strip():
                data["profile"]["age"]=age.strip()
                add_memory(data,"Age",f"My age is {age.strip()}")
                st.session_state.onb_step=2
                save_data(data)
                st.rerun()
    elif step == 2:
        hobbies = st.text_input("What are your hobbies?")
        if st.button("Next"):
            if hobbies.strip():
                data["profile"]["hobbies"]=hobbies.strip()
                add_memory(data,"Hobbies",hobbies.strip())
                st.session_state.onb_step=3
                save_data(data)
                st.rerun()
    elif step == 3:
        free_time = st.text_input("What do you like to do in free time?")
        if st.button("Finish"):
            if free_time.strip():
                data["profile"]["free_time"]=free_time.strip()
                add_memory(data,"Free time",free_time.strip())
                data["profile"]["intro_completed"]=True
                save_data(data)
                st.success("Thanks! I’ll remember this.")
                st.session_state.onb_step=0
                st.rerun()

# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="EchoSoul",layout="wide")
st.title("EchoSoul — Your personal companion")

if not data["profile"].get("intro_completed",False):
    run_onboarding(data)
    st.stop()

tabs = st.radio("",["Chat","Timeline","About"],horizontal=True)

if tabs=="Chat":
    st.header(f"Chat with EchoSoul — Hi {data['profile']['name']}")
    conv = data.get("conversations",[])[-10:]
    for c in conv:
        st.markdown(f"**You:** {c['user']}")
        st.markdown(f"**EchoSoul:** {c['bot']}")
        st.markdown("---")

    user_input = st.chat_input("Say something to EchoSoul")
    if user_input:
        reply = generate_reply(data,user_input)
        st.rerun()

elif tabs=="Timeline":
    st.header("Your Memories & Timeline")
    if not data["timeline"]:
        st.info("No memories yet.")
    else:
        for m in sorted(data["timeline"],key=lambda x:x["timestamp"],reverse=True):
            st.markdown(f"**{m['title']}** — {m['timestamp']}")
            st.write(m["content"])
            st.markdown("---")

elif tabs=="About":
    st.header("About EchoSoul")
    st.write("EchoSoul remembers your details, adapts its tone, and grows with you over time.")
    st.write("- Persistent Memory\n- Adaptive Personality\n- Emotion Recognition\n- Life Timeline\n- Private Vault (optional)\n- Legacy Mode (export memories)")
