# app.py
import streamlit as st
import openai
import json
import os
import base64
import uuid
import datetime
from pathlib import Path
import tempfile
import traceback
from typing import Optional
import streamlit.components.v1 as components

# ---------------------------
# Configuration & Secrets
# ---------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

DATA_FILE = "echosoul_data.json"
os.makedirs("uploads", exist_ok=True)

# ---------------------------
# Utility: load/save
# ---------------------------
def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {"profile": {"name": "User"}, "memories": [], "vault": {}, "voice_file": None}
    return {"profile": {"name": "User"}, "memories": [], "vault": {}, "voice_file": None}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

data = load_data()

# ---------------------------
# Small CSS + visual tweaks
# ---------------------------
NEON_CSS = """
<style>
/* Simple modern dark layout (not flashy) */
body { background-color: #0b0c0f; color: #E6EEF6; }
.stApp > main { padding: 18px 22px; }
.sidebar .element-container { background: #1f1f23; border-radius: 12px; padding: 12px; }
h1.neon { font-family: Inter, Arial; color: #bfe9ff; text-shadow: 0 0 24px rgba(140,220,255,0.25); }
.chat-bubble-user { background: linear-gradient(90deg,#5f2ea7,#8b2b9e); padding:10px 14px; border-radius:10px; color:white; }
.chat-bubble-assistant { background: linear-gradient(90deg,#2b9aff,#1d6fd7); padding:10px 14px; border-radius:10px; color:white; }
.small-muted { color:#98A0B2; font-size:0.9em; }
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------------------
# Sidebar: Profile / Wallpaper / Voice upload
# ---------------------------
with st.sidebar:
    st.subheader("Profile")
    name = st.text_input("Your name", value=data.get("profile", {}).get("name", "User"))
    if st.button("Save name"):
        data.setdefault("profile", {})["name"] = name
        save_data(data)
        st.success("Name saved.")

    st.markdown("---")
    st.subheader("Wallpaper")
    uploaded_wall = st.file_uploader("Upload wallpaper (jpg/png)", type=["jpg", "jpeg", "png"], key="wall_upload")
    if uploaded_wall is not None:
        dest = Path("uploads") / f"wall_{uuid.uuid4().hex}_{uploaded_wall.name}"
        with open(dest, "wb") as f:
            f.write(uploaded_wall.getbuffer())
        st.session_state["wallpaper_path"] = str(dest)
        st.success("Wallpaper uploaded — click 'Apply' to use it.")
    if st.button("Apply wallpaper"):
        if st.session_state.get("wallpaper_path"):
            wp = st.session_state["wallpaper_path"]
            # Save path to data for persistence
            data.setdefault("profile", {})["wallpaper"] = wp
            save_data(data)
            st.success("Wallpaper applied (reload main view if you don't see it).")
        else:
            st.warning("Upload a wallpaper first.")

    st.markdown("---")
    st.subheader("Custom voice (sample)")
    voice_upload = st.file_uploader("Upload EchoSoul voice sample (mp3/wav) — optional", type=["mp3", "wav"], key="voice_upload")
    if voice_upload is not None:
        dest = Path("uploads") / f"voice_{uuid.uuid4().hex}_{voice_upload.name}"
        with open(dest, "wb") as f:
            f.write(voice_upload.getbuffer())
        data["voice_file"] = str(dest)
        save_data(data)
        st.success("Voice sample saved. (This app currently plays uploaded file back as a sample during calls.)")

# Apply wallpaper if present
wall_path = data.get("profile", {}).get("wallpaper") or st.session_state.get("wallpaper_path")
if wall_path and Path(wall_path).exists():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{wall_path}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Header + Navigation
# ---------------------------
st.markdown(f"<h1 class='neon'>EchoSoul — Hi {data.get('profile', {}).get('name', 'User')}</h1>", unsafe_allow_html=True)

nav = st.radio("Navigate", ["Chat", "Call", "Life Timeline", "Vault", "Export", "About"], index=0, horizontal=False)

# ---------------------------
# Helper: OpenAI Chat call with memory context
# ---------------------------
def ask_model(messages, model="gpt-4o-mini"):
    if openai.api_key is None:
        raise RuntimeError("OpenAI API key missing. Add it to Streamlit secrets or env var OPENAI_API_KEY.")
    # messages = list of dicts: {"role":"system"/"user"/"assistant", "content":"..."}
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.9, max_tokens=600)
        return resp.choices[0].message["content"]
    except Exception as e:
        st.error("Error contacting OpenAI API: " + str(e))
        raise

# ---------------------------
# Page: Chat
# ---------------------------
if nav == "Chat":
    st.subheader("Chat with EchoSoul")
    st.markdown("<div class='small-muted'>This chat uses GPT (gpt-4o-mini). EchoSoul keeps a local memory and uses it to make replies more personal.</div>", unsafe_allow_html=True)

    # history stored in session_state for UI responsiveness
    if "history" not in st.session_state:
        st.session_state["history"] = data.get("memories", []) and [{"role": "user", "content": ""}] or []

    # Display conversation (stored in session)
    chat_box = st.container()
    with chat_box:
        # iterate over stored conversation (session)
        for i, msg in enumerate(st.session_state.get("chat_messages", [])):
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'>You: {st.session_state['chat_messages'][i]['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-assistant'>EchoSoul: {st.session_state['chat_messages'][i]['content']}</div>", unsafe_allow_html=True)

    # Use the new streamlit chat_input which clears automatically on send
    try:
        user_text = st.chat_input("Type here and press Enter...")
    except Exception:
        # fallback - older Streamlit installs
        user_text = st.text_input("Say something...", key="legacy_chat_input")

    if user_text:
        # append user message to session
        st.session_state.setdefault("chat_messages", []).append({"role": "user", "content": user_text})

        # build memory context (last 5 saved memories)
        memory_text = " ".join([m["text"] for m in data.get("memories", [])[-5:]])
        system_msg = {
            "role": "system",
            "content": ("You are EchoSoul, a friendly adaptive assistant. Use the local memory when replying. "
                        f"Memory context (recent): {memory_text}")
        }

        # Build chat messages for model: include the session messages
        model_messages = [system_msg]
        # include last up to 12 messages from session
        for m in st.session_state.get("chat_messages", [])[-12:]:
            role = "user" if m["role"] == "user" else "assistant"
            model_messages.append({"role": role, "content": m["content"]})

        # call model
        with st.spinner("EchoSoul is thinking..."):
            try:
                reply = ask_model(model_messages, model="gpt-4o-mini")
            except Exception as e:
                st.error("Failed to get reply from model. See logs.")
                st.error(traceback.format_exc())
                reply = "Sorry — I couldn't reach the AI service just now."

        # append assistant reply and show
        st.session_state.setdefault("chat_messages", []).append({"role": "assistant", "content": reply})

        # Optionally: store memory if the user chooses (not automatic). Here we just show.
        st.success("EchoSoul replied.")

        # Speak the reply in-browser (browser speechSynthesis).
        # This will try to use the browser TTS — it is quick and usually works well.
        safe_reply = reply.replace('"', '\\"').replace("\n", "\\n")
        speak_js = f"""
        <script>
        (function() {{
            try {{
                const utter = new SpeechSynthesisUtterance("{safe_reply}");
                // optional tuning
                utter.rate = 1.0;
                utter.pitch = 1.0;
                // choose default voice if available
                const voices = window.speechSynthesis.getVoices();
                // pick a default female/male if available
                if (voices && voices.length > 0) {{
                    // prefer neutral English voices
                    let v = voices.find(v => v.lang && v.lang.startsWith('en')) || voices[0];
                    utter.voice = v;
                }}
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utter);
            }} catch (e) {{
                console.log('TTS JS error', e);
            }}
        }})();
        </script>
        """
        st.components.v1.html(speak_js)

    # show an explicit "Save to memory" option below messages
    st.markdown("---")
    st.write("Save something important to your local Life Timeline?")
    mem_text = st.text_area("Add memory (optional)", key="memory_input")
    if st.button("Save memory"):
        if mem_text and mem_text.strip():
            new_mem = {"id": str(uuid.uuid4()), "text": mem_text.strip(), "time": str(datetime.datetime.now())}
            data.setdefault("memories", []).append(new_mem)
            save_data(data)
            st.success("Memory saved to Life Timeline.")
        else:
            st.warning("Write something first.")

# ---------------------------
# Page: Call (push-to-talk)
# ---------------------------
elif nav == "Call":
    st.subheader("Real-ish Call (push-to-talk) with EchoSoul")
    st.markdown("Record a short voice message (press Start → Stop). The app will transcribe your message using OpenAI speech recognition and EchoSoul will reply (both text + browser TTS).")

    # Provide a simple JS recorder component that returns Base64 audio to Python
    recorder_html = """
    <div style="font-family: Inter, Arial; color: #E6EEF6;">
      <button id="startBtn">Start Recording</button>
      <button id="stopBtn" disabled>Stop & Send</button>
      <div style="margin-top:8px;">
        <audio id="player" controls></audio>
      </div>
      <div id="status" style="margin-top:8px;color:#98A0B2;">Idle</div>
    </div>
    <script>
      const startBtn = document.getElementById('startBtn');
      const stopBtn = document.getElementById('stopBtn');
      const player = document.getElementById('player');
      const status = document.getElementById('status');

      let mediaRecorder;
      let chunks = [];

      startBtn.onclick = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({audio: true});
          mediaRecorder = new MediaRecorder(stream);
          chunks = [];
          mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
          mediaRecorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            player.src = URL.createObjectURL(blob);
            const reader = new FileReader();
            reader.onloadend = () => {
              // read as base64 and post to Streamlit
              const base64data = reader.result.split(',')[1];
              const payload = { audio_b64: base64data };
              // Use the Streamlit postMessage bridge
              window.parent.postMessage({isStreamlitMessage: true, type: 'AUDIO_DATA', payload: payload}, "*");
              status.innerText = "Sent to app.";
            };
            reader.readAsDataURL(blob);
          };
          mediaRecorder.start();
          startBtn.disabled = true;
          stopBtn.disabled = false;
          status.innerText = "Recording...";
        } catch (err) {
          status.innerText = "Microphone access denied or error: " + err.message;
        }
      };

      stopBtn.onclick = () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
          mediaRecorder.stop();
        }
        startBtn.disabled = false;
        stopBtn.disabled = true;
      };
    </script>
    """
    # The returned value is None normally; Streamlit captures postMessage payload into the return value of components.html in many environments.
    result = components.html(recorder_html, height=200)

    # After user records, Streamlit may capture the message in result. Also try reading a special key set by Streamlit (some Streamlit versions set window.message into return).
    # We will check for data passed back via the special `st.session_state` key captured by a hacky method (but we handle standard `result` too)
    audio_b64 = None
    # Common pattern: result will be the payload dictionary (some Streamlit builds support this)
    if isinstance(result, dict) and result.get("type") == "AUDIO_DATA":
        audio_b64 = result.get("payload", {}).get("audio_b64")
    # Also check session_state (a fallback if a custom component plumbing populates a key)
    if not audio_b64 and st.session_state.get("last_component_message"):
        m = st.session_state.get("last_component_message")
        if isinstance(m, dict) and m.get("type") == "AUDIO_DATA":
            audio_b64 = m.get("payload", {}).get("audio_b64")

    # NOTE: Some Streamlit versions capture the posted message as the return of components.html.
    # If audio_b64 is present, decode and transcribe using OpenAI Whisper (if key configured).
    if audio_b64:
        try:
            b = base64.b64decode(audio_b64)
            tmp_path = Path(tempfile.gettempdir()) / f"user_record_{uuid.uuid4().hex}.webm"
            with open(tmp_path, "wb") as f:
                f.write(b)
            st.info("Received audio — transcribing now (Whisper)...")

            # Transcribe using OpenAI Whisper (requires openai API & appropriate plan)
            try:
                with open(tmp_path, "rb") as af:
                    transcript_resp = openai.Audio.transcribe("whisper-1", af)
                # `transcript_resp` structure depends on openai version; commonly: {'text': '...'}
                transcript_text = transcript_resp.get("text") if isinstance(transcript_resp, dict) else getattr(transcript_resp, "text", None)
            except Exception as e:
                st.error("Transcription failed: " + str(e))
                transcript_text = None

            if transcript_text:
                st.success(f"You said (transcript): {transcript_text}")

                # send to chat model
                memory_text = " ".join([m["text"] for m in data.get("memories", [])[-5:]])
                sys_msg = {"role": "system", "content": f"You are EchoSoul. Memory context: {memory_text}"}
                user_msg = {"role": "user", "content": transcript_text}
                with st.spinner("EchoSoul is forming a response..."):
                    reply = ask_model([sys_msg, user_msg], model="gpt-4o-mini")
                st.markdown(f"**EchoSoul (text):** {reply}")

                # Speak reply via browser TTS
                safe_reply = reply.replace('"', '\\"').replace("\n", "\\n")
                tts_js = f"""
                <script>
                (function() {{
                  try {{
                    const u = new SpeechSynthesisUtterance("{safe_reply}");
                    u.rate = 1.0; u.pitch = 1.0;
                    const voices = window.speechSynthesis.getVoices();
                    if (voices && voices.length) {{
                      let pref = voices.find(v => v.lang && v.lang.startsWith('en')) || voices[0];
                      u.voice = pref;
                    }}
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(u);
                  }} catch (e) {{ console.log('TTS error', e); }}
                }})();
                </script>
                """
                st.components.v1.html(tts_js, height=0)
            else:
                st.warning("Transcription returned no text.")
        except Exception as e:
            st.error("Failed to process recorded audio: " + str(e))
            st.error(traceback.format_exc())

    # Manual text fallback in call page
    st.markdown("---")
    st.write("Or type a message for a live-like reply (typed messages use the same TTS playback):")
    typed = st.text_input("Say something (call)", key="call_text")
    if st.button("Send (call)"):
        if typed and typed.strip():
            sys_msg = {"role": "system", "content": "You are EchoSoul, an adaptive voice companion."}
            user_msg = {"role": "user", "content": typed}
            with st.spinner("EchoSoul answering..."):
                try:
                    reply = ask_model([sys_msg, user_msg], model="gpt-4o-mini")
                except Exception:
                    reply = "Sorry, I couldn't generate a reply."
            st.markdown(f"**EchoSoul:** {reply}")
            # use browser TTS for playback
            safe_reply = reply.replace('"', '\\"').replace("\n", "\\n")
            components.html(f"""
            <script>
             (function(){{
               var msg = new SpeechSynthesisUtterance("{safe_reply}");
               msg.rate = 1.0; msg.pitch = 1.0;
               var voices = window.speechSynthesis.getVoices();
               if (voices && voices.length) msg.voice = voices.find(v => v.lang.startsWith('en')) || voices[0];
               window.speechSynthesis.cancel();
               window.speechSynthesis.speak(msg);
             }})();
            </script>
            """, height=0)

# ---------------------------
# Page: Life Timeline
# ---------------------------
elif nav == "Life Timeline":
    st.subheader("Life Timeline / Memories")
    st.markdown("All stored memories are local to this app (echosoul_data.json). You can save important events and EchoSoul will use recent memories as context.")
    new_mem = st.text_area("Add a memory")
    if st.button("Save memory (timeline)"):
        if new_mem and new_mem.strip():
            item = {"id": str(uuid.uuid4()), "text": new_mem.strip(), "time": str(datetime.datetime.now())}
            data.setdefault("memories", []).append(item)
            save_data(data)
            st.success("Saved to timeline.")
        else:
            st.warning("Write something to save.")

    st.markdown("### Recent memories")
    for m in reversed(data.get("memories", [])[-30:]):
        st.markdown(f"- **{m['time']}** — {m['text']}")

# ---------------------------
# Page: Vault
# ---------------------------
elif nav == "Vault":
    st.subheader("Private Vault (Prototype)")
    st.markdown("Vault entries are stored locally in echosoul_data.json. This is a simple demo — not strong cryptography.")
    pwd = st.text_input("Set/enter simple vault password (note: demo XOR-like storage; not true secure crypto)", type="password")
    note = st.text_area("Secret note")
    if st.button("Save to vault"):
        if not pwd or not note:
            st.warning("Provide both password and note.")
        else:
            # simple demo: store with base64 encoding + password salt (NOT SECURE, only prototype)
            enc = base64.b64encode(note.encode("utf-8")).decode("utf-8")
            entry = {"id": str(uuid.uuid4()), "time": str(datetime.datetime.now()), "data": enc}
            data.setdefault("vault", {})[entry["id"]] = entry
            sav
