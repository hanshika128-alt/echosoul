# 🌌 EchoSoul — Your Digital Soul

EchoSoul is a futuristic AI companion app built with **Streamlit** + **OpenAI GPT**.  
It can **chat, remember, and even speak back to you** in natural voices.  
The app evolves over time, storing memories and creating a lasting legacy.

---

## ✨ Features
- 💬 **Conversational AI** — chat naturally with EchoSoul (GPT-4o-mini)
- 🎤 **Voice Chat** — speak into your mic, EchoSoul replies with voice
- 🗣 **Voice Options** — choose from preset AI voices (`alloy`, `verse`, `amber`)
- 🪞 **Mimic Mode** — upload your own voice sample, EchoSoul imitates it (prototype)
- 🕰 **Life Timeline** — add & browse memories over time
- 🔐 **Vault** — password-protected note storage with encryption
- 📤 **Legacy Export** — download your digital life as JSON
- 🌌 **Futuristic UI** — neon dark theme, glowing AI status orb
- ⚖️ **Ethical by Design** — local storage, privacy controls, inclusivity focus

---

## 🖥 Sidebar Navigation
- **🌌 Control Center** — profile, settings, ethics reminders
- **🎤 Voice Settings** — select AI voice or upload sample for mimic mode
- **⚙️ Settings** — adaptive learning toggle, theme toggle
- **📂 Multimodal Input** — file uploads, (future) audio input
- **🆘 Help** — support & documentation links

---

## 🚀 Deployment

### 1. Upload to GitHub
Push `app.py`, `requirements.txt`, and `README.md` to a public or private repo.

### 2. Deploy on Streamlit Cloud
- Go to [Streamlit Cloud](https://share.streamlit.io)
- Link your GitHub repo
- Add **OpenAI API Key** in **Settings → Secrets**:

```toml
OPENAI_API_KEY = "your_api_key_here"
