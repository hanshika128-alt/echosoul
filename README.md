# ✨ EchoSoul

**EchoSoul** is a personal AI companion built with [Streamlit](https://streamlit.io).  
It remembers you, adapts its personality, and grows with you over time.

---

## 🌟 Features
- **Conversational AI** — uses OpenAI GPT (`gpt-4o-mini`) for natural dialogue.
- **Persistent Memory** — stores facts & memories in `echosoul_data.json`.
- **Adaptive Personality** — adjusts tone (friendly / empathetic / energetic).
- **Life Timeline** — view, add, and manage significant memories.
- **Private Vault** — save encrypted notes with your own password (XOR demo or Fernet if `cryptography` installed).
- **Legacy Export** — export all your data (chat + memories + vault) as JSON.
- **Profile Settings** — set your name, age, hobbies, free-time.
- **Light & Dark Mode** — Neon dark theme by default, toggle to light theme.
- **Explainability & Trust**  
  - Shows persona tone and heuristic confidence.  
  - Lets you add/delete memories manually.  
  - All data stays local (`echosoul_data.json`).  

---

## 🚀 Deployment
1. **Clone or fork this repo** into your GitHub.
2. Create a free [Streamlit Cloud](https://streamlit.io/cloud) account.
3. Deploy the repo → select `app.py` as the main file.
4. In **Streamlit Secrets**, add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-..."
