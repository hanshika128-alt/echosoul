# EchoSoul — Your Personal AI Companion

EchoSoul is a **personal AI companion with persistent memory**, adaptive personality, a private vault, and a life timeline.  
It is built with **Streamlit** + **OpenAI GPT (gpt-4o-mini)** and designed to feel like an ongoing dialogue, not isolated Q&A.

---

## ✨ Features

### 1. Conversational AI
- Uses **OpenAI GPT** for natural replies.
- Stays on topic and refers back to past messages.
- Feels like a continuous conversation.
- Can roleplay as you if you ask it to.
- Adapts tone automatically:
  - Positive → energetic  
  - Negative → empathetic  
  - Neutral → friendly  

### 2. Persistent Memory
- All facts & conversations are stored in `echosoul_data.json`.
- **Life Timeline**: add, view, and browse memories chronologically.
- Add memories manually or directly from chat.
- Memories influence future replies.

### 3. Adaptive Personality
- Built-in **sentiment analysis** tunes EchoSoul’s tone.
- Learns your style over time.

### 4. Private Vault (Prototype)
- Store sensitive notes in a **password-protected vault**.
- Uses **Fernet encryption** if available, otherwise XOR fallback.
- ⚠️ Prototype only — not meant for real cryptography.

### 5. Legacy & Export
- Export all data as JSON.
- Generate a **Legacy Snapshot** (human-readable timeline of your life events).

### 6. Profile Settings
- On first launch, EchoSoul asks:
  - What’s your name?  
  - What’s your age?  
  - What are your hobbies?  
  - What do you like to do in your free time?  
- These are remembered forever (and editable later).

### 7. Customization
- Change the app background with any image from your gallery.
- Dark mode by default, with a **neon high-contrast mode**.
- Upload a voice sample — EchoSoul can (optionally) reply in your voice using TTS.

### 8. Sidebar Navigation
- **Profile**: Edit your name & settings.  
- **Favorites / Pinned**: Save frequently used chats & memories.  
- **Saved Content**: Store important AI outputs.  
- **Vault**: Lock away private notes.  
- **Appearance**: Upload background images.  
- **Support & Feedback**: Access help docs or send feedback.  
- **Theme Toggle**: Switch between dark & neon high-contrast.  
- **Updates**: Small notification badge for new features.  

---

## 🖥️ Deployment

EchoSoul runs **fully in the cloud** via [Streamlit](https://streamlit.io/).  
You can open it anytime from your browser, including on tablets like the Xiaomi Pad.

### 1. Clone this repo
```bash
git clone https://github.com/your-username/echosoul.git
cd echosoul
