# app.py
"""
EchoSoul — Neon AI Companion
"""

import streamlit as st
import streamlit.components.v1 as components
import json, os, base64, datetime, hashlib, re, html

# ---------------- CONFIG ----------------
st.set_page_config(page_title="EchoSoul — Neon", layout="wide")
DATA_FILE = "echosoul_data.json"

# ---------------- OPENAI ----------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI = True
except Exception:
    client = None
    OPENAI = False

# ---------------- HELPERS ----------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()+"Z"

def ensure_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE,"w") as f:
            json.dump({
                "profile":{"name":"User","persona":{"tone":"friendly"},"created":now_iso()},
                "timeline":[], "vault":[], "conversations":[], "wallpaper":None
            },f)

def load():
    ensure_file()
    return json.load(open(DATA_FILE,"r"))

def save(d):
    json.dump(d,open(DATA_FILE,"w"),indent=2)

def short_id(s): return hashlib.sha1(s.encode()).hexdigest()[:8]
def esc(s): return html.escape(s).replace("\n","<br>")

# XOR demo
import hashlib as _h
def _k(p,l): h=_h.sha256(p.encode()).digest();return (h*((l//len(h))+1))[:l]
def xor_enc(p,t): b=t.encode();k=_k(p,len(b));x=bytes([b[i]^k[i] for i in range(len(b))]);return base64.b64encode(x).decode()
def xor_dec(p,c): 
    try:
        b=base64.b64decode(c.encode());k=_k(p,len(b));x=bytes([b[i]^k[i] for i in range(len(b))]);return x.decode()
    except: return None

# Sentiment heuristic
POS={"good","great","happy","love","amazing","wonderful","nice"}
NEG={"bad","sad","angry","hate","terrible","awful"}
def sentiment(txt):
    toks=re.findall(r"\w+",txt.lower())
    pos=sum(t in POS for t in toks);neg=sum(t in NEG for t in toks)
    sc=(pos-neg)/max(1,len(toks))
    if sc>0.05: lab="positive"
    elif sc<-0.05: lab="negative"
    else: lab="neutral"
    return lab,round(sc,3)

# AI call
def ai_reply(data,user):
    if not OPENAI: return f"(offline) {user}"
    sys=f"You are EchoSoul, personal AI for {data['profile']['name']}."
    msgs=[{"role":"system","content":sys},{"role":"user","content":user}]
    r=client.chat.completions.create(model="gpt-4o-mini",messages=msgs)
    return r.choices[0].message.content

# Neon CSS
st.markdown("""
<style>
.stApp {background:#0d0d2e;color:#e8faff;}
.app-header{font-size:30px;color:#4ee0ff;text-shadow:0 0 12px #4ee0ff;}
.chat-bubble{padding:10px 14px;border-radius:10px;margin:8px 0;max-width:80%;}
.user-bubble{background:linear-gradient(90deg,#bb1b67,#5b2a86);color:#fff;}
.bot-bubble{background:linear-gradient(90deg,#023b5a,#07224a);color:#e8fbff;}
</style>
""",unsafe_allow_html=True)

# ---------------- DATA ----------------
ensure_file()
data=load()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.text_input("Your name",data["profile"]["name"],key="nm")
    if st.button("Save name"): data["profile"]["name"]=st.session_state.nm;save(data)

    up=st.file_uploader("Upload wallpaper",type=["jpg","png"])
    if up and st.button("Set as wallpaper"):
        fn=f"wp_{short_id(up.name)}.png";open(fn,"wb").write(up.getbuffer())
        data["wallpaper"]=fn;save(data);st.success("Wallpaper set");st.experimental_rerun()

# apply wallpaper
if data.get("wallpaper") and os.path.exists(data["wallpaper"]):
    b64=base64.b64encode(open(data["wallpaper"],"rb").read()).decode()
    st.markdown(f"<style>.stApp{{background:url(data:image/png;base64,{b64}) center/cover}}</style>",unsafe_allow_html=True)

# ---------------- NAV ----------------
st.markdown(f"<div class='app-header'>✨ EchoSoul — {data['profile']['name']}</div>",unsafe_allow_html=True)
page=st.radio("",["Chat","Call","Timeline","Vault","Export","About"],horizontal=True)

# ---------------- CHAT ----------------
if page=="Chat":
    for c in data["conversations"][-30:]:
        st.markdown(f"<div class='chat-bubble user-bubble'>You: {esc(c['user'])}</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble bot-bubble'>EchoSoul: {esc(c['bot'])}</div>",unsafe_allow_html=True)
    txt=st.text_input("Say something")
    if st.button("Send") and txt.strip():
        bot=ai_reply(data,txt)
        data["conversations"].append({"user":txt,"bot":bot,"ts":now_iso()})
        save(data);st.experimental_rerun()

# ---------------- CALL ----------------
elif page=="Call":
    st.write("🎙️ Call EchoSoul (speech recognition + speech synthesis)")
    if "call_hist" not in st.session_state: st.session_state.call_hist=[]
    for m in st.session_state.call_hist:
        css="user-bubble" if m["role"]=="user" else "bot-bubble"
        st.markdown(f"<div class='chat-bubble {css}'>{m['role']}: {esc(m['txt'])}</div>",unsafe_allow_html=True)

    # recorder widget (fixed triple quotes)
    recorder = """
    <button onclick="rec()">🎤 Start</button>
    <div id='out'></div>
    <script>
    const R = window.SpeechRecognition||window.webkitSpeechRecognition;
    function rec(){
      if(!R){document.getElementById('out').innerText='Not supported';return;}
      let r=new R();r.lang='en-US';r.onresult=e=>{
        const t=e.results[0][0].transcript;
        const u=new URL(window.location);
        u.searchParams.set('msg',t);window.location=u;
      };r.start();
    }
    </script>
    """
    components.html(recorder,height=120)

    p=st.experimental_get_query_params()
    if "msg" in p:
        u=p["msg"][0]
        st.session_state.call_hist.append({"role":"user","txt":u})
        bot=ai_reply(data,u)
        st.session_state.call_hist.append({"role":"bot","txt":bot})
        data["conversations"].append({"user":u,"bot":bot,"ts":now_iso()});save(data)
        js=f"""
        <script>
        const u=new SpeechSynthesisUtterance("{u}");
        const b=new SpeechSynthesisUtterance("{bot}");
        speechSynthesis.speak(b);
        </script>
        """
        components.html(js,height=0)
        st.experimental_set_query_params()

# ---------------- TIMELINE ----------------
elif page=="Timeline":
    st.write("📜 Life Timeline")
    for m in data["timeline"]:
        st.write(m["timestamp"],m["title"],m["content"])
    t=st.text_input("Title");c=st.text_area("Content")
    if st.button("Save memory"): 
        data["timeline"].append({"id":short_id(t+c), "title":t,"content":c,"timestamp":now_iso()})
        save(data);st.experimental_rerun()

# ---------------- VAULT ----------------
elif page=="Vault":
    pw=st.text_input("Vault password",type="password")
    for v in data["vault"]:
        st.write(v["title"],v["timestamp"])
        if pw: st.write(xor_dec(pw,v["cipher"]) or "Wrong password")
    t=st.text_input("Title");c=st.text_area("Secret")
    if st.button("Save secret") and pw:
        data["vault"].append({"title":t,"cipher":xor_enc(pw,c),"timestamp":now_iso()})
        save(data);st.experimental_rerun()

# ---------------- EXPORT ----------------
elif page=="Export":
    st.download_button("Download JSON",json.dumps(data,indent=2),"echosoul.json")

# ---------------- ABOUT ----------------
else:
    st.write("About EchoSoul — prototype AI companion with chat, memory, vault, call, export.")
