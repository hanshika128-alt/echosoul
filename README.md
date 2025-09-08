
# EchoSoul — Prototype

This is a lightweight, single-file Streamlit prototype of the EchoSoul concept described in the uploaded PDF.

Files:
- app.py : Streamlit app (single-file)
- requirements.txt

How to run (example):
1. Create a virtual environment and install requirements:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run:
   ```
   streamlit run app.py
   ```

Notes:
- Data is stored at `/mnt/data/echosoul_data.json` (this path is used in the prototype).
- The "Private Vault" uses a simple password-based XOR encoding for demonstration — NOT secure for real secrets.
- This is a prototype meant to mirror the high-level features described in the project's PDF.
