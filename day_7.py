#groq

from groq import Groq
import os, json, pandas as pd
from getpass import getpass
from langdetect import detect_langs, DetectorFactory
from datetime import datetime
import gradio as gr

DetectorFactory.seed = 0

# -------- API KEY --------
if "GROQ_API_KEY" in os.environ:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
else:
    key = getpass("Enter GROQ API Key: ").strip()
    if not key:
        raise SystemExit("Groq API key required")
    client = Groq(api_key=key)

# -------- CONFIG --------
MODEL = "llama-3.1-8b-instant"  # NEW WORKING MODEL
CSV_PATH = "C:/Users/FASI OWAIZ AHMED/Desktop/hidevs/translations.csv"

COMMON_LANG = {
    "en":"English","es":"Spanish","fr":"French","de":"German","it":"Italian",
    "pt":"Portuguese","ru":"Russian","zh-cn":"Chinese (Simplified)","zh-tw":"Chinese (Traditional)",
    "ja":"Japanese","ko":"Korean","ar":"Arabic","hi":"Hindi","bn":"Bengali"
}

# -------- Language Detect --------
def detect_language(text):
    try:
        langs = detect_langs(text)
        top = langs[0]
        code = str(top).split(':')[0]
        conf = float(str(top).split(':')[1])
        return {
            "code": code,
            "name": COMMON_LANG.get(code, code),
            "confidence": round(conf,3)
        }
    except Exception:
        return {"code":"unknown","name":"unknown","confidence":0.0}

# -------- Translation (Groq) --------
def translate_with_groq(original):
    prompt = f"""
You are a translation engine. Output ONLY valid JSON.

Task:
1. Translate the following message into fluent natural English.
2. Identify the source language (ISO code).
3. Add short translation notes (optional).

Respond ONLY in this JSON format:
{{
  "translation": "...",
  "source_language": "...",
  "notes": "..."
}}

Message:
\"\"\"{original}\"\"\"
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = res.choices[0].message.content.strip()
        
        # Extract JSON
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])

        return {
            "translation": text,
            "source_language": "unknown",
            "notes": "json_parse_failed"
        }

    except Exception as e:
        return {"translation":"", "source_language":"error", "notes":str(e)}

# -------- Reply Generation --------
def generate_reply(english_text, tone="professional"):
    prompt = f"""
Write a short, helpful, polite customer-support reply in ENGLISH.

Tone: {tone}
User message: "{english_text}"

Output ONLY the reply text.
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"(reply_error) {e}"

# -------- Logging --------
def append_log(row):
    df = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)

# -------- Gradio Logic --------
def handle(text, gen_reply, rating, tone):
    ts = datetime.utcnow().isoformat()

    if not text.strip():
        return "—", "Enter text.", "", "No input."

    det = detect_language(text)
    ai = translate_with_groq(text)
    translation = ai.get("translation", "") or text

    reply = ""
    if gen_reply:
        reply = generate_reply(translation, tone)

    row = {
        "timestamp": ts,
        "original": text,
        "detected_code": det["code"],
        "detected_name": det["name"],
        "detected_conf": det["confidence"],
        "translation": translation,
        "notes": ai.get("notes",""),
        "reply": reply,
        "rating": int(rating)
    }
    append_log(row)

    return (
        f"{det['name']} ({det['code']}) — conf {det['confidence']}",
        translation,
        reply,
        "OK"
    )

# -------- Gradio UI --------
with gr.Blocks(title="Groq Multilingual Query Handler (Fixed)") as demo:
    gr.Markdown("## Real-Time Multilingual Query Handler (Groq API - New Model)")
    gr.Markdown("Ultra-fast translation & reply using `llama-3.1-8b-instant`")

    with gr.Row():
        inp = gr.Textbox(label="Message", lines=4)
        with gr.Column(scale=1):
            gen_reply = gr.Checkbox(label="Generate reply?", value=True)
            tone = gr.Radio(["professional","friendly","formal"], value="professional", label="Tone")
            rating = gr.Radio(["0","1","2","3","4","5"], value="0", label="Rate Translation")
            btn = gr.Button("Translate")

    det_box = gr.Textbox(label="Detected Language", interactive=False)
    trans_box = gr.Textbox(label="English Translation", interactive=False)
    reply_box = gr.Textbox(label="Suggested Reply", interactive=False)
    status = gr.Label()

    btn.click(handle, [inp, gen_reply, rating, tone], [det_box, trans_box, reply_box, status])

demo.launch(share=True)
