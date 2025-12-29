"""
DietBite Pro Backend (Flask)
- Groq LLM + Retrieval (KB) + Simple session memory
- POST /api/chat  (mobile/web client)
- GET  /health
"""

import os
import uuid
import pickle
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "kb_index.pkl"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
kb_index = None

# sessions[session_id] = {"history": [{"role": "user"/"assistant", "content": str}]}
sessions = {}


# =========================
# KB LOAD + SEARCH
# =========================
def load_kb():
    global kb_index
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            kb_index = pickle.load(f)
        print(f"[KB] Loaded: {INDEX_PATH}")
    else:
        kb_index = None
        print(f"[KB] WARNING: kb_index.pkl not found at {INDEX_PATH}. Run build_kb.py first.")


def search_kb(query: str, top_k: int = 4):
    if not kb_index or not query.strip():
        return []

    vectorizer = kb_index["vectorizer"]
    matrix = kb_index["matrix"]
    chunks = kb_index["chunks"]

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = chunks[int(idx)]
        results.append(
            {
                "source": c.get("source", "unknown"),
                "page": c.get("page", "?"),
                "text": c.get("text", ""),
                "score": float(sims[int(idx)]),
            }
        )
    return results


# =========================
# SESSION
# =========================
def get_or_create_session(session_id: str | None):
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {"history": []}
    return session_id, sessions[session_id]


# =========================
# LLM RESPONSE
# =========================
def generate_reply(user_message: str, history: list, kb_snippets: list[dict]) -> str:
    if not client:
        return "Server missing GROQ_API_KEY. Add it in your environment variables (Render) and redeploy."

    context_parts = []
    for s in kb_snippets[:4]:
        context_parts.append(
            f"- Source: {s.get('source')} (page {s.get('page')}): {s.get('text')}"
        )
    context_text = "\n".join(context_parts) if context_parts else "No matching diet references were found."

    system_prompt = (
        "You are DietBite Pro, a clinical nutrition assistant for hospitals, dietitians, and patients.\n"
        "You provide educational support about therapeutic diets and hospital diet levels.\n"
        "You do NOT provide medical diagnosis.\n"
        "Always advise users to consult a licensed clinician for personalized medical advice.\n"
        "If asked for meal plans, provide safe general examples and common restrictions.\n"
        "Be clear, structured, and practical.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # add last few turns
    for m in history[-6:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    user_content = (
        f"User question:\n{user_message}\n\n"
        f"Relevant reference snippets:\n{context_text}\n\n"
        "Answer with:\n"
        "1) Recommended diet type(s) if applicable\n"
        "2) Key rules / restrictions\n"
        "3) Common foods to include / avoid\n"
        "4) A simple sample day menu (if asked)\n"
        "5) Safety note (consult clinician)\n"
    )

    messages.append({"role": "user", "content": user_content})

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return completion.choices[0].message.content


# =========================
# ROUTES
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}), 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id")

    if not user_message:
        return jsonify({"reply": "Please type a question so I can help."}), 400

    session_id, sess = get_or_create_session(session_id)

    sess["history"].append({"role": "user", "content": user_message})

    kb_snips = search_kb(user_message, top_k=4)

    reply = generate_reply(user_message=user_message, history=sess["history"], kb_snippets=kb_snips)

    sess["history"].append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply, "session_id": session_id, "matches": kb_snips}), 200


if __name__ == "__main__":
    load_kb()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

