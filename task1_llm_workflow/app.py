"""
Task 1 – LLM Workflow: Flask Backend
=====================================
Serves the HTML frontend and exposes a /api/chat endpoint
that communicates with the Google Gemini API.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env file.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.5-flash"

app = Flask(__name__, static_folder="static")

# In-memory session store: session_id -> list of Content objects
sessions: dict = {}


def build_history(raw_history: list) -> list:
    """Convert raw [{role, text}] list to google.genai Content objects."""
    contents = []
    for msg in raw_history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["text"])]
            )
        )
    return contents


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = (data.get("message") or "").strip()
    history_raw = data.get("history", [])   # [{role, text}, ...]

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Build Gemini content history
    contents = build_history(history_raw)
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful, concise, and knowledgeable AI assistant. "
                    "Provide clear and accurate answers. Use markdown formatting "
                    "where appropriate (bold, bullet points, etc.)."
                )
            ),
        )
        reply = response.text
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print("=" * 50)
    print("  Task 1 – LLM Workflow with HTML Frontend")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
