"""
Task 2 – Prompt Chaining: Flask Backend
=========================================
Serves the HTML frontend and streams the 3-step prompt chain
results via Server-Sent Events (SSE) for real-time step updates.
"""

import os
import sys
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env file.")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.5-flash"

app = Flask(__name__, static_folder="static")


# ─────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────
def llm(prompt: str, system: str = "") -> str:
    config = types.GenerateContentConfig(
        system_instruction=system or (
            "You are an expert academic assistant specializing in "
            "summarization, analysis, and education. Be clear and structured."
        )
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    return response.text.strip()


def step1_summarize(topic: str) -> str:
    prompt = f"""Write a clear, comprehensive summary of the following topic in 3–4 paragraphs.
Cover the core definition, how it works, and why it matters.

Topic: {topic}

Summary:"""
    return llm(prompt)


def step2_key_points(summary: str) -> str:
    prompt = f"""Based on the following summary, extract exactly 5 key points.
Format each key point as a short, clear bullet starting with "• ".
Be specific and insightful.

Summary:
{summary}

Key Points:"""
    return llm(prompt)


def step3_questions(key_points: str) -> str:
    prompt = f"""Based on the following key points, generate exactly 3 thought-provoking questions.
Encourage deeper thinking and critical analysis. Number them as Q1, Q2, Q3.

Key Points:
{key_points}

Questions:"""
    return llm(prompt)


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chain/stream")
def chain_stream():
    """
    Server-Sent Events endpoint — streams each step result as it completes.
    Query param: ?topic=...
    """
    topic = request.args.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    def generate():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        try:
            # Step 1
            yield sse("step_start", {"step": 1, "label": "Generating Summary"})
            summary = step1_summarize(topic)
            yield sse("step_done", {"step": 1, "label": "Summary", "content": summary})

            # Step 2
            yield sse("step_start", {"step": 2, "label": "Extracting Key Points"})
            key_points = step2_key_points(summary)
            yield sse("step_done", {"step": 2, "label": "Key Points", "content": key_points})

            # Step 3
            yield sse("step_start", {"step": 3, "label": "Generating Questions"})
            questions = step3_questions(key_points)
            yield sse("step_done", {"step": 3, "label": "Questions", "content": questions})

            yield sse("complete", {"topic": topic})

        except Exception as e:
            yield sse("error", {"message": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  Task 2 -- Prompt Chaining (3-Step Pipeline)")
    print("  Open http://127.0.0.1:5001 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5001)
