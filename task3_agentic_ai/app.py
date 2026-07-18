"""
Task 3 – Agentic AI: Flask Backend with SSE Streaming
=======================================================
Streams agent ReAct loop events in real-time to the frontend.
"""

import os
import sys
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
from main import run_agent

load_dotenv()

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/agent/run")
def agent_run():
    """
    SSE endpoint — streams ReAct agent events as they happen.
    Query param: ?task=...
    """
    task = request.args.get("task", "").strip()
    if not task:
        return jsonify({"error": "No task provided"}), 400

    def generate():
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        try:
            for event in run_agent(task):
                yield sse(event)
        except Exception as e:
            yield sse({"type": "error", "content": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  Task 3 -- Agentic AI (ReAct Loop)")
    print("  Open http://127.0.0.1:5002 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5002)
