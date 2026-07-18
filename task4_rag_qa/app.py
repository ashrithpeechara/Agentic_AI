"""
Task 4 – RAG Q&A: Flask Backend
=================================
Endpoints:
  POST /api/upload   — Upload PDF/TXT, chunk & embed it
  POST /api/query    — RAG query against loaded document
  GET  /api/status   — Check document loaded state
  POST /api/clear    — Clear loaded document
"""

import os
import sys
import json
import tempfile

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from rag import RAGPipeline

load_dotenv()

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

# Global RAG pipeline instance (single-user demo)
rag = RAGPipeline()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "loaded":    rag.is_loaded,
        "doc_name":  rag.doc_name,
        "num_chunks": len(rag.chunks),
        "total_words": len(rag.doc_text.split()) if rag.doc_text else 0,
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload and process a document."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Use: {ALLOWED_EXTENSIONS}"}), 400

    # Save to temp file
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        stats = rag.load(tmp_path)
        stats["doc_name"] = file.filename  # use original name
        rag.doc_name = file.filename
        return jsonify({"success": True, **stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/api/query", methods=["POST"])
def query():
    """Run a RAG query against the loaded document."""
    if not rag.is_loaded:
        return jsonify({"error": "No document loaded. Please upload a document first."}), 400

    data     = request.get_json()
    question = (data.get("question") or "").strip()
    top_k    = int(data.get("top_k", 4))

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        result = rag.query(question, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    """Clear the loaded document."""
    rag.chunks   = []
    rag.doc_name = ""
    rag.doc_text = ""
    return jsonify({"success": True})


@app.route("/api/sample", methods=["POST"])
def load_sample():
    """Load the built-in sample document."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample file not found"}), 404
    try:
        stats = rag.load(sample_path)
        return jsonify({"success": True, **stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  Task 4 -- RAG-Based Question Answering")
    print("  Open http://127.0.0.1:5003 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5003)
