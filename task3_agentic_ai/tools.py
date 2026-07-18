"""
Task 3 – Agentic AI: Tools
============================
Real tool implementations available to the AI agent.
Each tool returns a string result that feeds back into the agent loop.
"""

import math
import datetime
import re
import json
import urllib.request
import urllib.parse


# ─────────────────────────────────────────────────────────────
# Tool: Calculator
# ─────────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e, etc.
    """
    # Whitelist allowed names
    safe_dict = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "pow": math.pow,
    }
    try:
        # Remove any dangerous constructs
        clean = re.sub(r'[^0-9+\-*/().,%^sqrt sincotanlogexpabsroundpie \n]', '', expression)
        # Use the original expression but evaluate safely
        result = eval(expression, safe_dict)
        if isinstance(result, float):
            # Format nicely
            if result == int(result):
                return str(int(result))
            return f"{result:.6g}"
        return str(result)
    except Exception as ex:
        return f"Error evaluating expression: {ex}"


# ─────────────────────────────────────────────────────────────
# Tool: Current DateTime
# ─────────────────────────────────────────────────────────────
def current_datetime(fmt: str = "full") -> str:
    """Return the current date and time."""
    now = datetime.datetime.now()
    if fmt == "date":
        return now.strftime("%A, %B %d, %Y")
    if fmt == "time":
        return now.strftime("%I:%M %p")
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


# ─────────────────────────────────────────────────────────────
# Tool: Wikipedia Search
# ─────────────────────────────────────────────────────────────
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia and return a summary of the top result.
    Uses the Wikipedia REST API — no API key required.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgenticAI-Task3/1.0 (educational project)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        title   = data.get("title", query)
        extract = data.get("extract", "")

        if not extract:
            return f"No Wikipedia article found for '{query}'."

        # Trim to requested sentence count
        sent_list = re.split(r'(?<=[.!?])\s+', extract)
        trimmed   = " ".join(sent_list[:sentences])

        return f"[Wikipedia: {title}]\n{trimmed}"

    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Try search endpoint as fallback
            return _wikipedia_search_fallback(query)
        return f"Wikipedia HTTP error: {e.code}"
    except Exception as ex:
        return f"Wikipedia search error: {ex}"


def _wikipedia_search_fallback(query: str) -> str:
    """Fallback: use Wikipedia search API to find a close match."""
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}"
            f"&format=json&srlimit=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgenticAI-Task3/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results found for '{query}'."
        title   = results[0]["title"]
        snippet = re.sub(r'<[^>]+>', '', results[0]["snippet"])
        return f"[Wikipedia: {title}]\n{snippet}..."
    except Exception as ex:
        return f"Wikipedia fallback error: {ex}"


# ─────────────────────────────────────────────────────────────
# Tool: Text Analyzer
# ─────────────────────────────────────────────────────────────
def text_analyzer(text: str) -> str:
    """Analyze text: word count, character count, sentence count, etc."""
    words     = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chars     = len(text)
    chars_no_space = len(text.replace(" ", ""))
    unique_words = len(set(w.lower().strip('.,!?";:') for w in words))

    return (
        f"Word count: {len(words)}\n"
        f"Character count (with spaces): {chars}\n"
        f"Character count (no spaces): {chars_no_space}\n"
        f"Sentence count: {len(sentences)}\n"
        f"Unique words: {unique_words}"
    )


# ─────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────
TOOLS = {
    "calculator": {
        "fn":          calculator,
        "description": "Evaluates a mathematical expression. Input: math expression as string. E.g. calculator('25 * 0.15') or calculator('sqrt(144)').",
        "example":     "calculator('(100 + 200) * 3 / 2')",
    },
    "current_datetime": {
        "fn":          current_datetime,
        "description": "Returns the current date and/or time. Input: 'full' (default), 'date', or 'time'.",
        "example":     "current_datetime('full')",
    },
    "wikipedia_search": {
        "fn":          wikipedia_search,
        "description": "Searches Wikipedia and returns a factual summary. Input: search query string.",
        "example":     "wikipedia_search('Python programming language')",
    },
    "text_analyzer": {
        "fn":          text_analyzer,
        "description": "Analyzes a piece of text: word count, character count, sentence count, unique words.",
        "example":     "text_analyzer('your text here')",
    },
}


def get_tools_description() -> str:
    """Return a formatted description of all available tools for the LLM prompt."""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"- **{name}**: {info['description']}")
        lines.append(f"  Example: `{info['example']}`")
    return "\n".join(lines)


def execute_tool(name: str, arg: str) -> str:
    """Execute a tool by name with a string argument."""
    if name not in TOOLS:
        return f"Error: Unknown tool '{name}'. Available: {list(TOOLS.keys())}"
    tool_fn = TOOLS[name]["fn"]
    try:
        # Try to call with the arg as-is (string)
        return str(tool_fn(arg))
    except TypeError:
        try:
            # Try calling with no arguments
            return str(tool_fn())
        except Exception as ex:
            return f"Tool execution error: {ex}"
