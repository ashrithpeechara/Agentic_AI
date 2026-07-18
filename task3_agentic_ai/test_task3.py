"""
Verification test for Task 3 — Agentic AI
Tests individual tools AND runs a full agent loop.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

print("=" * 55)
print("   Task 3 -- Agentic AI Verification Test")
print("=" * 55)

# ── Test 1: Tool — Calculator ─────────────────────────
print("\n[Test 1] Calculator tool...")
from tools import calculator
r = calculator("25 * 0.15")
assert r == "3.75" or "3.75" in r, f"Expected 3.75, got {r}"
r2 = calculator("sqrt(144)")
assert "12" in r2, f"Expected 12, got {r2}"
print(f"  PASS  |  25 * 0.15 = {r}  |  sqrt(144) = {r2}")

# ── Test 2: Tool — Current Datetime ──────────────────
print("[Test 2] current_datetime tool...")
from tools import current_datetime
dt = current_datetime("full")
assert len(dt) > 5, f"Expected a date string, got: {dt}"
print(f"  PASS  |  {dt}")

# ── Test 3: Tool — Wikipedia Search ──────────────────
print("[Test 3] wikipedia_search tool...")
from tools import wikipedia_search
result = wikipedia_search("Python programming language")
assert "Python" in result or "python" in result.lower(), f"Unexpected result: {result[:100]}"
print(f"  PASS  |  Got {len(result)} chars from Wikipedia")

# ── Test 4: Tool — Text Analyzer ─────────────────────
print("[Test 4] text_analyzer tool...")
from tools import text_analyzer
analysis = text_analyzer("Hello world. This is a test.")
assert "Word count: 6" in analysis, f"Unexpected: {analysis}"
print(f"  PASS  |  {analysis.splitlines()[0]}")

# ── Test 5: Full Agent Loop ───────────────────────────
print("\n[Test 5] Full ReAct agent loop...")
print("  Task: 'What is 100 * 5 and what is today's date?'")
print("  Running agent (makes LLM calls)...\n")

from main import run_agent

events = list(run_agent("What is 100 * 5 and what is today's date?"))
types_seen = [e["type"] for e in events]

has_thought  = "thought"  in types_seen
has_action   = "action"   in types_seen
has_observe  = "observation" in types_seen
has_final    = "final"    in types_seen

print("  Agent trace:")
for ev in events:
    t, c = ev["type"], ev["content"]
    if t == "thought":    print(f"    THOUGHT: {c[:80]}...")
    elif t == "action":   print(f"    ACTION:  {c}")
    elif t == "observation": print(f"    OBSERVE: {c[:60]}")
    elif t == "final":    print(f"    FINAL:   {c[:120]}...")

print()
checks = [
    ("Agent emitted THOUGHT events",      has_thought),
    ("Agent emitted ACTION events",       has_action),
    ("Agent emitted OBSERVATION events",  has_observe),
    ("Agent reached FINAL_ANSWER",        has_final),
]

all_pass = True
for label, passed in checks:
    mark = "PASS" if passed else "FAIL"
    if not passed: all_pass = False
    print(f"  [{mark}] {label}")

print("\n" + "=" * 55)
if all_pass:
    print("  All tests passed! Task 3 is working correctly.")
    print("  Run  `python app.py`  to start the web UI (port 5002).")
else:
    print("  Some tests failed. Check output above.")
print("=" * 55 + "\n")
