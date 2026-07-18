"""
Verification test for Task 2 — Prompt Chaining
Runs the full 3-step pipeline on a test topic and validates
that each step produces meaningful output chained from the previous.
"""

import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from main import run_chain

load_dotenv()

print("=" * 55)
print("   Task 2 -- Prompt Chaining Verification Test")
print("=" * 55)

TEST_TOPIC = "Machine Learning"

print(f"\n[Test] Running 3-step chain on topic: '{TEST_TOPIC}'")
print("  (This will make 3 sequential API calls...)\n")

try:
    result = run_chain(TEST_TOPIC)

    print("\n" + "=" * 55)
    print("  Validation Checks")
    print("=" * 55)

    checks = [
        ("Summary generated",        len(result["summary"]) > 100),
        ("Key points generated",     len(result["key_points"]) > 50),
        ("Questions generated",      len(result["questions"]) > 50),
        ("Summary chained to step2", result["key_points"] != result["summary"]),
        ("Key points chained to step3", result["questions"] != result["key_points"]),
        ("Questions contain Q1",     "Q1" in result["questions"] or "1." in result["questions"] or "1)" in result["questions"]),
    ]

    all_pass = True
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {label}")

    print("\n" + "=" * 55)
    if all_pass:
        print("  All checks passed! Task 2 is working correctly.")
        print("  Run  `python app.py`  to start the web UI (port 5001).")
    else:
        print("  Some checks failed. Review the output above.")
    print("=" * 55 + "\n")

except Exception as e:
    print(f"\nERROR: {e}")
    sys.exit(1)
