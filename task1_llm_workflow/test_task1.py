"""
Quick verification test for Task 1 -- LLM Workflow
Run this to confirm your Gemini API key works and responses are generated correctly.
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

print("=" * 55)
print("   Task 1 -- LLM Workflow Verification Test")
print("=" * 55)

# -- Test 1: API Key present ------------------------------
print("\n[Test 1] Checking API key in .env ...", end=" ")
if not API_KEY or API_KEY == "your_gemini_api_key_here":
    print("FAIL")
    print("\n  Please set your GEMINI_API_KEY in the .env file first.")
    sys.exit(1)
print("PASS")

# -- Test 2: Create client --------------------------------
print("[Test 2] Creating Gemini client ...", end=" ")
try:
    client = genai.Client(api_key=API_KEY)
    print("PASS")
except Exception as e:
    print(f"FAIL\n  Error: {e}")
    sys.exit(1)

# -- Test 3: Single-turn response -------------------------
print("[Test 3] Sending a test prompt to Gemini ...", end=" ")
try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="In one sentence, what is artificial intelligence?",
    )
    assert response.text and len(response.text) > 10
    print("PASS")
    print(f"\n  Gemini says:\n  \"{response.text.strip()}\"\n")
except Exception as e:
    print(f"FAIL\n  Error: {e}")
    sys.exit(1)

# -- Test 4: Multi-turn chat using history list -----------
print("[Test 4] Testing multi-turn chat history ...", end=" ")
try:
    history = [
        types.Content(role="user",  parts=[types.Part(text="My name is TestUser.")]),
        types.Content(role="model", parts=[types.Part(text="Hello TestUser! How can I help you?")]),
    ]
    history.append(
        types.Content(role="user", parts=[types.Part(text="What is my name?")])
    )
    r2 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=history,
    )
    assert "test" in r2.text.lower() or "user" in r2.text.lower()
    print("PASS")
    print(f"  Gemini (remembers context):\n  \"{r2.text.strip()}\"")
except Exception as e:
    print(f"FAIL\n  Error: {e}")
    sys.exit(1)

print("\n" + "=" * 55)
print("  All tests passed! Task 1 is working correctly.")
print("  Run  `python main.py`  to start the chat.")
print("=" * 55 + "\n")
