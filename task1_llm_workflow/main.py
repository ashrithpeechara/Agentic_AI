"""
Task 1 - LLM Workflow
=====================
A Python program that accepts user input and generates a response
using the Google Gemini LLM API (google-genai SDK).

Requirements:
    pip install google-genai python-dotenv
"""

import os
import sys

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found.\n"
        "Please set it in the .env file or as an environment variable."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash"


# ---------------------------------------------------------
# Core LLM function
# ---------------------------------------------------------
def generate_response(user_input: str, history: list) -> str:
    """
    Send user_input to Gemini and return the model response text.
    Maintains conversation history for multi-turn chat.
    """
    # Append user message to history
    history.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful, concise, and knowledgeable AI assistant. "
                "Provide clear and accurate answers to the user's questions."
            )
        ),
    )

    reply = response.text

    # Append model response to history
    history.append(
        types.Content(role="model", parts=[types.Part(text=reply)])
    )

    return reply


# ---------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------
DIVIDER = "-" * 60


def print_banner():
    print("\n" + "=" * 60)
    print("       LLM Workflow -- Google Gemini Chat")
    print("=" * 60)
    print("  Type your message and press Enter to chat.")
    print("  Commands:  'quit' or 'exit' to stop  |  'clear' to reset")
    print("=" * 60 + "\n")


def print_response(text: str):
    print(f"\n{DIVIDER}")
    print("Gemini:")
    print(DIVIDER)
    for line in text.splitlines():
        print(f"  {line}" if line.strip() else "")
    print(DIVIDER + "\n")


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
def main():
    print_banner()

    history: list = []
    turn = 1

    while True:
        try:
            user_input = input(f"You [{turn}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("\nSession ended. Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            turn = 1
            print("\nConversation history cleared. Starting fresh!\n")
            continue

        print("\nThinking...")
        try:
            response = generate_response(user_input, history)
            print_response(response)
        except Exception as exc:
            print(f"\nError: {exc}\n")

        turn += 1


if __name__ == "__main__":
    main()
