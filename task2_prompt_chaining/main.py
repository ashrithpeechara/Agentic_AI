"""
Task 2 – Prompt Chaining
=========================
Multi-step LLM workflow:
  Step 1 → Generate a summary of the given topic
  Step 2 → Extract key points from the summary
  Step 3 → Produce 3 questions from the key points

Each step's output is chained as input to the next step.
"""

import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env file.")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.5-flash"

DIVIDER = "=" * 60
STEP_DIV = "-" * 60


# ─────────────────────────────────────────────────────────────
# LLM call helper
# ─────────────────────────────────────────────────────────────
def llm(prompt: str) -> str:
    """Send a single prompt to Gemini and return the response text."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert academic assistant specializing in "
                "summarization, analysis, and education. Be clear and structured."
            )
        ),
    )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────
# Step functions
# ─────────────────────────────────────────────────────────────
def step1_summarize(topic: str) -> str:
    """Step 1: Generate a comprehensive summary of the topic."""
    prompt = f"""Write a clear, comprehensive summary of the following topic in 3–4 paragraphs.
Cover the core definition, how it works, and why it matters.

Topic: {topic}

Summary:"""
    return llm(prompt)


def step2_key_points(summary: str) -> str:
    """Step 2: Extract key points from the summary (chained from Step 1)."""
    prompt = f"""Based on the following summary, extract exactly 5 key points.
Format each key point as a short, clear bullet starting with "• ".
Be specific and insightful — avoid vague statements.

Summary:
{summary}

Key Points:"""
    return llm(prompt)


def step3_questions(key_points: str) -> str:
    """Step 3: Generate 3 thought-provoking questions from the key points (chained from Step 2)."""
    prompt = f"""Based on the following key points, generate exactly 3 thought-provoking questions.
The questions should encourage deeper thinking, critical analysis, or further exploration.
Number them as Q1, Q2, Q3.

Key Points:
{key_points}

Questions:"""
    return llm(prompt)


# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────
def run_chain(topic: str) -> dict:
    """Run the full 3-step prompt chain and return all outputs."""
    print(f"\n{DIVIDER}")
    print(f"  Topic: {topic}")
    print(DIVIDER)

    print("\n[Step 1] Generating summary...")
    summary = step1_summarize(topic)
    print(f"\n{STEP_DIV}\nSUMMARY\n{STEP_DIV}")
    print(summary)

    print(f"\n\n[Step 2] Extracting key points from summary...")
    key_points = step2_key_points(summary)
    print(f"\n{STEP_DIV}\nKEY POINTS\n{STEP_DIV}")
    print(key_points)

    print(f"\n\n[Step 3] Generating questions from key points...")
    questions = step3_questions(key_points)
    print(f"\n{STEP_DIV}\nQUESTIONS\n{STEP_DIV}")
    print(questions)

    print(f"\n{DIVIDER}")
    print("  Prompt chain complete!")
    print(DIVIDER + "\n")

    return {
        "topic":      topic,
        "summary":    summary,
        "key_points": key_points,
        "questions":  questions,
    }


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + DIVIDER)
    print("  Task 2 -- Prompt Chaining (3-Step LLM Pipeline)")
    print(DIVIDER)
    print("  Enter a topic to analyze, or 'quit' to exit.\n")

    while True:
        topic = input("Enter topic: ").strip()
        if not topic:
            continue
        if topic.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        try:
            run_chain(topic)
        except Exception as e:
            print(f"\nError: {e}\n")

        again = input("Analyze another topic? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
