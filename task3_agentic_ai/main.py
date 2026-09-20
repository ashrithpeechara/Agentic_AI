"""
Task 3 – Agentic AI: Core Agent (ReAct Loop)
=============================================
Implements the ReAct (Reason + Act) pattern:

  THOUGHT  → The agent reasons about what to do next
  ACTION   → The agent calls a tool with an argument
  OBSERVATION → The tool result is fed back to the agent
  ... (loop until FINAL_ANSWER)

The agent loop continues until the LLM outputs a FINAL_ANSWER
or until the maximum iteration limit is reached.
"""

import os
import sys
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tools import TOOLS, get_tools_description, execute_tool

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=API_KEY)
MODEL  = "gemini-3.5-flash"

MAX_ITERATIONS = 8  # Prevent infinite loops


# ─────────────────────────────────────────────────────────────
# System Prompt (ReAct format)
# ─────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    return f"""You are an intelligent AI agent. You solve tasks step by step using a strict ReAct format.

## Available Tools
{get_tools_description()}

## Response Format (STRICT — follow exactly)
You MUST ALWAYS begin your output with THOUGHT: followed by your reasoning, then write ACTION: or FINAL_ANSWER:.

THOUGHT: [Your explicit reasoning about what to do next]
ACTION: tool_name('argument')

OR, when you have enough information to answer:

THOUGHT: [Your explicit reasoning about why you have enough info]
FINAL_ANSWER: [Complete, well-formatted answer to the original task]

## Rules
1. ALWAYS include THOUGHT: as the very first line of every response.
2. Output ONLY one THOUGHT + ACTION pair at a time (or THOUGHT + FINAL_ANSWER).
3. Never make up tool results — wait for the OBSERVATION.
4. Use the calculator for ANY math, even simple calculations.
5. Use wikipedia_search for factual questions about topics, people, places, events.
6. Use current_datetime when asked about today's date or time.
7. Use text_analyzer for word/character counts.
8. After receiving OBSERVATION results, reason about them before continuing.
9. Your FINAL_ANSWER must be complete, clear, and directly answer the user's task.
10. Do NOT wrap tool arguments in extra quotes if they contain quotes — use single quotes inside.
"""


# ─────────────────────────────────────────────────────────────
# Parse agent response
# ─────────────────────────────────────────────────────────────
def parse_response(text: str) -> dict:
    """
    Parse the LLM response for THOUGHT, ACTION, and FINAL_ANSWER.
    Returns a dict with keys: thought, action, action_arg, final_answer
    """
    result = {
        "thought":      "",
        "action":       None,
        "action_arg":   None,
        "final_answer": None,
    }

    # Extract THOUGHT
    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    else:
        # Fallback: check if there is leading text before ACTION or FINAL_ANSWER
        leading = re.split(r"(?:ACTION:|FINAL_ANSWER:)", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if leading:
            result["thought"] = leading

    # Extract FINAL_ANSWER
    final_match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        if not result["thought"]:
            result["thought"] = "I have gathered all the necessary information to formulate the final answer."
        return result

    # Extract ACTION
    action_match = re.search(
        r"ACTION:\s*(\w+)\s*\((['\"]?)(.*?)(\2)\s*\)",
        text, re.DOTALL | re.IGNORECASE
    )
    if action_match:
        result["action"]     = action_match.group(1).strip()
        result["action_arg"] = action_match.group(3).strip()
    else:
        alt_match = re.search(r"ACTION:\s*(\w+)\s*\((.*?)\)", text, re.DOTALL | re.IGNORECASE)
        if alt_match:
            result["action"]     = alt_match.group(1).strip()
            result["action_arg"] = alt_match.group(2).strip().strip("'\"`")

    if not result["thought"] and result["action"]:
        result["thought"] = f"Executing tool {result['action']} to obtain required information."

    return result


# ─────────────────────────────────────────────────────────────
# Agent Step — yields events for streaming
# ─────────────────────────────────────────────────────────────
def run_agent(task: str):
    """
    Generator that runs the ReAct agent loop.
    Yields dicts: {type, content} for streaming to frontend.

    Types: 'plan', 'thought', 'action', 'observation', 'final', 'error', 'limit'
    """
    # Build initial plan
    yield {"type": "plan", "content": f"Task received: {task}\nStarting ReAct agent loop..."}

    conversation = []  # list of Content objects

    # First message — the task
    user_msg = f"Task: {task}"
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=user_msg)])
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        yield {"type": "iteration", "content": str(iteration)}

        # Call Gemini
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=conversation,
                config=types.GenerateContentConfig(
                    system_instruction=build_system_prompt(),
                    temperature=0.2,  # Lower temp for more predictable structured output
                ),
            )
            reply_text = response.text.strip()
        except Exception as ex:
            yield {"type": "error", "content": str(ex)}
            return

        # Add model response to conversation
        conversation.append(
            types.Content(role="model", parts=[types.Part(text=reply_text)])
        )

        # Parse the response
        parsed = parse_response(reply_text)

        # Emit thought
        if parsed["thought"]:
            yield {"type": "thought", "content": parsed["thought"]}

        # Check for final answer
        if parsed["final_answer"]:
            yield {"type": "final", "content": parsed["final_answer"]}
            return

        # Execute action
        if parsed["action"]:
            tool_name = parsed["action"]
            tool_arg  = parsed["action_arg"] or ""

            yield {
                "type":    "action",
                "content": f"{tool_name}('{tool_arg}')",
                "tool":    tool_name,
                "arg":     tool_arg,
            }

            # Run the tool
            observation = execute_tool(tool_name, tool_arg)
            yield {"type": "observation", "content": observation}

            # Feed observation back into conversation
            obs_msg = f"OBSERVATION: {observation}"
            conversation.append(
                types.Content(role="user", parts=[types.Part(text=obs_msg)])
            )
        else:
            # No action and no final answer — likely a formatting issue
            # Ask the model to continue
            conversation.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text="Please continue with the next ACTION or provide the FINAL_ANSWER.")]
                )
            )

    # Max iterations reached
    yield {"type": "limit", "content": f"Maximum iterations ({MAX_ITERATIONS}) reached."}


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    DIVIDER = "=" * 60

    print(f"\n{DIVIDER}")
    print("  Task 3 -- Agentic AI (ReAct Pattern)")
    print(f"  Tools: {', '.join(TOOLS.keys())}")
    print(f"{DIVIDER}\n")

    task_examples = [
        "What is 15% of 847 and what is today's date?",
        "Search Wikipedia for Python programming language and count how many words are in the summary.",
        "Calculate the area of a circle with radius 7 and find today's date.",
    ]
    print("Example tasks:")
    for i, ex in enumerate(task_examples, 1):
        print(f"  {i}. {ex}")
    print()

    while True:
        task = input("Enter task (or 'quit'): ").strip()
        if not task:
            continue
        if task.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print()
        for event in run_agent(task):
            t = event["type"]
            c = event["content"]
            if t == "plan":
                print(f"\n[AGENT STARTED]\n{c}")
            elif t == "iteration":
                print(f"\n--- Iteration {c} ---")
            elif t == "thought":
                print(f"THOUGHT: {c}")
            elif t == "action":
                print(f"ACTION:  {c}")
            elif t == "observation":
                print(f"OBSERVATION: {c}")
            elif t == "final":
                print(f"\n{'='*50}\nFINAL ANSWER:\n{c}\n{'='*50}")
            elif t == "error":
                print(f"ERROR: {c}")
            elif t == "limit":
                print(f"LIMIT: {c}")
        print()


if __name__ == "__main__":
    main()
