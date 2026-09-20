import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def llm_call(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text.strip()

def run_prompt_chain(topic: str):
    print(f"\n==========================================")
    print(f"       Prompt Chaining: {topic}")
    print(f"==========================================")
    
    # Step 1: Summary
    print("\n[Step 1] Generating Comprehensive Summary...")
    p1 = f"Write a clear, concise summary of the topic: {topic}"
    summary = llm_call(p1)
    print(f"\nSummary:\n{summary}")
    
    # Step 2: Key Points
    print("\n[Step 2] Extracting Key Points...")
    p2 = f"Extract exactly 5 key points from the following summary:\n\n{summary}"
    key_points = llm_call(p2)
    print(f"\nKey Points:\n{key_points}")
    
    # Step 3: Questions
    print("\n[Step 3] Generating Questions...")
    p3 = f"Generate 3 thought-provoking questions based on these key points:\n\n{key_points}"
    questions = llm_call(p3)
    print(f"\nQuestions:\n{questions}")

if __name__ == "__main__":
    print("--- Running Task 3: Multi-Step Prompt Chaining ---")
    run_prompt_chain("Artificial Intelligence in Healthcare")
