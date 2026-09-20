import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def summarize(topic):
    prompt = f"Write a short summary about: {topic}"
    return ask(prompt)

def extract_key_points(summary):
    prompt = f"Extract the key points from this summary:\n{summary}"
    return ask(prompt)

def generate_questions(key_points):
    prompt = f"Based on these key points, generate 3 questions:\n{key_points}"
    return ask(prompt)

if __name__ == "__main__":
    topic = input("Enter a topic: ")
    summary = summarize(topic)
    print("\nStep 1 - Summary:\n", summary)
    key_points = extract_key_points(summary)
    print("\nStep 2 - Key Points:\n", key_points)
    questions = generate_questions(key_points)
    print("\nStep 3 - Three Questions:\n", questions)