import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_response(user_input):
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    user_input = input("Enter your question: ")
    response = get_response(user_input)
    print("\nLLM Response:\n", response)