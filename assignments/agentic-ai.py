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

def plan_task(task):
    prompt = f"Break the following task into a numbered list of clear steps:\n{task}"
    return ask(prompt)

def execute_steps(task, plan):
    prompt = (
        f"Task: {task}\n"
        f"Plan:\n{plan}\n\n"
        "Now execute each step above and produce the final combined output."
    )
    return ask(prompt)

class SimpleAgent:
    def __init__(self, task):
        self.task = task
        self.plan = None
        self.result = None

    def run(self):
        print("Planning task...")
        self.plan = plan_task(self.task)
        print("\nPlan:\n", self.plan)
        print("\nExecuting steps...")
        self.result = execute_steps(self.task, self.plan)
        print("\nFinal Output:\n", self.result)
        return self.result

if __name__ == "__main__":
    task = input("Enter a task for the agent: ")
    agent = SimpleAgent(task)
    agent.run()