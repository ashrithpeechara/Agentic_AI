import os
import sqlite3
import json
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

db_path = "employees.db"

def ensure_database():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date DATE NOT NULL
        )
    ''')
    cursor.execute('SELECT COUNT(*) FROM employees')
    if cursor.fetchone()[0] == 0:
        sample_data = [
            (1, 'Alice Smith', 'Engineering', 120000, '2021-03-15'),
            (2, 'Bob Johnson', 'Sales', 85000, '2022-01-10'),
            (3, 'Charlie Brown', 'Engineering', 110000, '2021-08-22'),
            (4, 'Diana Prince', 'Marketing', 95000, '2023-05-01'),
            (5, 'Evan Wright', 'Sales', 88000, '2020-11-15')
        ]
        cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?)', sample_data)
        conn.commit()
    conn.close()

ensure_database()

def get_schema():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='employees'")
    schema = cursor.fetchone()[0]
    conn.close()
    return schema

def run_sql(query: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return json.dumps([dict(zip(cols, r)) for r in rows])
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()

def run_sql_agent(question: str):
    schema = get_schema()
    prompt = f"""You are an autonomous SQL Database Agent.
Database Schema:
{schema}

User Question: {question}

Follow this reasoning process:
1. Determine what SQL query answers the question.
2. Execute the query mentally on the schema.
3. Formulate the precise SQL query.
4. Provide the exact answer to the user.

Generate your response in this format:
SQL: <The SELECT query>
ANSWER: <Clear explanation and answer>"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    
    text = response.text
    sql_match = re.search(r"SQL:\s*(SELECT.+?)(?=ANSWER:|$)", text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip().replace("`", "")
        db_result = run_sql(sql)
        print(f"\n[Agent SQL Query]: {sql}")
        print(f"[Database Result]: {db_result}")
    
    ans_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    ans = ans_match.group(1).strip() if ans_match else text.strip()
    print(f"[Agent Answer]: {ans}")

if __name__ == "__main__":
    print("--- Running Task 4: SQL Agent with Tool Use ---")
    run_sql_agent("What is the average salary of employees in the Engineering department?")
    run_sql_agent("Who is the most recently hired employee?")
