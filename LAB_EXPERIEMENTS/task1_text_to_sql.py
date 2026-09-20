import os
import sqlite3
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

db_path = "employees.db"

def setup_database():
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

setup_database()

def get_schema():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(employees)")
    columns = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
    conn.close()
    return f"Table employees: {', '.join(columns)}"

def run_text_to_sql(question: str):
    schema = get_schema()
    prompt = f"""You are a SQL expert.
Database Schema:
{schema}

User Question: {question}

Convert this question into a valid SQLite SQL query.
Return ONLY the raw SQL query without markdown code blocks or explanations."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    sql = response.text.strip()
    sql = re.sub(r"```(?:sql)?\s*", "", sql).replace("```", "").strip()

    print(f"\n[Question]: {question}")
    print(f"[Generated SQL]: {sql}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        print(f"[SQL Results]: {results}")

        synth_prompt = f"""Question: {question}
SQL Query: {sql}
Database Results: {results}

Provide a clear, human-readable answer to the question based strictly on the results."""
        synth_resp = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=synth_prompt
        )
        print(f"[Answer]: {synth_resp.text.strip()}")
    except Exception as e:
        print(f"[Error]: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("--- Running Task 1: Text-to-SQL Workflow ---")
    run_text_to_sql("How many employees are in the Engineering department?")
    run_text_to_sql("Who is the highest paid employee and what is their salary?")
