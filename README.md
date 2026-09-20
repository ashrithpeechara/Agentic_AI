# Agentic AI Assignment — Applied Agentic AI (LLM & RAG)

A complete 4-task assignment covering LLM workflows, prompt chaining, agentic AI, and RAG-based question answering using **Google Gemini**.

## Project Structure

```
agentic_ai_assignment/
├── assignments/              ← Standalone single-file assignment scripts & specialized agents
│   ├── llm-workflow.py
│   ├── prompt-chaining.py
│   ├── agentic-ai.py
│   ├── rag_ai.py
│   ├── document_rag_agent.py
│   ├── research_agent.py
│   └── security_analysis_agent.py
├── LAB_EXPERIEMENTS/         ← Lab Experiments 1-5 with Gemini & SQLite
│   ├── task1_text_to_sql.py   (Text-to-SQL Workflow)
│   ├── task2_rag_qa.py        (RAG-based QA System)
│   ├── task3_prompt_chaining.py (Multi-Step Prompt Chaining)
│   ├── task4_sql_agent.py     (Autonomous SQL Agent)
│   ├── task5_multi_agent_sdr.py (Multi-Agent SDR Pipeline)
│   └── company.db
├── task1_llm_workflow/       ← Task 1: LLM Chat (CLI + Web UI)
├── task2_prompt_chaining/    ← Task 2: 3-Step Prompt Chain (CLI + Web UI)
├── task3_agentic_ai/         ← Task 3: ReAct Agent with Tools (CLI + Web UI)
├── task4_rag_qa/             ← Task 4: RAG Question Answering (CLI + Web UI)
├── assignment_1_agentic_ai.pdf ← Assignment submission PDF report
└── generate_assignment_pdf.py ← Automated PDF report generator
```

---

## Task 1 — LLM Workflow (5 Marks)

**Goal:** Accept user input and generate a response using Google Gemini.

**Features:**
- Multi-turn conversation with full history
- CLI (`main.py`) + Web UI (`app.py` on port 5000)
- Beautiful dark-mode chat interface with markdown rendering

**Run:**
```bash
cd task1_llm_workflow
pip install -r requirements.txt
# Add your GEMINI_API_KEY to .env
python app.py       # Web UI at http://127.0.0.1:5000
python main.py      # CLI version
python test_task1.py  # Verification tests
```

---

## Task 2 — Prompt Chaining (5 Marks)

**Goal:** Multi-step LLM workflow: Summary → Key Points → Questions.

**Pipeline:**
1. **Step 1** — Generate a comprehensive summary of the topic
2. **Step 2** — Extract 5 key points from the summary
3. **Step 3** — Generate 3 thought-provoking questions from the key points

Each step's output feeds directly into the next step as input (true chaining).

**Run:**
```bash
cd task2_prompt_chaining
pip install -r requirements.txt
python app.py         # Web UI at http://127.0.0.1:5001 (with SSE streaming)
python main.py        # CLI version
python test_task2.py  # Verification tests
```

---

## Task 3 — Agentic AI (5 Marks)

**Goal:** Build an AI agent that plans, executes tools, and delivers a final answer.

**Architecture:** ReAct (Reason + Act) Pattern

**Tools available:**
| Tool | Description |
|------|-------------|
| `calculator` | Evaluates math expressions safely |
| `current_datetime` | Returns the current date/time |
| `wikipedia_search` | Searches Wikipedia (free API) |
| `text_analyzer` | Word/character count analysis |

**Agent Loop:** `THOUGHT → ACTION → OBSERVATION → (repeat) → FINAL_ANSWER`

**Run:**
```bash
cd task3_agentic_ai
pip install -r requirements.txt
python app.py         # Web UI at http://127.0.0.1:5002 (real-time trace)
python main.py        # CLI version
python test_task3.py  # Verification tests
```

---

## Task 4 — RAG-Based Question Answering (5 Marks)

**Goal:** Retrieve relevant information from a PDF/TXT document and answer queries using an LLM.

**RAG Pipeline:**
1. **Load** — Upload PDF or TXT document
2. **Chunk** — Split into overlapping word-based chunks (400 words, 80 overlap)
3. **Embed** — Vectorize chunks with `gemini-embedding-001`
4. **Retrieve** — Cosine similarity search for top-k relevant chunks
5. **Generate** — Gemini answers the question grounded in retrieved context

**Run:**
```bash
cd task4_rag_qa
pip install -r requirements.txt
python app.py         # Web UI at http://127.0.0.1:5003 (drag-and-drop upload)
python main.py        # CLI version
python test_task4.py  # Verification tests
```

---

## Setup (All Tasks)

### 1. Get a Gemini API Key
Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create a free API key.

### 2. Set the API Key
Each task folder has a `.env` file. Add your key:
```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt   # from each task folder
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 3.5 Flash |
| Embeddings | Google gemini-embedding-001 |
| Backend | Python + Flask |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Markdown | marked.js |
| Streaming | Server-Sent Events (SSE) |
| PDF Parsing | pypdf |

---

## Model Used
- **Generation:** `gemini-3.5-flash`
- **Embeddings:** `gemini-embedding-001`
