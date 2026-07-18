# Task 1 — LLM Workflow

## 📌 Objective
Build a Python program that accepts user input and generates a response using **Google Gemini**.

## 📁 Files
| File | Purpose |
|------|---------|
| `main.py` | Core application — multi-turn chat with Gemini |
| `.env` | Stores your API key (never commit this!) |
| `requirements.txt` | Python dependencies |

## ⚙️ Setup

### 1. Get a Gemini API Key
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a free API key

### 2. Set the API Key
Open `.env` and replace the placeholder:
```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Program
```bash
python main.py
```

## 💬 Usage
```
═══════════════════════════════════════════════════════════
       🤖  LLM Workflow — Google Gemini Chat
═══════════════════════════════════════════════════════════
  Type your message and press Enter to chat.
  Commands:  'quit' or 'exit' → stop  |  'clear' → reset
═══════════════════════════════════════════════════════════

You [1]: What is machine learning?
⏳ Thinking…

────────────────────────────────────────────────────────────
🤖 Gemini:
────────────────────────────────────────────────────────────
  Machine learning is a subset of AI where systems learn from data...
────────────────────────────────────────────────────────────
```

## 🔑 Key Features
- **Multi-turn conversation** — Gemini remembers previous messages
- **System prompt** — Assistant is tuned to be helpful and concise
- **Commands** — `clear` resets history, `quit`/`exit` ends session
- **Error handling** — API errors are caught and displayed gracefully
