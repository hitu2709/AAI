# AAI8 — Experiment No. 8: Tool-Enabled AI Agent

This experiment builds a tool-enabled AI agent with LangChain, uses Groq as the language model provider, traces execution with LangSmith, and compares two prompt configurations.

## Project files

- `main.py` — runs ten benchmark queries against Prompt V1 and Prompt V2.
- `tools.py` — calculator and student-information tools.
- `prompts.py` — the two system prompts.
- `evaluation.py` — writes benchmark CSV files and compares metrics.
- `requirements.txt` — Python dependencies.
- `.env.example` — environment variable template. Copy it to `.env` and add your own keys locally.
- `results/` — generated CSV results.

## Setup (Windows PowerShell)

From the `AAI8` directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally and provide your own Groq and LangSmith keys:

```dotenv
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Experiment-8-Tool-Agent
```

Never commit the actual `.env` file or disclose API keys.

Run the benchmark:

```powershell
python main.py
```

Generated files:

- `results/prompt_v1_results.csv`
- `results/prompt_v2_results.csv`
- `results/comparison.csv`

## Evaluation notes

The benchmark reports task success, tool selection accuracy, average latency, and token usage when available from the model response. Inspect LangSmith traces to verify tool calls and token counts. Review outputs manually before using results in a practical report.

The student data in `tools.py` is demonstration data intended for the experiment; update it if required for your coursework.
