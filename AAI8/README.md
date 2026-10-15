<<<<<<< HEAD
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
=======
# Experiment No. 8 — Tool-Enabled AI Agent using LangChain and LangSmith

## Aim
Build a tool-enabled AI agent, trace execution with LangSmith, and evaluate task success,
tool-calling accuracy, latency, and token consumption across two prompt versions.

## Requirements
- Python 3.10 or 3.11 recommended
- OpenAI API key (model access and billing may be required)
- LangSmith API key

## Setup in VS Code (Windows PowerShell)
>>>>>>> 72f20c1 (Add AAI8 Experiment 8)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
<<<<<<< HEAD
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

=======
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and replace the placeholder values with your keys. Never upload `.env` to GitHub.

Run:
>>>>>>> 72f20c1 (Add AAI8 Experiment 8)
```powershell
python main.py
```

<<<<<<< HEAD
Generated files:

=======
## Google Colab alternative
Upload/extract the project files into the Colab session, then run:
```python
!pip install -r requirements.txt
```
Set `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and
`LANGSMITH_PROJECT=Experiment-8-Tool-Agent` in the Colab Secrets or environment before running:
```python
!python main.py
```
You can also execute the Python modules from notebook cells after placing them in the working directory.

## LangSmith
1. Create/sign in to a LangSmith account.
2. Copy your LangSmith API key into `.env`.
3. Keep `LANGSMITH_TRACING=true`.
4. Run `main.py`; inspect the project named `Experiment-8-Tool-Agent`.
5. For each trace, check the selected tool, input arguments, tool result, final answer, latency, and token usage.

## Output files
>>>>>>> 72f20c1 (Add AAI8 Experiment 8)
- `results/prompt_v1_results.csv`
- `results/prompt_v2_results.csv`
- `results/comparison.csv`

<<<<<<< HEAD
## Evaluation notes

The benchmark reports task success, tool selection accuracy, average latency, and token usage when available from the model response. Inspect LangSmith traces to verify tool calls and token counts. Review outputs manually before using results in a practical report.

The student data in `tools.py` is demonstration data intended for the experiment; update it if required for your coursework.
=======
The benchmark values are measured when you run the code; do not report invented values.
The automatic success check is a first-pass heuristic. Review outputs and traces manually and
correct any success labels before submitting your lab record.

## Notes
- `tools.py` contains demonstration student records. Change these only if your experiment needs different sample records.
- The calculator uses a restricted arithmetic parser and does not call Python `eval()`.
- The token callback reports tokens for supported model calls. Provider/model and package versions can affect the fields available.
- Add `.env` to `.gitignore` before pushing the project to GitHub.
>>>>>>> 72f20c1 (Add AAI8 Experiment 8)
