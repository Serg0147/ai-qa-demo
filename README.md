# AI QA Demo — DeepEval + OpenRouter

Demo for evaluating HR-bot answers with **Faithfulness**, **Answer Relevancy**, and related metrics (DeepEval), using OpenRouter as the judge LLM.

## API key security (keeping the key off AI servers)

- **The key is not sent to Cursor/Claude** as long as it is not in open files or in the chat. The repo code does not contain the key — it only reads the environment variable or your local `.env`.
- **Recommendations:**
  1. **Do not open `.env` in the editor** when using the AI chat, and **do not paste the key into the chat**.
  2. **Exclude `.env` from Cursor** so its contents are not included in context: create a **`.cursorignore`** file in the project root with one line:
     ```
     .env
     ```
     Then Cursor will not index `.env` and will not send its contents to the server.
  3. **Do not commit `.env` to git** — it is already in `.gitignore`, so the repo will not pick up the key.
- **Option without a file:** you can avoid storing the key in `.env` and set it only in the terminal before running (then the key stays only in process memory):
  ```bash
  export OPENROUTER_KEY=sk-or-v1-your_key
  python main.py
  ```
  In that case you do not need a `.env` file with the key.

## Quick start

1. **OpenRouter key**  
   Go to [openrouter.ai](https://openrouter.ai) → Keys → Create Key. Put the key in `.env`:
   ```env
   OPENROUTER_KEY=sk-or-v1-your_key
   ```
   Do not leave placeholders like `REPLACE_ME` or `YOUR_KEY` — use your real key.

2. **Environment and run**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

3. Results appear in the console and in `reports/results_YYYY-MM-DD_HH-MM.csv`.

**Detailed guide:** [GUIDE_TEST_CASES.md](GUIDE_TEST_CASES.md) — how to load test cases, CSV format, evaluation flow, and working with reports.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Evaluation script: loads document + dataset, runs metrics, writes report |
| `.env` | `OPENROUTER_KEY` (do not commit) |
| `document.txt` | Policy/document text (knowledge base) |
| `dataset.csv` | Test cases: `input;expected_output;actual_output;category` (delimiter `;`) |

## If you see "invalid JSON" errors

The judge model sometimes returns invalid JSON. In `main.py`, in the `OpenRouterLLM` class, switch to a more reliable model, for example:

```python
model="google/gemini-2.0-flash-exp:free"
```

## Optional: Streamlit UI

Run the web UI (upload CSV/document, edit table, run evaluation with progress):

```bash
source .venv/bin/activate
streamlit run app.py
```

If you see **PermissionError: Operation not permitted: '.../.streamlit'**, create the directory so Streamlit can write its config/cache:

```bash
mkdir -p ~/.streamlit
```

Then run `streamlit run app.py` again.

## DeepTeam red teaming (step-by-step, no UI)

This project now includes a backend-only DeepTeam setup that uses OpenRouter for the target model callback.

### 0) Python version requirement for DeepTeam

`deepteam` currently supports Python `< 3.14`. If your current venv is Python 3.14, create a dedicated 3.13 venv for red teaming:

```bash
python3.13 -m venv .venv-deepteam
source .venv-deepteam/bin/activate
```

You can also use the helper script (auto-creates `.venv-deepteam`):

```bash
./scripts/run_deepteam.sh
```

### 1) Install dependencies

```bash
source .venv-deepteam/bin/activate   # or your Python<=3.13 env
pip install -r requirements.txt
```

### 2) Set keys

- Target app model key (OpenRouter):

Create `.env` from example first:

```bash
cp .env.example .env
```

```bash
export OPENROUTER_KEY=sk-or-v1-your_key
```

- DeepTeam generator/evaluator key (required by DeepTeam internals):

```bash
deepteam set-api-key sk-proj-your_openai_or_other_supported_key
```

Note: `OPENROUTER_KEY` is used by your target model (`model_callback`). DeepTeam still needs its own API key for attack generation + evaluation.

### 3) Run first red team (Python API flow)

```bash
python red_team_llm.py
```

This runs:
- Vulnerability: `Bias(types=["race"])`
- Attack: `PromptInjection()`

And saves JSON output to `reports/deepteam_results_YYYY-MM-DD_HH-MM-SS.json`.

### 4) Run with CLI + YAML config

```bash
deepteam run deepteam.yaml
```

Using helper script:

```bash
./scripts/run_deepteam.sh cli
```

Optional overrides:

```bash
deepteam run deepteam.yaml -c 20 -a 5 -o reports
```

### 5) Files added for this integration

- `red_team_llm.py` — async `model_callback` and first red-team run
- `openrouter_target_model.py` — custom DeepEval model class for OpenRouter
- `deepteam.yaml` — CLI config for repeatable runs
- `.env.example` — example OpenRouter settings
- `scripts/run_deepteam.sh` — one-command setup + run
