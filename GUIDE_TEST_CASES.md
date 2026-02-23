# Guide: test cases and evaluation flow

How to load test cases, what format they should use, and how to work with them in AI QA Demo.

---

## 1. Overall flow

```
document.txt          →  Knowledge base (single document)
        ↓
dataset.csv           →  Test case table (question + bot answer + category)
        ↓
main.py               →  Reads both files, creates LLMTestCase per CSV row
        ↓
DeepEval metrics      →  Answer Relevancy, Faithfulness, etc. score each case
        ↓
reports/results_*.csv →  Output: scores, reasons, PASS/FAIL
```

**Idea:** you have a document (policy, instructions) and bot answers to questions. The script checks: (1) the answer is on-topic (Relevancy), (2) the answer does not contradict the document (Faithfulness). Test cases are the “question + bot answer” pairs you prepare in CSV.

---

## 2. Test case file format

**File:** `dataset.csv` in the project root (next to `main.py`).

**Column delimiter:** semicolon `;` (not comma), so text can contain commas.

**Encoding:** UTF-8.

**First row — headers (required):**

```text
input;expected_output;actual_output;category
```

| Column            | Required | Description |
|-------------------|----------|-------------|
| `input`           | Yes      | User question (what was asked). |
| `expected_output` | No*      | Ideal answer (for your analysis; Relevancy/Faithfulness metrics do not use it). |
| `actual_output`   | Yes      | Bot answer being evaluated. |
| `category`        | No*      | Your label: `good` / `hallucination` / `irrelevant`. Used only in the report and for filtering. |

\* These columns are not required for metric calculation but must exist in the CSV. You can leave them empty or use a dash.

---

## 3. How the script loads test cases

In `main.py` the flow is:

1. **Read document**  
   All text from `document.txt` is loaded into `doc_text`.

2. **Read CSV**  
   `dataset.csv` is opened with delimiter `;`. Each row (except the header) is one test case.

3. **Create LLMTestCase per row**  
   For each CSV row, a DeepEval object is created with:
   - `input` — from column `input`;
   - `expected_output` — from column `expected_output`;
   - `actual_output` — from column `actual_output`;
   - `retrieval_context` — **same for all**: `[doc_text]` (the full document).

4. **Evaluation**  
   For each case, metrics (Answer Relevancy, Faithfulness, etc.) are run. They use `input`, `actual_output`, and `retrieval_context` (the document).

So: **test cases = rows in `dataset.csv`**. Add a row → new case. Remove a row → that case is not run.

---

## 4. Categories (good / hallucination / irrelevant)

You use these for labeling and report analysis. The script only stores `category` in the report; metrics are not computed from it.

- **good** — bot answer is correct and grounded in the document. Expect PASS on the metrics.
- **hallucination** — bot answer invents facts not in the document (or contradicts it). Expect low Faithfulness, often FAIL.
- **irrelevant** — answer is off-topic (bot answered the wrong thing). Expect low Answer Relevancy, often FAIL.

That way you can see in the report: “this case I labeled good — did it pass?” or “this hallucination — did the metrics correctly fail it?”.

---

## 5. How to prepare test cases

### Option A: Manually in an editor

1. Open `dataset.csv` in Excel, Google Sheets, or Cursor (as text).
2. Ensure the delimiter is `;` and the first row is the header.
3. Add rows: question in `input`, bot answer in `actual_output`, and optionally `expected_output` and `category`.
4. Save as UTF-8 (in Excel: “Save as” → CSV UTF-8 or export with `;` as delimiter).

If text contains semicolons or newlines, wrap the value in double quotes (standard CSV).

### Option B: Generate with ChatGPT / Gemini (as in your plan)

1. Prepare the document text (or paste from `document.txt`).
2. Ask the model to generate 15 test cases in three groups: 5 good, 5 hallucination, 5 irrelevant.
3. Specify the format: “Return only CSV, delimiter semicolon, first row: input;expected_output;actual_output;category”.
4. Paste the result into `dataset.csv` (replacing old content or appending).

### Option C: Export from your system

If test cases live in another system (spreadsheet, database), export to CSV with columns `input`, `expected_output`, `actual_output`, `category` and delimiter `;`, then save as `dataset.csv` in the project folder.

---

## 6. Step-by-step workflow

1. **Put the document in `document.txt`**  
   One file = one “source of truth”. All Faithfulness evaluation is relative to this text.

2. **Fill or update `dataset.csv`**  
   Each row = one test case (question + bot answer + optional category).

3. **Run the evaluation**  
   ```bash
   source .venv/bin/activate
   python main.py
   ```

4. **Check console output**  
   Table: case number, question (truncated), Relevancy, Faithfulness, Tox., Bias, PASS/FAIL.

5. **Open the report**  
   File `reports/results_YYYY-MM-DD_HH-MM.csv`. It has full text, scores, reasons, and `category`. Use it to see which cases failed and why.

6. **Adjust test cases if needed**  
   Edit `dataset.csv` (or `document.txt`) and run `python main.py` again. A new report is created with a new timestamp in the filename.

---

## 7. Important notes

- **One document for all cases.** Currently `retrieval_context` is the same for every row — the full `document.txt`. To use different documents per case, you’d need to change the code (e.g. add a `document_id` column or file path and pass different context).
- **File name and path.** The script reads test cases only from `dataset.csv` in the project folder. For another file or name, change `DATASET_PATH` in `main.py`.
- **Empty rows.** Empty rows in the CSV can yield empty `input` or `actual_output`; such cases are better removed or avoided — otherwise metrics may behave unpredictably.
- **Quotes in CSV.** If a cell contains `;` or a newline, wrap the value in double quotes `"..."`. Standard `csv.DictReader` will handle them.

---

## 8. Pre-run checklist

- [ ] `document.txt` is filled (knowledge base).
- [ ] `dataset.csv` is UTF-8, delimiter `;`, header row is `input;expected_output;actual_output;category`.
- [ ] Each row has at least `input` and `actual_output` filled.
- [ ] `.env` has a valid `OPENROUTER_KEY` (no `REPLACE_ME` or `YOUR_KEY`).
- [ ] Run: `source .venv/bin/activate` → `python main.py`.

After that, test cases are loaded automatically from `dataset.csv`; you work with them by editing that file and re-running the script.
