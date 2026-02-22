"""
AI QA Demo: оценка ответов HR-бота через DeepEval (Faithfulness, Answer Relevancy).
Использует OpenRouter как LLM для метрик.
"""
import os
import csv
import json
from pathlib import Path
from datetime import datetime

# Отключаем телеметрию DeepEval до загрузки библиотек
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

DEBUG_LOG = Path(__file__).parent / ".cursor" / "debug-9215d5.log"
def _dlog(msg: str, data: dict, hypothesis_id: str = ""):
    try:
        payload = {"sessionId": "9215d5", "location": "main.py", "message": msg, "data": data, "timestamp": int(datetime.now().timestamp() * 1000), "hypothesisId": hypothesis_id}
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

load_dotenv()
# #region agent log
_dlog("after load_dotenv", {"cwd": os.getcwd(), "env_path_exists": (Path(__file__).parent / ".env").exists(), "key_len": len((os.environ.get("OPENROUTER_KEY") or "").strip()), "key_start": ((os.environ.get("OPENROUTER_KEY") or "").strip()[:6] if (os.environ.get("OPENROUTER_KEY") or "").strip() else "empty")}, "A")
# #endregion

# Пути к данным
DOCUMENT_PATH = Path(__file__).parent / "document.txt"
DATASET_PATH = Path(__file__).parent / "dataset.csv"
REPORTS_DIR = Path(__file__).parent / "reports"


class OpenRouterLLM(DeepEvalBaseLLM):
    """Кастомный LLM для DeepEval через OpenRouter (LangChain)."""
    def __init__(self):
        api_key = (os.environ.get("OPENROUTER_KEY") or "").strip()
        # частый косяк: лишняя буква в начале (например ssk-or вместо sk-or)
        if api_key.startswith("ssk-or"):
            api_key = api_key[1:]
        # #region agent log
        _dlog("OpenRouterLLM.__init__", {"key_len_after_fix": len(api_key), "key_start": (api_key[:6] if api_key else "empty")}, "B")
        # #endregion
        self._model = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="arcee-ai/trinity-large-preview:free",
            default_headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )

    def load_model(self):
        return self._model

    def generate(self, prompt: str, **kwargs):
        # DeepEval вызывает generate(prompt, schema=...) и ждёт объект схемы (с .statements и т.д.), не строку
        # #region agent log
        if not getattr(self, "_generate_called", False):
            self._generate_called = True
            _dlog("generate() first call", {"prompt_len": len(prompt), "has_key": bool((os.environ.get("OPENROUTER_KEY") or "").strip())}, "C,E")
        # #endregion
        chat_model = self.load_model()
        try:
            out = chat_model.invoke(prompt).content
        except Exception as e:
            # #region agent log
            _dlog("generate() exception", {"exc_type": type(e).__name__, "exc_msg": str(e)[:200]}, "C,E")
            # #endregion
            raise
        schema = kwargs.get("schema")
        if schema is not None:
            # Извлекаем JSON из ответа (модель может обернуть в ```json ... ```)
            text = (out or "").strip()
            for marker in ("```json", "```"):
                if marker in text:
                    idx = text.find(marker)
                    start = idx + len(marker)
                    end = text.find("```", start)
                    text = text[start:end] if end != -1 else text[start:]
                    break
            text = text.strip()
            try:
                data = json.loads(text)
                return schema.model_validate(data) if hasattr(schema, "model_validate") else schema(**data)
            except (json.JSONDecodeError, Exception):
                pass
        return out

    async def a_generate(self, prompt: str, **kwargs):
        chat_model = self.load_model()
        try:
            result = await chat_model.ainvoke(prompt)
            out = result.content
        except Exception:
            return self.generate(prompt, **kwargs)
        schema = kwargs.get("schema")
        if schema is not None:
            text = (out or "").strip()
            for marker in ("```json", "```"):
                if marker in text:
                    idx = text.find(marker)
                    start = idx + len(marker)
                    end = text.find("```", start)
                    text = text[start:end] if end != -1 else text[start:]
                    break
            text = text.strip()
            try:
                data = json.loads(text)
                return schema.model_validate(data) if hasattr(schema, "model_validate") else schema(**data)
            except (json.JSONDecodeError, Exception):
                pass
        return out

    def get_model_name(self) -> str:
        return "OpenRouter-Trinity-Large"


def load_document() -> str:
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_test_cases(doc_text: str) -> list[tuple[LLMTestCase, str]]:
    """Читает dataset.csv (разделитель ;), возвращает список (LLMTestCase, category)."""
    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            tc = LLMTestCase(
                input=(row.get("input") or "").strip(),
                expected_output=(row.get("expected_output") or "").strip(),
                actual_output=(row.get("actual_output") or "").strip(),
                retrieval_context=[doc_text],
            )
            category = (row.get("category") or "").strip()
            cases.append((tc, category))
    return cases


def main():
    api_key = (os.environ.get("OPENROUTER_KEY") or "").strip()
    # #region agent log
    _dlog("main key check", {"key_len": len(api_key), "key_start": (api_key[:6] if api_key else "empty"), "has_placeholder": "ВСТАВЬ" in api_key, "key_check_passed": bool(api_key and "ВСТАВЬ" not in api_key)}, "B,D")
    # #endregion
    if not api_key or "ВСТАВЬ" in api_key:
        print("⚠️  Заполни OPENROUTER_KEY в файле .env (ключ с openrouter.ai)")
        return

    try:
        doc_text = load_document()
        test_data = load_test_cases(doc_text)
    except Exception as e:
        # #region agent log
        _dlog("load_document or load_test_cases failed", {"exc_type": type(e).__name__, "exc_msg": str(e)[:200]}, "D")
        # #endregion
        raise
    if not test_data:
        print("⚠️  Нет тест-кейсов в dataset.csv")
        return

    custom_llm = OpenRouterLLM()
    # #region agent log
    _dlog("OpenRouterLLM created", {"model_type": type(custom_llm._model).__name__}, "B,E")
    # #endregion
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        async_mode=False,
        include_reason=True,
        model=custom_llm,
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        async_mode=False,
        include_reason=True,
        model=custom_llm,
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = REPORTS_DIR / f"results_{timestamp}.csv"
    report_rows = []

    print("🚀 Запускаю оценку... (~3–5 минут)\n")
    print(f"{'№':<4} {'Вопрос':<38} {'Relev.':<7} {'Faith.':<7} {'Итог':<6} Причина")
    print("─" * 95)

    passed = 0
    failed = 0

    for i, (test_case, category) in enumerate(test_data, 1):
        # #region agent log
        if i == 1:
            _dlog("first measure() call", {"test_index": i}, "C,D,E")
        # #endregion
        relevancy_metric.measure(test_case)
        faithfulness_metric.measure(test_case)

        r_score = getattr(relevancy_metric, "score", 0.0) or 0.0
        f_score = getattr(faithfulness_metric, "score", 0.0) or 0.0
        r_reason = getattr(relevancy_metric, "reason", "") or ""
        f_reason = getattr(faithfulness_metric, "reason", "") or ""

        r_ok = r_score >= relevancy_metric.threshold
        f_ok = f_score >= faithfulness_metric.threshold
        overall_pass = r_ok and f_ok
        if overall_pass:
            passed += 1
        else:
            failed += 1

        short_input = (test_case.input[:36] + "…") if len(test_case.input) > 37 else test_case.input
        icon = "✅ PASS" if overall_pass else "❌ FAIL"
        # Краткая причина в таблице: для FAIL — первые 45 символов, для PASS — прочерк
        if overall_pass:
            reason_short = "—"
        else:
            parts = []
            if not r_ok and r_reason:
                parts.append("R: " + (r_reason.replace(chr(10), " ")[:42] + "…" if len(r_reason) > 42 else r_reason.replace(chr(10), " ")))
            if not f_ok and f_reason:
                parts.append("F: " + (f_reason.replace(chr(10), " ")[:42] + "…" if len(f_reason) > 42 else f_reason.replace(chr(10), " ")))
            reason_short = " | ".join(parts)[:70] + ("…" if len(" | ".join(parts)) > 70 else "") if parts else "(нет причины)"
        print(f"{i:<4} {short_input:<38} {r_score:<7.2f} {f_score:<7.2f} {icon:<6} {reason_short}")
        if not overall_pass and (r_reason or f_reason):
            max_len = 300
            if not r_ok and r_reason:
                rr = (r_reason[:max_len] + "…") if len(r_reason) > max_len else r_reason
                print(f"       └ Relevancy: {rr.replace(chr(10), ' ')}")
            if not f_ok and f_reason:
                fr = (f_reason[:max_len] + "…") if len(f_reason) > max_len else f_reason
                print(f"       └ Faithfulness: {fr.replace(chr(10), ' ')}")
        elif not overall_pass and not (r_reason or f_reason) and i == 1:
            print("       └ (причины пустые: модель не вернула reason в JSON — смотри CSV)")

        report_rows.append({
            "date": timestamp,
            "test_id": i,
            "category": category,
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "relevancy_score": r_score,
            "faithfulness_score": f_score,
            "relevancy_reason": r_reason.replace("\n", " "),
            "faithfulness_reason": f_reason.replace("\n", " "),
            "overall_pass_fail": "PASS" if overall_pass else "FAIL",
        })

    total = passed + failed
    pct = (100 * passed / total) if total else 0
    print("─" * 95)
    print(f"\n📊 ИТОГО: {passed} прошли ✅  |  {failed} провалились ❌")
    print(f"📈 Процент успеха: {pct:.0f}%")
    print(f"💾 Результаты сохранены: {report_path}")
    print("   (колонки relevancy_reason и faithfulness_reason — полные причины по каждому кейсу)")

    fieldnames = [
        "date", "test_id", "category", "input", "actual_output",
        "relevancy_score", "faithfulness_score",
        "relevancy_reason", "faithfulness_reason",
        "overall_pass_fail",
    ]
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


if __name__ == "__main__":
    main()
