# AI QA Demo — DeepEval + OpenRouter

Демо для оценки ответов HR-бота метриками **Faithfulness** и **Answer Relevancy** (DeepEval), с судьёй на OpenRouter.

## Безопасность ключа (чтобы ключ не уходил на серверы AI)

- **Ключ не уходит в Cursor/Claude**, если его нет в открытых файлах и в чате. Код в репозитории ключа не содержит — он только читает переменную окружения или `.env` у тебя локально.
- **Рекомендации:**
  1. **Не открывай `.env` в редакторе**, когда пользуешься AI-чатом, и **не вставляй ключ в чат**.
  2. **Добавь `.env` в исключения Cursor**, чтобы содержимое файла не попадало в контекст: создай в корне проекта файл **`.cursorignore`** с одной строкой:
     ```
     .env
     ```
     Тогда Cursor не будет индексировать `.env` и не отправит его содержимое на сервер.
  3. **Не коммить `.env` в git** — он уже в `.gitignore`, репозиторий ключ не подхватит.
- **Вариант без файла:** можно не хранить ключ в `.env`, а задавать только в терминале перед запуском (тогда ключ только в памяти процесса):
  ```bash
  export OPENROUTER_KEY=sk-or-v1-твой_ключ
  python main.py
  ```
  В этом случае файл `.env` с ключом не нужен.

## Быстрый старт

1. **Ключ OpenRouter**  
   Зайди на [openrouter.ai](https://openrouter.ai) → Keys → Create Key. Вставь ключ в `.env`:
   ```env
   OPENROUTER_KEY=sk-or-v1-твой_ключ
   ```

2. **Окружение и запуск**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

3. Результаты появятся в консоли и в `reports/results_YYYY-MM-DD_HH-MM.csv`.

**Подробный гайд:** [GUIDE_TEST_CASES.md](GUIDE_TEST_CASES.md) — как загружать тест-кейсы, формат CSV, флоу оценки и работа с отчётами.

## Файлы

| Файл | Назначение |
|------|------------|
| `main.py` | Скрипт оценки: загрузка document + dataset, метрики, отчёт |
| `.env` | `OPENROUTER_KEY` (не коммитить) |
| `document.txt` | Текст политики/документа (база знаний) |
| `dataset.csv` | Тест-кейсы: `input;expected_output;actual_output;category` (разделитель `;`) |

## Если ошибка «invalid JSON»

Модель-судья иногда возвращает невалидный JSON. В `main.py` в классе `OpenRouterLLM` смени модель на более устойчивую, например:

```python
model="google/gemini-2.0-flash-exp:free"
```

## Опционально: Streamlit UI

После того как `main.py` стабильно работает, можно добавить веб-интерфейс (см. план в Части 5) и запускать:

```bash
streamlit run app.py
```
