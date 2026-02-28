import asyncio
import inspect
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepteam import red_team
from deepteam.attacks.single_turn import PromptInjection
from deepteam.vulnerabilities import Bias

load_dotenv()

REPORTS_DIR = Path(__file__).parent / "reports"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def _build_openrouter_chat_model() -> ChatOpenAI:
    api_key = (os.getenv("OPENROUTER_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_KEY is missing. Add it to .env or export it in your shell.")
    if api_key.startswith("ssk-or"):
        api_key = api_key[1:]

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=DEFAULT_MODEL,
        default_headers={"Authorization": f"Bearer {api_key}"},
    )


async def model_callback(input: str) -> str:
    model = _build_openrouter_chat_model()
    response = await model.ainvoke(input)
    content = response.content

    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    chunks.append(str(text))
            else:
                chunks.append(str(part))
        return "".join(chunks).strip()

    return str(content).strip()


async def run_first_red_team():
    bias = Bias(types=["race"])
    prompt_injection = PromptInjection()

    risk_assessment = red_team(
        model_callback=model_callback,
        vulnerabilities=[bias],
        attacks=[prompt_injection],
    )

    if inspect.isawaitable(risk_assessment):
        risk_assessment = await risk_assessment

    return risk_assessment


def _to_jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return {"repr": repr(value)}


async def _main():
    print("Starting DeepTeam red-team run (Bias + PromptInjection)...")
    assessment = await run_first_red_team()

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = REPORTS_DIR / f"deepteam_results_{stamp}.json"

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_jsonable(assessment), file_obj, indent=2, ensure_ascii=False)

    print(f"Red-team completed. Saved report to: {output_path}")


if __name__ == "__main__":
    asyncio.run(_main())
