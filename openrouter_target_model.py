import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM

load_dotenv()


class OpenRouterTargetLLM(DeepEvalBaseLLM):
    def __init__(self):
        api_key = (os.getenv("OPENROUTER_KEY") or "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_KEY is missing. Add it to .env or export it in your shell.")
        if api_key.startswith("ssk-or"):
            api_key = api_key[1:]

        model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self._model_name = f"OpenRouter-{model_name}"
        self._chat_model = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self):
        return self

    def generate(self, prompt: str, **kwargs):
        response = self._chat_model.invoke(prompt)
        out = response.content

        schema = kwargs.get("schema")
        if schema is None:
            return str(out)

        text = (str(out) or "").strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)
        return schema.model_validate(data) if hasattr(schema, "model_validate") else schema(**data)

    async def a_generate(self, prompt: str, **kwargs):
        response = await self._chat_model.ainvoke(prompt)
        out = response.content

        schema = kwargs.get("schema")
        if schema is None:
            return str(out)

        text = (str(out) or "").strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)
        return schema.model_validate(data) if hasattr(schema, "model_validate") else schema(**data)
