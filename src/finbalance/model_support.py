"""Configuration and public context shared by optional model adapters."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def load_environment() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)


def llm_configuration() -> dict[str, str]:
    load_environment()
    names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Missing LLM configuration: " + ", ".join(missing)
                         + ". Copy .env_example to .env and fill in these values.")
    return {name: os.environ[name].strip() for name in names}


def make_llm_model(config: dict[str, str], timeout: float, *, client_factory=AsyncOpenAI,
                   provider_factory=OpenAIProvider, model_factory=OpenAIChatModel):
    client = client_factory(base_url=config["LLM_BASE_URL"], api_key=config["LLM_API_KEY"],
                            timeout=timeout, max_retries=0)
    return model_factory(config["LLM_MODEL"], provider=provider_factory(openai_client=client)), client


def public_chart() -> list[dict]:
    from .coa import build_chart_of_accounts
    return [{**asdict(a), "type": a.type.value, "normal_side": a.normal_side.value}
            for a in build_chart_of_accounts()]


def chart_context() -> str:
    return json.dumps(public_chart())
