"""The baseline tool loop with native JEV decisions and GPT handoffs."""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace

from pydantic_ai import ToolOutput
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models import Model
from pydantic_ai.models.decision import DecisionHandOff
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from pydantic_ai.usage import UsageLimits
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .harness import AgentSession
from .llm_agent import build_agent
from .model_support import llm_configuration, load_environment, make_llm_model, public_chart
from .verifier import Submission

REQUEST_LIMIT = 40
REQUEST_TIMEOUT_SECONDS = 60
RUN_TIMEOUT_SECONDS = 300


@dataclass
class ProviderRequests:
    jev: int = 0
    gpt: int = 0

    def spend(self, provider: str):
        if self.jev + self.gpt >= REQUEST_LIMIT:
            raise UsageLimitExceeded(f"JEV agent exceeded {REQUEST_LIMIT} actual provider requests")
        setattr(self, provider, getattr(self, provider) + 1)


ACTIVE_REQUESTS: ContextVar[ProviderRequests] = ContextVar("jev_provider_requests")


class CountedTypeSafeModel(TypeSafeModel):
    async def decide(self, request, model_settings):
        # Native routing can make another SDK call to fill the chosen arguments.
        ACTIVE_REQUESTS.get().spend("jev")
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            return await super().decide(request, model_settings)


class CountedModel(WrapperModel):
    def __init__(self, model, provider):
        super().__init__(model)
        self.provider_name = provider

    async def request(self, messages, model_settings, model_request_parameters):
        ACTIVE_REQUESTS.get().spend(self.provider_name)
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            return await super().request(messages, model_settings, model_request_parameters)


def prepare_read_tools(ctx, tool_defs):
    """Bound read arguments using the chart and document IDs already returned."""
    documents = []
    for message in ctx.messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "list_documents":
                documents = list(part.content)
    prepared = []
    for tool in tool_defs:
        schema = deepcopy(tool.parameters_json_schema)
        if tool.name == "query_ledger":
            schema["properties"]["account"] = {
                "anyOf": [{"type": "string", "enum": [a["code"] for a in public_chart()]}, {"type": "null"}],
                "default": None, "description": "Account to inspect, or null for the complete ledger.",
            }
        elif tool.name == "read_document":
            if not documents:
                continue
            schema["properties"]["doc_id"] = {
                "type": "string", "enum": documents, "description": "Which listed statement or policy should be read?",
            }
        prepared.append(replace(tool, parameters_json_schema=schema))
    return prepared


@dataclass(frozen=True)
class JevSettings:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"
    allow_llm: bool = True

    def __post_init__(self):
        if not self.base_url or not self.model:
            raise ValueError("TYPESAFE_BASE_URL and TYPESAFE_DEFAULT_MODEL must not be empty")

    @classmethod
    def from_env(cls):
        load_environment()
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise ValueError("Missing JEV configuration: TYPESAFE_API_KEY. Copy .env_example to .env and fill it in.")
        enabled = os.environ.get("JEV_ALLOW_LLM_ESCALATION", "true").strip().lower()
        if enabled not in ("true", "false"):
            raise ValueError("JEV_ALLOW_LLM_ESCALATION must be true or false")
        return cls(api_key=key, base_url=os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").strip(),
                   model=os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest").strip(), allow_llm=enabled == "true")


class JevAgent:
    def __init__(self, *, model: Model | None = None, llm_model: Model | None = None,
                 settings: JevSettings | None = None):
        self.settings = settings or (JevSettings.from_env() if model is None else JevSettings(allow_llm=llm_model is not None))
        self._clients = []
        config = llm_configuration() if self.settings.allow_llm and llm_model is None else None
        if model is None:
            client = AsyncTypeSafeClient(api_key=self.settings.api_key, base_url=self.settings.base_url,
                                         retry=RetryPolicy(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS),
                                         timeout=REQUEST_TIMEOUT_SECONDS)
            self._clients.append(client)
            model = CountedTypeSafeModel(self.settings.model, provider=TypeSafeProvider(typesafe_client=client))
        elif isinstance(model, TypeSafeModel):
            model = CountedTypeSafeModel(model.model_name, provider=model.provider,
                                         profile=model.profile, settings=model.settings)
        else:
            model = CountedModel(model, "jev")
        if config is not None:
            llm_model, client = make_llm_model(config, REQUEST_TIMEOUT_SECONDS)
            self._clients.append(client)
        if self.settings.allow_llm:
            model = FallbackModel(model, CountedModel(llm_model, "gpt"), fallback_on=DecisionHandOff)
        self._agent = build_agent(model, capabilities=[PrepareTools(prepare_read_tools)],
            text_output=ToolOutput(str, name="finish_without_submission",
                description="Explain why reconciliation cannot be submitted; ends this attempt as a failure."))
        self.requests: ProviderRequests | None = None

    def run(self, session: AgentSession) -> Submission:
        return asyncio.run(self._run(session))

    async def _run(self, session: AgentSession) -> Submission:
        self.requests = requests = ProviderRequests()
        token = ACTIVE_REQUESTS.set(requests)
        try:
            async with AsyncExitStack() as stack:
                for client in self._clients:
                    await stack.enter_async_context(client)
                async with asyncio.timeout(RUN_TIMEOUT_SECONDS):
                    result = await self._agent.run(
                        "Reconcile the company-month and submit your final balances.", deps=session,
                        usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
                    )
            if not isinstance(result.output, Submission):
                raise RuntimeError("JEV agent finished without calling submit_reconciliation")
            return result.output
        except UsageLimitExceeded as exc:
            raise UsageLimitExceeded(f"{exc}; JEV={requests.jev}, GPT={requests.gpt}") from exc
        finally:
            ACTIVE_REQUESTS.reset(token)
