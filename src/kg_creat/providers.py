"""Elicitation providers other than OpenRouter.

The pool was elicited through OpenRouter and `src.dat_eval.llm` speaks that dialect. Two later runs
need different endpoints, and neither may borrow the ``LLM_BASE_URL`` switch: that flag means "local
serving, therefore free" to `run_elicit`/`score`, and setting it would silently disable the budget cap
and the cost ledger -- and, because `score.py` builds its judge client from the same helper, would
also drag the judge panel off OpenRouter.

  openai_compatible   a LiteLLM gateway. Same wire format as OpenAI, but reasoning effort is a
                      TOP-LEVEL ``reasoning_effort`` rather than OpenRouter's ``extra_body.reasoning``,
                      and models the gateway has not registered as reasoning-capable reject it unless
                      ``allowed_openai_params`` forces it through. Returns reasoning TOKEN COUNTS only;
                      the gateway strips the trace text (``merge_reasoning_content_in_choices: false``).

  anthropic           the native Messages API. Content comes back as typed blocks, so the thinking
                      blocks are the trace. NOTE: the API rejects temperature != 1 while extended
                      thinking is on, so a thinking run cannot reproduce the pool's T = 0.9 exactly.

Both return the same triple as ``call_llm_async``: ``(content, reasoning_trace, usage)`` with
``usage = {"in", "out"}`` so the existing cost ledger keeps working.
"""
import os

_ANTHROPIC_MAX_T = 1.0


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"FATAL: {name} is not set (needed by the configured provider)")
    return v


class OpenAICompatibleProvider:
    """A LiteLLM/OpenAI-compatible gateway."""

    def __init__(self, cfg: dict):
        from openai import AsyncOpenAI
        base = _env(cfg.get("base_url_env", "LITELLM_BASE_URL")).rstrip("/")
        self.client = AsyncOpenAI(base_url=base + "/v1", api_key=_env(cfg.get("api_key_env", "LITELLM_API_KEY")))
        self.effort = cfg.get("effort")
        # models the gateway has not registered as reasoning-capable reject reasoning_effort outright;
        # LiteLLM's allowed_openai_params forces it through (verified on gpt-6-astra-flex).
        self.force_params = list(cfg.get("allowed_openai_params") or [])

    @staticmethod
    def api_name(model_id: str) -> str:
        return model_id.split("/", 1)[-1]

    async def acall(self, *, messages, model, temperature, max_tokens):
        kw = dict(model=self.api_name(model), messages=messages, max_completion_tokens=max_tokens)
        if self.effort:
            kw["reasoning_effort"] = self.effort
        if self.force_params:
            kw["extra_body"] = {"allowed_openai_params": self.force_params}
        r = await self.client.chat.completions.create(**kw)
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        rt = (getattr(det, "reasoning_tokens", 0) if det else 0) or 0
        usage = {"in": getattr(u, "prompt_tokens", 0) or 0,
                 "out": getattr(u, "completion_tokens", 0) or 0, "reasoning": rt}
        if not r.choices:
            return None, None, usage
        m = r.choices[0].message
        trace = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        return m.content, trace, usage


class AnthropicProvider:
    """Anthropic's native Messages API.

    SDK 1.4 removed ``temperature`` from this endpoint altogether and added ``output_config.effort``
    (low|medium|high|xhigh|max). So a direct-API run CANNOT reproduce the pool's T = 0.9 -- there is no
    temperature to set -- and that deviation is recorded rather than papered over. Thinking blocks come
    back as typed content blocks, which is where the trace is taken from.
    """

    EFFORTS = ("low", "medium", "high", "xhigh", "max")

    def __init__(self, cfg: dict):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=_env(cfg.get("api_key_env", "ANTHROPIC_API_KEY")))
        self.effort = cfg.get("effort")
        if self.effort and self.effort not in self.EFFORTS:
            raise ValueError(f"FATAL: effort {self.effort!r} not in {self.EFFORTS}")
        self.thinking_budget = cfg.get("thinking_budget_tokens")
        # "enabled" (with a token budget) is the older interface; the 4.7/4.8/fable-5.1 generation
        # rejects it and requires "adaptive", where the model decides how much to think and
        # output_config.effort is the dial. Only some models return the thinking blocks at all.
        self.thinking_type = cfg.get("thinking_type")

    @staticmethod
    def api_name(model_id: str) -> str:
        # ledger key "anthropic/claude-opus-4.8" -> API name "claude-opus-4-8"
        return model_id.split("/", 1)[-1].replace(".", "-")

    async def acall(self, *, messages, model, temperature, max_tokens):
        kw = dict(model=self.api_name(model), max_tokens=max_tokens,
                  messages=[{"role": m["role"], "content": m["content"]} for m in messages])
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        if self.thinking_type == "adaptive":
            kw["thinking"] = {"type": "adaptive"}
        elif self.thinking_budget:
            if max_tokens <= self.thinking_budget:
                raise ValueError(f"FATAL: max_tokens ({max_tokens}) must exceed the thinking budget "
                                 f"({self.thinking_budget}) or the model has no room to answer")
            kw["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        # The SDK refuses a non-streaming request whose max_tokens could take >10 min, which any
        # thinking-sized budget can. Stream and reassemble; get_final_message() returns the same
        # Message object (typed content blocks + usage) the non-streaming call would have.
        async with self.client.messages.stream(**kw) as stream:
            r = await stream.get_final_message()
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        trace = "\n".join(getattr(b, "thinking", "") for b in r.content
                          if getattr(b, "type", "") == "thinking") or None
        usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
        return (text or None), trace, usage


def build(cfg: dict):
    kind = (cfg or {}).get("kind")
    if kind == "openai_compatible":
        return OpenAICompatibleProvider(cfg)
    if kind == "anthropic":
        return AnthropicProvider(cfg)
    raise ValueError(f"FATAL: unknown provider kind {kind!r} (expected openai_compatible|anthropic)")
