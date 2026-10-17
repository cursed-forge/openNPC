"""Remote LLM backend for OpenNPC (CursedForge fork).

Calls an OpenAI-compatible chat completions endpoint (for example an
Open-WebUI or llama.cpp server) instead of loading a model into this
process. It has the same ``generate`` / ``status`` / ``is_loaded`` /
``load`` surface as :class:`opennpc.llm.LLMEngine`, so
:class:`opennpc.dialogue.NPCDialogue` and other callers can use either.

Design rules of the fork:

- Standard library only. No model download, no remote code.
- The model id is pinned. ``load`` refuses to start when the server does
  not list that exact id, and ``generate`` never sends another id.
- Structured output: ``generate`` passes ``response_format`` through, so a
  caller can ask for a strict JSON schema.
- Extra body fields (for example ``chat_template_kwargs``) pass through
  unchanged, so a server specific switch needs no code change here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from opennpc.llm import LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_CHAT_PATH = "/api/chat/completions"
DEFAULT_MODELS_PATH = "/api/models"
DEFAULT_USER_AGENT = "OpenNPC-RemoteLLM/0.1"


class RemoteLLMError(RuntimeError):
    """A remote call failed. ``code`` is a short stable reason."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


Opener = Callable[[urllib.request.Request, float], Any]


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


class RemoteLLMEngine:
    """Engine that calls an OpenAI-compatible chat endpoint.

    Parameters
    ----------
    base_url : str
        Server root, for example ``https://ai.example.com``.
    model : str
        The pinned model id. Required.
    api_key : str
        Bearer token. Never logged.
    chat_path, models_path : str
        Endpoint paths. The defaults match Open-WebUI.
    timeout_s : float
        Timeout of one HTTP call.
    max_new_tokens : int
        ``max_tokens`` of each call.
    temperature : float
        Default sampling temperature.
    extra_body : mapping
        Fields merged into every chat request body (for example
        ``{"chat_template_kwargs": {"enable_thinking": False}}``).
    user_agent : str
        Sent as ``User-Agent``. Some proxies block the urllib default.
    opener : callable
        Test seam: ``opener(request, timeout)`` returns a response object.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        chat_path: str = DEFAULT_CHAT_PATH,
        models_path: str = DEFAULT_MODELS_PATH,
        timeout_s: float = 30.0,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        extra_body: Mapping[str, Any] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        opener: Opener | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required: a remote engine never picks a model by itself")
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self._api_key = api_key
        self._chat_path = chat_path
        self._models_path = models_path
        self.timeout_s = float(timeout_s)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._extra_body = dict(extra_body or {})
        self._user_agent = user_agent
        self._opener: Opener = opener or _default_opener
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._calls = 0
        self._failures = 0

    # ------------------------------------------------------------------ surface

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "remote"

    def load(self) -> None:
        """Checks that the server lists the pinned model. Raises otherwise."""
        if self._loaded:
            return
        try:
            body = self._request("GET", self._models_path, None)
        except RemoteLLMError as error:
            self._load_error = error.code
            raise
        ids = _model_ids(body)
        if self.model_name not in ids:
            self._load_error = "model_missing"
            raise RemoteLLMError("model_missing", f"the server does not list {self.model_name}")
        self._loaded = True
        self._load_error = None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool = True,
        response_format: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """One chat call. ``use_cache`` is accepted for interface parity; the
        server owns any prompt cache."""
        del use_cache
        if not self._loaded:
            self.load()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = dict(self._extra_body)
        payload["model"] = self.model_name
        payload["messages"] = messages
        payload["max_tokens"] = int(max_new_tokens or self.max_new_tokens)
        payload["temperature"] = float(self.temperature if temperature is None else temperature)
        payload["stream"] = False
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        start = time.monotonic()
        with self._lock:
            self._calls += 1
            try:
                body = self._request("POST", self._chat_path, payload)
            except RemoteLLMError:
                self._failures += 1
                raise
        elapsed = (time.monotonic() - start) * 1000.0

        text, tokens, model = _first_choice(body)
        if model and model != self.model_name:
            self._failures += 1
            raise RemoteLLMError("model_mismatch", f"the server answered with {model}")
        return LLMResponse(
            text=text,
            model=self.model_name,
            latency_ms=round(elapsed, 1),
            cached=False,
            tokens_generated=tokens,
        )

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "model": self.model_name,
            "device": self.device,
            "base_url": self.base_url,
            "calls": self._calls,
            "failures": self._failures,
            "load_error": self._load_error,
        }

    def clear_cache(self) -> None:
        """Interface parity with LLMEngine. The remote server owns its cache."""

    # ------------------------------------------------------------------ http

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, method=method)
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", self._user_agent)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if self._api_key:
            request.add_header("Authorization", "Bearer " + self._api_key)
        try:
            response = self._opener(request, self.timeout_s)
            try:
                raw = response.read()
                status = getattr(response, "status", 200)
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
        except urllib.error.HTTPError as error:
            raise RemoteLLMError("http_" + str(error.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            code = "timeout" if "timed out" in str(reason) else "unreachable"
            raise RemoteLLMError(code) from None
        if status >= 400:
            raise RemoteLLMError("http_" + str(status))
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RemoteLLMError("bad_json") from None


def _model_ids(body: Any) -> set[str]:
    ids: set[str] = set()
    rows = body.get("data") if isinstance(body, dict) else body
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                ids.add(row["id"])
    return ids


def _first_choice(body: Any) -> tuple[str, int, str]:
    if not isinstance(body, dict):
        raise RemoteLLMError("bad_answer", "the answer is not an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RemoteLLMError("bad_answer", "the answer has no choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RemoteLLMError("bad_answer", "the first choice has no text")
    usage = body.get("usage")
    tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
    model = body.get("model") if isinstance(body.get("model"), str) else ""
    return content.strip(), int(tokens or 0), model
