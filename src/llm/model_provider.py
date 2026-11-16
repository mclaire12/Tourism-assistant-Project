"""Configurable language-model layer.

The project model remains Qwen/Qwen3-4B-Instruct-2507. This module does not
substitute another model. When no endpoint is configured, callers receive
None and the assistant quotes retrieved records instead of inventing a reply.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger("rwanda_tourism.llm")


class ModelError(Exception):
    """Safe to show to a visitor. Technical detail stays in the log."""


class ModelProvider:
    label = "language model"

    def generate(self, prompt: str, history: list[dict] | None = None) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(ModelProvider):
    """Chat-completions client for a Hugging Face router or any compatible host."""

    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: float):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.label = model_name

    def generate(self, prompt: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": "You are a Rwanda Tourism Assistant."}]
        for turn in history or []:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:1200]})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 400,
        }
        if "qwen" in self.model_name.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return _post_chat(self.api_url, self.api_key, payload, self.timeout)
        except ModelError:
            if "chat_template_kwargs" not in payload:
                raise
            payload.pop("chat_template_kwargs", None)
            return _post_chat(self.api_url, self.api_key, payload, self.timeout)


class LocalQwenProvider(ModelProvider):
    """Local Qwen3. Loads weights on the first question, not at startup."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.label = model_name
        self._tokenizer = None
        self._model = None

    def generate(self, prompt: str, history: list[dict] | None = None) -> str:
        tokenizer, model = self._load()
        messages = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(templated, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return answer.strip()

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ModelError(
                "Local Qwen3 dependencies are not installed. Use a hosted endpoint or the retrieval-only mode."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant,
                device_map="auto",
            )
        else:
            logger.warning("Loading Qwen3 on CPU. This needs several gigabytes of memory.")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model


def build_model_provider(settings) -> tuple[ModelProvider | None, str]:
    provider = (settings.model_provider or "").lower()
    if provider in {"", "none", "retrieval", "retrieval_only"}:
        return None, (
            "Qwen3 is not connected. Answers quote the retrieved Rwanda records. "
            "Set MODEL_PROVIDER, MODEL_API_URL, MODEL_API_KEY, and MODEL_NAME to enable the language model."
        )
    if provider == "local":
        return LocalQwenProvider(settings.model_name), ""
    if provider in {"huggingface", "openai", "openai_compatible", "groq"}:
        if not settings.model_api_key or not settings.model_api_url:
            return None, (
                "The model provider is missing MODEL_API_URL or MODEL_API_KEY. "
                "Answers quote the retrieved Rwanda records until both are set."
            )
        return (
            OpenAICompatibleProvider(
                settings.model_api_url,
                settings.model_api_key,
                settings.model_name,
                settings.request_timeout,
            ),
            "",
        )
    return None, (
        f"MODEL_PROVIDER '{settings.model_provider}' is not recognized. "
        "Use huggingface, openai_compatible, local, or leave it empty."
    )


def parse_chat_response(body: dict) -> str:
    if not isinstance(body, dict):
        raise ModelError("The language model returned an unexpected response.")
    if body.get("error"):
        raise ModelError("The language model rejected the request.")
    choices = body.get("choices") or []
    if not choices:
        raise ModelError("The language model returned no answer.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text
    raise ModelError("The language model returned an empty answer.")


def _post_chat(api_url: str, api_key: str, payload: dict, timeout: float) -> str:
    url = api_url if api_url.endswith("/chat/completions") else f"{api_url}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        logger.error("Model HTTP %s: %s", exc.code, _redact(detail, api_key))
        raise ModelError("The language model could not complete the request. Please try again.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error("Model network error: %s", exc.__class__.__name__)
        raise ModelError("The language model could not be reached. Check the connection and try again.") from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Model response was not JSON")
        raise ModelError("The language model returned a response that could not be read.") from exc
    return parse_chat_response(body)


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text
