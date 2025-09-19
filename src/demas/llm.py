from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import requests

DEFAULT_CHUTES_BASE_URL = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    """Raised when an LLM provider returns an error response."""


class BaseLLMClient:
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        timeout: float,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError


@dataclass
class ChutesClient(BaseLLMClient):
    api_key: str
    base_url: str = DEFAULT_CHUTES_BASE_URL

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        timeout: float,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            raise LLMError(f"Chutes API error {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Chutes response: {json.dumps(data)[:200]}") from exc


@dataclass
class OpenRouterClient(BaseLLMClient):
    api_key: str
    base_url: str = DEFAULT_OPENROUTER_BASE_URL

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        timeout: float,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "demas-i0",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            raise LLMError(f"OpenRouter API error {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenRouter response: {json.dumps(data)[:200]}") from exc


def select_llm_client(provider: str, creds: Dict[str, str]) -> BaseLLMClient:
    provider_normalized = provider.lower()
    if provider_normalized == "chutes":
        key = creds.get("CHUTES_API_KEY") or creds.get("CHUTES_API_KEY_ALT")
        if not key:
            raise RuntimeError("missing CHUTES_API_KEY in secrets/credentials.txt")
        return ChutesClient(api_key=key)
    if provider_normalized == "openrouter":
        key = creds.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("missing OPENROUTER_API_KEY in secrets/credentials.txt")
        return OpenRouterClient(api_key=key)
    raise RuntimeError(f"unsupported provider '{provider}'")


__all__ = [
    "BaseLLMClient",
    "ChutesClient",
    "LLMError",
    "OpenRouterClient",
    "select_llm_client",
]
