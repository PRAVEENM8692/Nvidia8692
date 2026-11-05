"""NVIDIA build.nvidia.com API client (OpenAI-compatible).

Setup:
    pip install openai
    export NVIDIA_API_KEY=nvapi-...      # GitHub Actions-ல secret-ஆ வைங்க

Usage:
    from nvidia_client import ask
    print(ask("TNPSC பற்றி ஒரு வரி சொல்லு"))
"""
import os
import time

from openai import OpenAI, RateLimitError, APIError

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"  # vendor/model பேர் கட்டாயம்

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY செட் பண்ணப்படவில்லை")
        _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client


def ask(prompt: str, system: str | None = None, model: str = DEFAULT_MODEL,
        temperature: float = 0.7, max_tokens: int = 1024, retries: int = 3) -> str:
    """ஒரு prompt அனுப்பி பதிலை text-ஆ திருப்பும். Rate limit (40/min) வந்தா retry பண்ணும்."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            time.sleep(5 * (attempt + 1))
        except APIError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("Rate limit காரணமா தோல்வி; சிறிது நேரம் கழித்து முயற்சிக்கவும்")


def list_models() -> list[str]:
    """உங்க key-க்கு கிடைக்கும் மாடல்கள்."""
    return [m.id for m in get_client().models.list().data]


if __name__ == "__main__":
    print(ask("வணக்கம்! ஒரு வரியில் உன்னை அறிமுகம் செய்."))
