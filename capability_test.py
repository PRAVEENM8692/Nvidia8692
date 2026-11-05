"""Kids-story capability check for NVIDIA build.nvidia.com free key.

Checks: (1) which models the key can list, (2) Tamil story text, (3) image generation.
TTS: only reports whether speech-like models appear in the model list (no audio call).
The API key is read from NVIDIA_API_KEY and never printed.
"""
import base64
import os
import sys

import requests

KEY = os.environ["NVIDIA_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
BASE = "https://integrate.api.nvidia.com/v1"
results = []


def report(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}", flush=True)


# 1. Model list
models = []
try:
    r = requests.get(f"{BASE}/models", headers=H, timeout=30)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    report("model list", True, f"({len(models)} models)")
except Exception as e:
    report("model list", False, str(e)[:200])

for label, words in [
    ("speech/TTS-like", ["tts", "magpie", "riva", "chatterbox", "fugatto", "parakeet"]),
    ("image-like", ["flux", "stable-diffusion", "sdxl", "sd3"]),
]:
    found = [m for m in models if any(w in m.lower() for w in words)]
    print(f"  {label} in /v1/models: {found if found else 'none listed'}")

# 2. Tamil story text
story_models = [m for m in ["meta/llama-3.3-70b-instruct", "qwen/qwen2.5-72b-instruct",
                            "mistralai/mixtral-8x22b-instruct-v0.1"] if m in models] or \
               ["meta/llama-3.3-70b-instruct"]
for m in story_models[:3]:
    try:
        r = requests.post(
            f"{BASE}/chat/completions", headers=H, timeout=60,
            json={"model": m, "max_tokens": 200,
                  "messages": [{"role": "user",
                                "content": "குழந்தைகளுக்கான 4 வரி சிறிய தமிழ் கதை எழுது."}]})
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"]
            report(f"story text: {m}", True)
            print("   ", txt.replace("\n", " ")[:300])
        else:
            report(f"story text: {m}", False, f"HTTP {r.status_code} {r.text[:120]}")
    except Exception as e:
        report(f"story text: {m}", False, str(e)[:150])

# 3. Image generation
image_endpoints = [
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
    "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium",
]
for url in image_endpoints:
    name = url.rsplit("/", 1)[-1]
    try:
        r = requests.post(url, headers=H, timeout=120,
                          json={"prompt": "a cute cartoon elephant reading a book, kids story illustration",
                                "width": 768, "height": 768, "seed": 0, "steps": 4})
        if r.status_code == 200:
            data = r.json()
            b64 = data.get("image") or (data.get("artifacts") or [{}])[0].get("base64")
            if b64:
                with open(f"test_{name}.jpg", "wb") as f:
                    f.write(base64.b64decode(b64))
            report(f"image: {name}", True, "(image saved)" if b64 else "(200, no image field)")
        else:
            report(f"image: {name}", False, f"HTTP {r.status_code} {r.text[:120]}")
    except Exception as e:
        report(f"image: {name}", False, str(e)[:150])

print("\n=== SUMMARY ===")
for name, ok, detail in results:
    print(("OK   " if ok else "FAIL ") + name, detail)
sys.exit(0)
