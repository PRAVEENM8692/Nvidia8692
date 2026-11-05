"""Kids-story capability check (v2) for an NVIDIA build.nvidia.com free key.

1. Prints the full model list the key can see
2. Tests Tamil story text on several currently-listed chat models
3. Tests image generation (flux.1-dev with valid steps)
TTS: only reports speech-like models in the list. The API key is never printed.
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


models = []
try:
    r = requests.get(f"{BASE}/models", headers=H, timeout=30)
    r.raise_for_status()
    models = sorted(m["id"] for m in r.json().get("data", []))
    report("model list", True, f"({len(models)} models)")
except Exception as e:
    report("model list", False, str(e)[:200])

print("\n--- ALL MODELS ---")
for m in models:
    print("  ", m)
print("--- END MODELS ---\n")

# candidate chat models: from the real list, skipping non-chat types
skip = ("embed", "rerank", "guard", "safety", "retriev", "vision", "vl", "translate",
        "reward", "parse", "clip", "nemoretriever", "bge", "arctic-embed", "paligemma")
prefs = ("llama-3.3", "llama-4", "qwen", "mistral", "gemma", "nemotron", "deepseek", "gpt-oss", "kimi")
chat = [m for m in models if not any(s in m.lower() for s in skip)]
ordered = [m for p in prefs for m in chat if p in m.lower()]
seen, candidates = set(), []
for m in ordered:
    if m not in seen:
        seen.add(m)
        candidates.append(m)
candidates = candidates[:8]
print("Testing chat models:", candidates, "\n")

for m in candidates:
    try:
        r = requests.post(
            f"{BASE}/chat/completions", headers=H, timeout=60,
            json={"model": m, "max_tokens": 200,
                  "messages": [{"role": "user",
                                "content": "குழந்தைகளுக்கான 4 வரி சிறிய தமிழ் கதை எழுது."}]})
        if r.status_code == 200:
            txt = (r.json()["choices"][0]["message"].get("content") or "")
            report(f"story text: {m}", True)
            print("     ", txt.replace("\n", " ")[:300], flush=True)
        else:
            report(f"story text: {m}", False, f"HTTP {r.status_code} {r.text[:100]}")
    except Exception as e:
        report(f"story text: {m}", False, str(e)[:120])

# image generation
for url, extra in [
    ("https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
     {"cfg_scale": 3.5, "steps": 28}),
]:
    name = url.rsplit("/", 1)[-1]
    try:
        r = requests.post(url, headers=H, timeout=240,
                          json={"prompt": "a cute cartoon elephant reading a book, kids story illustration",
                                "width": 768, "height": 768, "seed": 0, **extra})
        if r.status_code == 200:
            data = r.json()
            b64 = data.get("image") or (data.get("artifacts") or [{}])[0].get("base64")
            if b64:
                with open(f"test_{name}.jpg", "wb") as f:
                    f.write(base64.b64decode(b64))
            report(f"image: {name}", bool(b64), "(image saved)" if b64 else "(200 but no image field)")
        else:
            report(f"image: {name}", False, f"HTTP {r.status_code} {r.text[:150]}")
    except Exception as e:
        report(f"image: {name}", False, str(e)[:150])

speech = [m for m in models if any(w in m.lower() for w in
          ("tts", "magpie", "riva-tts", "chatterbox", "fugatto", "parakeet", "speech"))]
print("\nspeech-like models in list:", speech or "none")

print("\n=== SUMMARY ===")
for name, ok, detail in results:
    print(("OK   " if ok else "FAIL ") + name, detail)
sys.exit(0)
