"""NVIDIA API RPM test: ஒரு நிமிஷத்துக்கு எத்தனை requests போகுதுன்னு பார்க்கும்.

Run:  NVIDIA_API_KEY=nvapi-... python rpm_test.py [model] [seconds]
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta/llama-3.3-70b-instruct"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60
KEY = os.environ["NVIDIA_API_KEY"]

ok = 0
limited = 0
errors = 0
first_429_at = None
start = time.time()


def call(_):
    global ok, limited, errors, first_429_at
    try:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=30,
        )
        if r.status_code == 200:
            ok += 1
        elif r.status_code == 429:
            limited += 1
            if first_429_at is None:
                first_429_at = ok
        else:
            errors += 1
            print("status", r.status_code, r.text[:120])
    except Exception as e:
        errors += 1
        print("error", e)


with ThreadPoolExecutor(max_workers=8) as pool:
    while time.time() - start < DURATION:
        list(pool.map(call, range(8)))

elapsed = time.time() - start
print(f"model={MODEL}  time={elapsed:.0f}s")
print(f"successful={ok}  rate-limited(429)={limited}  other errors={errors}")
print(f"requests/min (successful) ≈ {ok / elapsed * 60:.1f}")
if first_429_at is not None:
    print(f"முதல் 429 வந்தது {first_429_at} வெற்றிகரமான requests-க்கு பிறகு")
