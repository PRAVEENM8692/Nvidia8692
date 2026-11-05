# Nvidia8692

NVIDIA build.nvidia.com API (OpenAI-compatible) helpers.

- `nvidia_client.py` – `ask(prompt)` helper with retry on rate limits
- `rpm_test.py` – measures requests/minute for a model
- `.github/workflows/rpm-test.yml` – manual workflow to run the RPM test

Requires the `NVIDIA_API_KEY` repository secret (Settings → Secrets and variables → Actions).
