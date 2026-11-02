FROM python:3.12-slim
WORKDIR /app
# runtime deps only: no corpus build tools, no corpus data (refine needs neither)
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
# Render injects PORT. Required env: ANTHROPIC_API_KEY, HARNESS_TOKEN. Optional: HARNESS_BUDGET_USD (default 2 per request)
CMD ["python", "-m", "harness.server"]
