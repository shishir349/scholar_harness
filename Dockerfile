FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# the 3 small runtime files the writer/audit read (style targets, field list, human skeletons, LAMP span examples). NOT the corpus: it is downloaded at boot (corpus/boot.py)
COPY data/style_baseline.json data/style_profile.json data/fields.selected.json data/human_skeletons.json data/lamp_span_examples.json data/claude_sentence_detector.json ./data/
RUN pip install --no-cache-dir .
# run the code in /app/src, not the pip copy: config.ROOT must be /app so ROOT/data finds the files above
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 HARNESS_DB=/tmp/corpus.db
# Render injects PORT. Required env: ANTHROPIC_API_KEY, HARNESS_TOKEN.
# Optional: HARNESS_BUDGET_USD (default 2/request), OPENALEX_API_KEY (research), HARNESS_CORPUS_URL + HARNESS_CORPUS_SHA256 (exemplar corpus)
CMD ["python", "-m", "harness.server"]
