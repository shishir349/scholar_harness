FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# the 3 small runtime files the writer/audit read (style targets, field list, human skeletons, LAMP span examples). NOT the corpus: it is downloaded at boot (corpus/boot.py)
COPY data/style_baseline.json data/style_profile.json data/fields.selected.json data/human_skeletons.json data/lamp_span_examples.json data/surprisal_baseline.json ./data/
RUN pip install --no-cache-dir ".[surprisal]"
# local surprisal model (distilgpt2 int8, 83MB + tokenizer), fetched once at build and checked by hash; without it the step is skipped (HARNESS_SURPRISAL=off also skips)
RUN mkdir -p data/models && python -c "import urllib.request as u,hashlib;B='https://huggingface.co/Xenova/distilgpt2/resolve/main/';[(u.urlretrieve(B+a,'data/models/'+b),(hashlib.sha256(open('data/models/'+b,'rb').read()).hexdigest()==h or exit('bad hash '+b))) for a,b,h in (('onnx/decoder_model_quantized.onnx','lm-q.onnx','1d3ab4d7e08ca9b8ea9a42dda13a05b7ff0f1b0c9bdc4f6eba27e7f0ceff8431'),('tokenizer.json','tokenizer.json','cda20b8ca044949aa07ac4078420c80d1a57139d5f9f33700e46fb2d891e7c66'))]"
# run the code in /app/src, not the pip copy: config.ROOT must be /app so ROOT/data finds the files above
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 HARNESS_DB=/tmp/corpus.db
# Render injects PORT. Required env: ANTHROPIC_API_KEY, HARNESS_TOKEN.
# Optional: HARNESS_BUDGET_USD (default 2/request), OPENALEX_API_KEY (research), HARNESS_CORPUS_URL + HARNESS_CORPUS_SHA256 (exemplar corpus)
CMD ["python", "-m", "harness.server"]
