# Sanwaad, in one image.
#
# Two things make a first boot slow: downloading the 470MB ONNX embedding model
# and building the policy index over it. Both are deterministic outputs of
# files that are already in this repo, so both happen at build time and the
# container starts in seconds. That is also why the layer order below is what
# it is — requirements, then the model, then the source. Editing a Python file
# rebuilds the last layer only; the model download is cached until the pinned
# fastembed version or the model name changes.
#
#   docker build -t sanwaad .
#   docker run --rm -p 7870:7870 sanwaad              # offline stubs, no key
#   docker run --rm -p 7870:7870 --env-file .env sanwaad
#
# The image ships no key and needs none: without GOOGLE_API_KEY the model layer
# degrades to offline stubs by design, so the console and the evals still run.

FROM python:3.12-slim AS base

# onnxruntime, which fastembed runs the embedding model on, links against
# libgomp. It is the only system package this image needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    SANWAAD_PORT=7870

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Warm the embedding model into the image. Naming the model here rather than
# importing sanwaad.config keeps this layer independent of the source: it is
# the same default, and a mismatch shows up immediately as a download at boot.
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBED_MODEL}')"

COPY . .

# Build the policy index over the baked model, so /ready is true almost at once
# instead of a minute after boot.
RUN python -c "from sanwaad.rag.store import get_store; print(len(get_store().clauses), 'clauses indexed')"

# Runtime state — the checkpoint database, traces, the audit log — is written
# under sanwaad/data. It must be writable by a non-root user, and it is worth a
# volume if you want cases to survive `docker run --rm`.
RUN useradd --create-home --uid 10001 sanwaad \
    && chown -R sanwaad:sanwaad /app/sanwaad/data
USER sanwaad

EXPOSE 7870

# Liveness only. Readiness is /ready, which an orchestrator should poll
# separately — Docker has one probe, so it gets the cheap one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7870/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "sanwaad.api.server:app", "--host", "0.0.0.0", "--port", "7870"]
