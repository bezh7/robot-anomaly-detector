FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app

COPY requirements-runtime.txt requirements-runtime.txt

RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
 && pip install -r requirements-runtime.txt

COPY src src
COPY scripts scripts
COPY README.md README.md

CMD ["python", "-m", "src.modeling.run_model_search", "--artifact-root", "artifacts/features", "--output-root", "artifacts/modeling"]
