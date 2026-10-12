FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_AUTOINSTALL=false YOLO_OFFLINE=true OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --no-compile -r requirements-serving.txt && pip check
COPY src/serving/api.py src/serving/api.py
COPY src/data/preprocess.py src/data/validate.py src/data/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["python", "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
