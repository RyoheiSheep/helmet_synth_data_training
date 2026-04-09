FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finetune.py .

ENTRYPOINT ["python3.11", "finetune.py"]
CMD ["--config", "/data/config/step_d.yaml", "--dataset-dir", "/data/dataset", "--output-dir", "/data/models"]
