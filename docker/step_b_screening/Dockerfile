FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY screen_and_label.py .
COPY vllm_teacher.py .

ENTRYPOINT ["python3.11", "screen_and_label.py"]
CMD ["--config", "/data/config/step_b.yaml", "--provider", "vllm", "--generated-dir", "/data/generated", "--output-dir", "/data/screened"]
