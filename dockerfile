FROM --platform=linux/amd64 python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIPER_PORT=5001
ENV MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY app.py .

RUN mkdir -p /app/piper_output/cache && chmod -R 777 /app/piper_output
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app

USER appuser
EXPOSE 5001
CMD ["python", "app.py"]
