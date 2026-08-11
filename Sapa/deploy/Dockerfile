# Dockerfile for Railway deployment of the from-scratch AI platform

FROM python:3.11-slim

WORKDIR /app

# System deps needed by Pillow/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway sets $PORT dynamically - respect it
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
