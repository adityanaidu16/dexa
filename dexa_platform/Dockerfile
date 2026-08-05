# Dexa gateway — small, CPU-only. Runs in front of your backend (hosted or BYOC).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway ./dexa_platform/gateway
COPY dashboard ./dexa_platform/dashboard
COPY __init__.py ./dexa_platform/__init__.py

EXPOSE 8080
CMD ["uvicorn", "dexa_platform.gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
