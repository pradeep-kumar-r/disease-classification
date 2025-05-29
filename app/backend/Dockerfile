FROM python:3.9-slim

COPY requirements.txt .
COPY config.yaml .
COPY app/backend/ app/backend/
COPY app/logger.py app/logger.py
COPY src/ src/
COPY artefacts/*.pth artefacts/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]