FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY config.yaml .
COPY app/backend/ app/backend/
COPY app/frontend/ app/frontend/
COPY app/logger.py app/logger.py
COPY src/ src/
COPY artefacts/*.pth artefacts/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

WORKDIR /app/frontend

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]