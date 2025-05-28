# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port 8000 for FastAPI BE and 8501 for Streamlit FE
EXPOSE 8000 8501

RUN echo '#!/bin/bash \
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
    & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 \
    && fg' > /app/start.sh && chmod +x /app/start.sh

CMD ["bash", "-c", "exec /app/start.sh"]