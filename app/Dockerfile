# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Create a script to run both services
RUN echo '#!/bin/bash\n\
uvicorn backend.main:app --host 0.0.0.0 --port 8000 & \
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"] 