FROM python:3.12-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_DATA_DIR=/app/data \
    PORT=8000

WORKDIR /app

# System deps (needed for numpy/faiss)
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download nltk data
RUN python -m nltk.downloader punkt stopwords

# Copy project
COPY .. .

EXPOSE 8000
CMD ["python", "app.py"]