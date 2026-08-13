FROM python:3.11-slim

# Install system build dependencies for potential native libraries (e.g. numpy, networkx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and static assets
COPY app/ ./app/
COPY static/ ./static/

# Create folders for persistent database and GraphRAG indexes
RUN mkdir -p /app/graphrag_data

# Set environmental variables
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
