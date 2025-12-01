FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU only
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Sentence-Transformers (install dependencies normally)
RUN pip install --no-cache-dir sentence-transformers

# Copy requirements.txt (without sentence-transformers)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Pre-download SBERT model
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-mpnet-base-v2")
EOF

# Copy app code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
