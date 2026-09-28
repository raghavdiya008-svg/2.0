# ==============================================================================
# 2.0 Autonomous Video Repurposing Studio — Production GPU Container
# ==============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies: Python 3.10, FFmpeg, libfribidi for ASS subtitles, zstd for Ollama
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libfribidi-dev \
    libass-dev \
    zstd \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy<2.0.0" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Copy project source
COPY . .

# Create volume directories
RUN mkdir -p inputs outputs assets/logo assets/emojis temp

EXPOSE 8000

# Start Ollama daemon and Studio Web Server
CMD ["bash", "-c", "ollama serve & sleep 3 && ollama pull llama3.2:3b & python -m src.server --port 8000"]
