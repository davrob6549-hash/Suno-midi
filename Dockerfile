# Use a PyTorch base image so torch/demucs don't need to compile from scratch
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages in stages so Docker can cache them
# Heavy ML packages first (cached separately)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    demucs==4.0.1 \
    basic-pitch==0.3.3 \
    librosa==0.10.2 \
    pretty_midi==0.2.10 \
    soundfile==0.12.1

# Lightweight packages
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    gunicorn==22.0.0 \
    numpy==1.26.4 \
    scipy==1.13.1

# Copy app files
COPY app.py processor.py ./
COPY templates/ templates/

# Create required dirs
RUN mkdir -p uploads outputs

EXPOSE 8080

CMD gunicorn app:app --workers 1 --timeout 900 --bind 0.0.0.0:${PORT:-8080}
