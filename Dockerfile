FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p uploads outputs

EXPOSE 8080

CMD gunicorn app:app --workers 1 --timeout 900 --bind 0.0.0.0:${PORT:-8080}
