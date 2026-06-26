FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/* [cite: 1]

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt [cite: 1, 2]

# ── Bake the segmentation model into the image ─────────────────────────────
# Pre-loading weights at build time eliminates cold-start runtime network downloads[cite: 3].
RUN python -c "from rembg import new_session; new_session('isnet-general-use')" [cite: 5]

COPY . .

# FIXED: Removed hardcoded 8080 port. Uses shell-form fallback to handle Railway's dynamic $PORT environment injection.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]