FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Bake the segmentation model into the image (RedPill change) ─────────────
# main.py calls new_session("isnet-general-use") at startup. rembg downloads
# that model on first use; baking it at BUILD time means the running container
# never needs network egress to fetch it (no cold-start download / failure).
# Requires network during the Docker build only (available on Railway builds).
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

COPY . .

# NOTE: CMD hardcodes port 8080 (unchanged). The Procfile uses $PORT. If Railway
# builds from this Dockerfile and injects a different PORT, the healthcheck can
# miss the container. Left as-is here per the no-unrelated-refactor rule — see
# the master doc "rembg PORT mismatch" flag to decide whether to switch CMD to
# a shell-form `--port ${PORT:-8080}`.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]