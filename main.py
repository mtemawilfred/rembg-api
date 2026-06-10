# ── main.py ───────────────────────────────────────────────────────────────
# rembg-api: background removal (+ Whisper transcribe, unused by RedPill but kept).
#
# CHANGE (RedPill): switched the segmentation model from the default 'u2net'
# to 'isnet-general-use'. u2net cannot see the edge between a same-colored
# subject and background, so it erases parts of the subject (e.g. light
# highlights on the Sovereign against a light bg). isnet-general-use segments
# subjects far more cleanly. The PRIMARY fix is still upstream (generate poses
# on a #7A7A7A grey bg), but the better model is a safety net.
#
# The /remove-bg endpoint now also exposes OPTIONAL alpha-matting tuning. The
# workflow calls /remove-bg with NO query params, so defaults apply and behave
# like a normal cutout with the better model. Pass alpha_matting=true only when
# edges need extra polish.
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from rembg import remove, new_session   # new_session lets us pick a better model
import whisper
import tempfile, os

app = FastAPI()
model = whisper.load_model("base")

# Pre-load the human/general segmentation session ONCE at startup so every
# request reuses it (avoids re-loading the model per call). The model weights
# are baked into the Docker image at build time (see Dockerfile), so this does
# not hit the network at runtime.
_rembg_session = new_session("isnet-general-use")

@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    # Optional edge-quality tuning. Defaults keep current behavior (just the
    # better model). alpha_matting smooths the cutout edge; thresholds control
    # how aggressive the matte is.
    alpha_matting: bool = Query(False),
    fg_threshold: int = Query(240),   # alpha_matting_foreground_threshold
    bg_threshold: int = Query(10),    # alpha_matting_background_threshold
    erode_size: int = Query(10),      # alpha_matting_erode_size
    post_process: bool = Query(True), # clean up the mask for smoother edges
):
    input_bytes = await file.read()
    output_bytes = remove(
        input_bytes,
        session=_rembg_session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=fg_threshold,
        alpha_matting_background_threshold=bg_threshold,
        alpha_matting_erode_size=erode_size,
        post_process_mask=post_process,
    )
    return Response(content=output_bytes, media_type="image/png")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path, word_timestamps=True)
    os.unlink(tmp_path)

    words = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            words.append({
                "word": word["word"].strip(),
                "start": word["start"],
                "end": word["end"]
            })

    return {"words": words}

@app.get("/health")
def health():
    return {"status": "ok"}