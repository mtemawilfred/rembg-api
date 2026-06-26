# ── main.py ───────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from rembg import remove, new_session   
from PIL import Image                    
import io                                
import whisper
import tempfile
import os
import gc  # Added for explicit garbage collection memory clearing

app = FastAPI()

# Pre-load ONLY the segmentation session at startup.
# The weights are baked into the image, so this is fast and handles background removal instantly.
_rembg_session = new_session("isnet-general-use")

# CRITICAL COST FIX: Removed global 'model = whisper.load_model("base")' from here.
# This prevents 1GB+ of RAM from being permanently held host when the server is idle.

@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    alpha_matting: bool = Query(False),
    fg_threshold: int = Query(240),   
    bg_threshold: int = Query(10),    
    erode_size: int = Query(10),      
    post_process: bool = Query(True), 
    autocrop: bool = Query(True),
    crop_pad: int = Query(0),         
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

    if autocrop:
        img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        bbox = img.split()[-1].getbbox()   
        if bbox:
            img = img.crop(bbox)
            if crop_pad > 0:
                padded = Image.new(
                    "RGBA",
                    (img.width + 2 * crop_pad, img.height + 2 * crop_pad),
                    (0, 0, 0, 0),
                )
                padded.paste(img, (crop_pad, crop_pad))
                img = padded
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            output_bytes = buf.getvalue()

    return Response(content=output_bytes, media_type="image/png")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Lazy-load the model inside the request function.
    # It will only allocate RAM when this specific endpoint is hit.
    local_model = whisper.load_model("base")

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = local_model.transcribe(tmp_path, word_timestamps=True)
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

    finally:
        # COST FIX: Force the system to clear Whisper out of RAM immediately after the request finishes
        del local_model
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}