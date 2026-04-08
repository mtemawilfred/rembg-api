from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rembg import remove
import whisper
import tempfile, os

app = FastAPI()
model = whisper.load_model("base")

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    input_bytes = await file.read()
    output_bytes = remove(input_bytes)
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
