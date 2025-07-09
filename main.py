from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Speech to Text Service", version="0.1.0")

class SpeechToTextResponse(BaseModel):
    text: str
    confidence: float

@app.get("/")
async def root():
    return {"message": "Speech to Text Service"}

@app.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(file: UploadFile = File(...)):
    # Placeholder implementation
    return SpeechToTextResponse(
        text=f"Transcribed text from {file.filename}",
        confidence=0.95
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}