from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import logging
from datetime import datetime
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Creole Speech-to-Text Service",
    description="Speech-to-Text API for Haitian Creole and other languages",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TranscriptionRequest(BaseModel):
    language: str = "auto"
    model: str = "whisper-base"
    
class StreamTranscriptionRequest(BaseModel):
    language: str = "auto"
    model: str = "whisper-base"
    chunk_size: int = 1024

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration: float
    
class PartialTranscriptionResponse(BaseModel):
    text: str
    is_final: bool
    confidence: float
    
class LanguageDetectionResponse(BaseModel):
    detected_language: str
    confidence: float
    
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str

class WebSocketMessage(BaseModel):
    type: str  # "audio_chunk", "config", "stop"
    data: Optional[str] = None  # base64 encoded audio or JSON config
    
class WebSocketResponse(BaseModel):
    type: str  # "partial_transcript", "final_transcript", "error"
    data: dict

# Mock transcription function (replace with actual ML model)
async def transcribe_audio(audio_data: bytes, language: str = "auto") -> TranscriptionResponse:
    """Mock transcription function - replace with actual model inference"""
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # Mock transcriptions for demo
    mock_transcriptions = {
        "ht": [
            "Bonjou, koman ou ye?",
            "Mwen bezwen èd tanpri.",
            "Kote lopital la ye?",
            "Mèsi anpil pou èd la.",
            "Nou bezwen dlo ak manje."
        ],
        "en": [
            "Hello, how are you?",
            "I need help please.",
            "Where is the hospital?",
            "Thank you very much for the help.",
            "We need water and food."
        ]
    }
    
    # Simple mock based on audio length
    audio_duration = len(audio_data) / 16000  # Assume 16kHz audio
    
    # Auto-detect language (mock)
    if language == "auto":
        language = "ht" if len(audio_data) % 2 == 0 else "en"
    
    # Get mock transcription
    transcriptions = mock_transcriptions.get(language, mock_transcriptions["en"])
    text = transcriptions[len(audio_data) % len(transcriptions)]
    
    return TranscriptionResponse(
        text=text,
        language=language,
        confidence=0.89,
        duration=audio_duration
    )

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_sessions: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_sessions[websocket] = {
            "language": "auto",
            "model": "whisper-base",
            "buffer": b""
        }
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_sessions:
            del self.connection_sessions[websocket]
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")

    async def send_message(self, websocket: WebSocket, message: WebSocketResponse):
        try:
            await websocket.send_text(message.json())
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")

manager = ConnectionManager()

# Routes
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Creole Speech-to-Text Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="speech-to-text",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = "auto",
    model: str = "whisper-base"
):
    """Transcribe audio file to text"""
    try:
        logger.info(f"Transcribing file: {file.filename} (language: {language})")
        
        # Validate file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        result = await transcribe_audio(audio_data, language)
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/api/v1/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(file: UploadFile = File(...)):
    """Detect language of audio file"""
    try:
        logger.info(f"Detecting language for: {file.filename}")
        
        # Validate file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await file.read()
        
        # Mock language detection
        await asyncio.sleep(0.2)
        detected = "ht" if len(audio_data) % 2 == 0 else "en"
        
        return LanguageDetectionResponse(
            detected_language=detected,
            confidence=0.92
        )
        
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.websocket("/api/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time speech-to-text streaming"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_message(websocket, WebSocketResponse(
            type="connected",
            data={"message": "WebSocket connection established"}
        ))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = WebSocketMessage.parse_raw(data)
                session = manager.connection_sessions[websocket]
                
                if message.type == "config":
                    # Update session configuration
                    config = json.loads(message.data)
                    session.update(config)
                    
                    await manager.send_message(websocket, WebSocketResponse(
                        type="config_updated",
                        data={"config": session}
                    ))
                    
                elif message.type == "audio_chunk":
                    # Process audio chunk
                    audio_data = base64.b64decode(message.data)
                    session["buffer"] += audio_data
                    
                    # Send partial transcription (mock)
                    partial_text = "Listening..." if len(session["buffer"]) < 8000 else "Processing..."
                    
                    await manager.send_message(websocket, WebSocketResponse(
                        type="partial_transcript",
                        data={
                            "text": partial_text,
                            "is_final": False,
                            "confidence": 0.5
                        }
                    ))
                    
                    # If buffer is large enough, process it
                    if len(session["buffer"]) >= 16000:  # ~1 second at 16kHz
                        result = await transcribe_audio(session["buffer"], session["language"])
                        
                        await manager.send_message(websocket, WebSocketResponse(
                            type="final_transcript",
                            data={
                                "text": result.text,
                                "is_final": True,
                                "confidence": result.confidence,
                                "language": result.language
                            }
                        ))
                        
                        # Clear buffer
                        session["buffer"] = b""
                        
                elif message.type == "stop":
                    # Process final buffer if any
                    if session["buffer"]:
                        result = await transcribe_audio(session["buffer"], session["language"])
                        
                        await manager.send_message(websocket, WebSocketResponse(
                            type="final_transcript",
                            data={
                                "text": result.text,
                                "is_final": True,
                                "confidence": result.confidence,
                                "language": result.language
                            }
                        ))
                    
                    break
                    
            except json.JSONDecodeError:
                await manager.send_message(websocket, WebSocketResponse(
                    type="error",
                    data={"message": "Invalid JSON message"}
                ))
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await manager.send_message(websocket, WebSocketResponse(
                    type="error",
                    data={"message": str(e)}
                ))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

@app.get("/api/v1/models")
async def get_models():
    """Get available speech-to-text models"""
    return {
        "available_models": [
            {
                "id": "whisper-base",
                "name": "Whisper Base",
                "description": "Fast and accurate for most use cases",
                "languages": ["ht", "en", "fr", "es"]
            },
            {
                "id": "whisper-large",
                "name": "Whisper Large",
                "description": "Highest accuracy but slower",
                "languages": ["ht", "en", "fr", "es"]
            }
        ]
    }

@app.get("/api/v1/languages")
async def get_languages():
    """Get supported languages"""
    return {
        "supported_languages": [
            {"code": "ht", "name": "Haitian Creole", "native_name": "Kreyòl Ayisyen"},
            {"code": "en", "name": "English", "native_name": "English"},
            {"code": "fr", "name": "French", "native_name": "Français"},
            {"code": "es", "name": "Spanish", "native_name": "Español"},
            {"code": "auto", "name": "Auto-detect", "native_name": "Auto-detect"}
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)