# Speech-to-Text Service

Microservice providing speech recognition using OpenAI's Whisper model.

## Features
- Support for 99 languages including Haitian Creole
- File upload and real-time streaming
- WebSocket support for live transcription
- Multiple model sizes available

## Quick Start

```bash
docker-compose up
```

## API Endpoints

- `POST /api/v1/transcribe` - Transcribe audio file
- `WS /api/v1/stream` - Real-time transcription via WebSocket

## License
MIT
