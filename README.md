# EmbeddingGemma 2 — Local API Server

A self-hosted embedding API server powered by Google's **EmbeddingGemma 2** model, running on Apple Silicon (M4 Pro) with auto-start on login.

## Features
- OpenAI-compatible `/v1/embeddings` endpoint
- API key authentication
- Auto-starts on macOS login via LaunchAgent
- Runs fully locally — no cloud costs

## Setup

### Requirements
- macOS (Apple Silicon recommended)
- Python 3.11+
- HuggingFace account + token

### Install dependencies
```bash
pip3.11 install fastapi uvicorn sentence-transformers torch torchvision transformers
```

### Run manually
```bash
./start.sh
```

### Install as auto-start service
```bash
cp com.oluwatosin.embedding-server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.oluwatosin.embedding-server.plist
```

## API Usage

### Health check
```bash
curl http://localhost:8765/health
```

### Generate embeddings
```bash
curl -X POST http://localhost:8765/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"input": "Hello world"}'
```

### Python client example
```python
import requests

response = requests.post(
    "http://YOUR_SERVER_IP:8765/v1/embeddings",
    headers={"X-API-Key": "YOUR_API_KEY"},
    json={"input": "Hello world"}
)
embedding = response.json()["data"][0]["embedding"]
print(f"Embedding dimensions: {len(embedding)}")
```

## Model
- **Model:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
- **Parameters:** 740M (text-only: 270M active, ~191MB RAM)
- **License:** Apache 2.0
