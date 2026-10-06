"""
EmbeddingGemma 2 — Local API Server
Serves text embeddings over HTTP with API key authentication.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("embedding-server")

API_KEY = os.environ.get("EMBEDDING_API_KEY", "zuFw3OLTYYkpsLkH8RfsGhZzHeDpGW9rK5HxV_ybz3A")
MODEL_ID = os.environ.get("EMBEDDING_MODEL", "google/embeddinggemma-2")

model: SentenceTransformer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    log.info(f"Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    log.info("Model ready.")
    yield
    model = None

app = FastAPI(
    title="EmbeddingGemma 2 API",
    description="Local embedding server powered by Google EmbeddingGemma 2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


class EmbedRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = MODEL_ID


class EmbedResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[dict]
    usage: dict


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "ready": model is not None}


@app.post("/v1/embeddings", response_model=EmbedResponse)
def embed(req: EmbedRequest, _: str = Depends(verify_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    texts = [req.input] if isinstance(req.input, str) else req.input
    t0 = time.time()
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    elapsed = round(time.time() - t0, 3)

    log.info(f"Embedded {len(texts)} text(s) in {elapsed}s")

    return EmbedResponse(
        model=MODEL_ID,
        data=[
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
