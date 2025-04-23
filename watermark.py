# server.py
# ---------
# FastAPI service wrapping the watermark logic (soft red-list method).
# Includes watermark implementation and API endpoints in one file.

import hashlib
import hmac
import math
import random
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---- Watermark logic ----

# Stores registered models: model_id -> {vocab, secret_bytes, gamma, delta}
MODEL_KEYS: dict[str, dict] = {}

def register_model(model_id: str, vocab: list[str], secret: str, gamma: float = 0.5, delta: float = 2.0):
    """
    Register a model with its vocabulary, secret key, green-list fraction γ, and hardness δ.
    """
    MODEL_KEYS[model_id] = {
        "vocab": vocab,
        "secret": secret.encode(),
        "gamma": gamma,
        "delta": delta
    }

def _is_green(secret: bytes, prev_token: str, token: str, gamma: float) -> bool:
    """
    Pseudo-randomly decide if `token` is in the green list for this position.
    Uses HMAC(secret, prev_token||token) as a PRF → uniform [0,1).
    """
    data = (prev_token + token).encode()
    digest = hmac.new(secret, data, hashlib.sha256).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return u < gamma

def select_watermarked_token(model_id: str, distribution: dict[str, float], prev_token: str) -> str:
    """
    Apply soft watermark: boost green-list logits by δ, renormalize, and sample.
    `distribution` is token→prob from the LM for the next token.
    """
    model = MODEL_KEYS[model_id]  # KeyError if missing
    secret = model["secret"]
    gamma = model["gamma"]
    delta = model["delta"]

    boosted: dict[str, float] = {}
    total = 0.0
    alpha = math.exp(delta)

    for tok, p in distribution.items():
        w = p * alpha if _is_green(secret, prev_token, tok, gamma) else p
        boosted[tok] = w
        total += w

    for tok in boosted:
        boosted[tok] /= total

    tokens, weights = zip(*boosted.items())
    return random.choices(tokens, weights, k=1)[0]

def detect_watermark(model_id: str, tokens: list[str]) -> float:
    """
    Compute the z-score for green-list density over the sequence:
    z = (count_green - γ·T) / sqrt(T·γ(1-γ)).
    """
    model = MODEL_KEYS[model_id]  # KeyError if missing
    secret = model["secret"]
    gamma = model["gamma"]

    count = 0
    T = len(tokens)
    for i, tok in enumerate(tokens):
        prev = tokens[i-1] if i > 0 else ""
        if _is_green(secret, prev, tok, gamma):
            count += 1

    if T == 0:
        return 0.0
    return (count - gamma * T) / math.sqrt(T * gamma * (1 - gamma))


# ---- FastAPI service ----

app = FastAPI()

class RegisterRequest(BaseModel):
    model_id: str
    vocab: list[str]
    secret: str
    gamma: float = Field(0.5, ge=0.0, le=1.0)
    delta: float = Field(2.0, ge=0.0)

class WatermarkRequest(BaseModel):
    model_id: str
    distribution: dict[str, float]
    prev_token: str = Field("", description="Last generated token (or empty for start)")

class DetectRequest(BaseModel):
    model_id: str
    tokens: list[str]

@app.post("/register")
def register(req: RegisterRequest):
    register_model(req.model_id, req.vocab, req.secret, req.gamma, req.delta)
    return {"status": "registered"}

@app.post("/watermark")
def watermark_token(req: WatermarkRequest):
    tok = select_watermarked_token(req.model_id, req.distribution, req.prev_token)
    return {"token": tok}

@app.post("/detect")
def detect(req: DetectRequest):
    z = detect_watermark(req.model_id, req.tokens)
    return {"z_score": z}

# To run:
#   pip install fastapi uvicorn
#   uvicorn server:app --reload --port 8000
