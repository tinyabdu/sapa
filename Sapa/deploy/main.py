"""
deploy/main.py

FastAPI app that serves your trained GAN (image generation) and
Transformer (text/code generation) models as HTTP endpoints. Deploy
this directly to Railway.

BEFORE DEPLOYING:
1. Train both models on Colab (see image_gen/train_gan.py and
   text_gen/train_transformer.py)
2. Download the checkpoint files:
   - checkpoints/generator_final.pth  (from image_gen training)
   - checkpoints/model_final.pth + vocab.json  (from text_gen training)
3. Place them in this deploy/ folder under models/ (see folder structure below)
4. Push to GitHub, connect the repo to Railway, deploy

Folder structure expected:
    deploy/
      main.py              <- this file
      gan_model.py          <- copy from image_gen/
      transformer_model.py  <- copy from text_gen/
      models/
        generator_final.pth
        model_final.pth
        vocab.json
      requirements.txt
      Dockerfile
"""

import io
import json
import base64
import os
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torchvision.utils as vutils
from PIL import Image

from gan_model import Generator
from transformer_model import MiniGPT

app = FastAPI(title="From-Scratch AI Platform", version="1.0", lifespan=lifespan)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------------
# Load models at startup (fails loudly if checkpoints are missing, so
# you know immediately rather than getting a confusing 500 later)
# ---------------------------------------------------------------------
gan_generator = None
text_model = None
vocab = None


@asynccontextmanager
async def lifespan(app):
    global gan_generator, text_model, vocab

    gan_path = os.path.join(MODELS_DIR, "generator_final.pth")

    if os.path.exists(gan_path):
        gan_generator = Generator(latent_dim=100, img_channels=1)
        gan_generator.load_state_dict(
            torch.load(gan_path, map_location=DEVICE)
        )
        gan_generator.to(DEVICE).eval()
        print("GAN generator loaded.")
    else:
        print(
            f"WARNING: {gan_path} not found - "
            "/generate-image will be unavailable."
        )

    text_model_path = os.path.join(MODELS_DIR, "model_final.pth")
    vocab_path = os.path.join(MODELS_DIR, "vocab.json")

    if os.path.exists(text_model_path) and os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)

        text_model = MiniGPT(
            vocab_size=vocab["vocab_size"],
            embed_dim=256,
            n_heads=8,
            n_layers=6,
            block_size=256,
        )

        text_model.load_state_dict(
            torch.load(text_model_path, map_location=DEVICE)
        )
        text_model.to(DEVICE).eval()

        print("Text/code Transformer loaded.")
    else:
        print(
            f"WARNING: {text_model_path} not found - "
            "/generate-text will be unavailable."
        )

    yield

# ---------------------------------------------------------------------
# Image generation endpoint
# ---------------------------------------------------------------------
class ImageGenRequest(BaseModel):
    num_images: int = Field(default=1, ge=1, le=16)
    seed: int | None = None


@app.post("/generate-image")
def generate_image(req: ImageGenRequest):
    if gan_generator is None:
        raise HTTPException(status_code=503, detail="Image model not loaded - deploy generator_final.pth")

    if req.seed is not None:
        torch.manual_seed(req.seed)

    with torch.no_grad():
        noise = torch.randn(req.num_images, 100, device=DEVICE)
        fake_images = gan_generator(noise).cpu()

    # Build a grid and return as base64 PNG
    grid = vutils.make_grid(fake_images, normalize=True, nrow=4)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({"image_base64": b64_img, "format": "png"})



# Text/code generation endpoint
class TextGenRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = Field(default=200, ge=1, le=1000)
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    top_k: int = Field(default=40, ge=1)


@app.post("/generate-text")
def generate_text(req: TextGenRequest):
    if text_model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded - deploy model_final.pth + vocab.json")

    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}

    # Encode prompt; fall back to a single start token if prompt is empty
    # or contains characters outside the training vocabulary
    try:
        if req.prompt:
            idx = torch.tensor([[stoi[c] for c in req.prompt]], dtype=torch.long, device=DEVICE)
        else:
            idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Character {e} not in training vocabulary")

    with torch.no_grad():
        generated = text_model.generate(
            idx, max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, top_k=req.top_k,
        )

    generated_text = "".join([itos[i] for i in generated[0].tolist()])
    return JSONResponse({"generated_text": generated_text})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "image_model_loaded": gan_generator is not None,
        "text_model_loaded": text_model is not None,
        "device": str(DEVICE),
    }


@app.get("/")
def root():
    return {
        "message": "From-scratch AI platform - see /docs for API endpoints",
        "endpoints": ["/generate-image", "/generate-text", "/health"],
    }
