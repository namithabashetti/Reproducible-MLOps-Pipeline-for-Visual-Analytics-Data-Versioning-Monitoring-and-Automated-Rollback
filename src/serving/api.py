"""Load one verified local YOLO checkpoint and serve bounded image requests on CPU."""

import asyncio
from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import warnings

import cv2
from fastapi import FastAPI, HTTPException, Request
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.data.preprocess import letterbox


def decode_image(body, max_pixels):
    """Bound decoded allocation before loading; accept only static PNG/JPEG images."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(body)) as source:
                if source.format not in {"PNG", "JPEG"} or getattr(source, "n_frames", 1) != 1:
                    raise HTTPException(415, "Use a single PNG or JPEG image")
                if source.width * source.height > max_pixels:
                    raise HTTPException(413, "Decoded image exceeds the pixel limit")
                source.load()
                return cv2.cvtColor(np.array(ImageOps.exif_transpose(source).convert("RGB")),
                                    cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, "Decoded image exceeds the pixel limit") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(422, "Image is empty, corrupt, or cannot be decoded") from error


def predict_image(model, image, settings):
    padded, (sx, sy, left, top) = letterbox(image, settings["image_size"], settings["pad_value"])
    result = model.predict(padded, device="cpu", imgsz=settings["image_size"],
                           conf=settings["confidence"], iou=settings["iou"],
                           max_det=settings["max_detections"], rect=False,
                           half=False, augment=False, verbose=False, save=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().copy()
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / sx, 0, image.shape[1])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / sy, 0, image.shape[0])
    return [{"class_id": int(cls), "class_name": model.names[int(cls)],
             "confidence": float(conf), "xyxy": box.tolist()}
            for box, cls, conf in zip(boxes, result.boxes.cls.tolist(), result.boxes.conf.tolist())]


def load_model(directory):
    from ultralytics import YOLO
    import torch

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = directory / "best.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("Serving checkpoint does not match the deployment manifest")
    if manifest["inference"]["device"] != "cpu":
        raise ValueError("This serving image requires CPU inference")
    torch.set_num_threads(manifest["inference"]["cpu_threads"])
    model = YOLO(str(checkpoint), task="detect")
    if model.names != {int(k): v for k, v in manifest["classes"].items()}:
        raise ValueError("Checkpoint class names differ from the approved dataset")
    size = manifest["inference"]["image_size"]
    predict_image(model, np.full((size, size, 3), 114, dtype=np.uint8), manifest["inference"])
    return model, manifest


@asynccontextmanager
async def lifespan(app):
    app.state.ready = False
    app.state.model, app.state.manifest = await run_in_threadpool(
        load_model, Path(os.environ.get("MODEL_DIRECTORY", "/model")))
    app.state.inference_lock = asyncio.Lock()
    app.state.ready = True
    logging.getLogger(__name__).info("Loaded %s version %s (%s)",
                                    app.state.manifest["model_name"],
                                    app.state.manifest["model_version"],
                                    app.state.manifest["deployment_mode"])
    try:
        yield
    finally:
        app.state.ready = False
        app.state.model = None


app = FastAPI(title="Hard-hat detector", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "Model is not ready")
    return {"status": "ready", **request.app.state.manifest}


@app.post("/predict")
async def predict(request: Request):
    state = request.app.state
    if not getattr(state, "ready", False):
        raise HTTPException(503, "Model is not ready")
    if request.headers.get("content-type", "").split(";")[0].strip() not in {"image/png", "image/jpeg"}:
        raise HTTPException(415, "Send PNG/JPEG bytes with image/png or image/jpeg Content-Type")
    settings = state.manifest["inference"]
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > settings["max_upload_bytes"]:
            raise HTTPException(413, "Upload exceeds the byte limit")
        body.extend(chunk)
    image = await run_in_threadpool(decode_image, body, settings["max_image_pixels"])
    # Ultralytics keeps mutable predictor state; serialize inference without blocking /health.
    async with state.inference_lock:
        try:
            detections = await run_in_threadpool(predict_image, state.model, image, settings)
        except Exception as error:
            logging.getLogger(__name__).exception("Prediction failed")
            raise HTTPException(500, "Model inference failed") from error
    return {"model_name": state.manifest["model_name"],
            "model_version": state.manifest["model_version"],
            "checkpoint_sha256": state.manifest["checkpoint_sha256"],
            "deployment_mode": state.manifest["deployment_mode"],
            "width": image.shape[1], "height": image.shape[0], "detections": detections}
