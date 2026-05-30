import io

import numpy as np
import torch
import torch.nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from preprocessing import IMG_SIZE, rgb_2_gray, rgb_2_hsv
from train import DEVICE, load_class_names, load_model


class ModelInput(BaseModel):
    pass


app = FastAPI()
MODELS = {}

try:
    MODELS["rgb"] = load_model("./models/CNN_rgb.zip", "rgb")
    MODELS["hsv"] = load_model("./models/CNN_hsv.zip", "hsv")
    MODELS["gray"] = load_model("./models/CNN_gray.zip", "gray")
    CLASS_NAMES = load_class_names("./models/class_names.json")
except Exception as e:
    raise HTTPException(status_code=400, detail=f"{e}")


@app.get("/")
async def root():
    return {"loaded_models": list(MODELS.keys())}


def predict(model_name: str, tensor: torch.Tensor):
    if model_name not in ("rgb", "hsv", "gray"):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    model = MODELS[model_name]

    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]

    index = int(probs.argmax().item())
    class_name = CLASS_NAMES[index]

    return {
        "predicted_class_name": class_name,
        "confidence": float(probs[index].item()),
    }


def preprocess(raw: bytes, model_name: str) -> torch.Tensor:
    """
    This method preprocesses raw image bytes into a normalized PyTorch tensor

    Args:
        raw (bytes): The raw binary data of the input image.
        model_name (str): The target model name ("rgb", "hsv", or "gray").

    Returns:
        torch.Tensor: A 4D floating-point tensor with shape (1, C, H, W) and values in the range [0.0, 1.0].

    Raises:
        HTTPException:
            - 400 error if the raw bytes cannot be parsed as a valid image.
            - 400 error if an unsupported model name is provided.
    """

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    orig_batch = (
        np.array(img)[np.newaxis, ...].astype(np.float32) / 255.0
    )  # (1, H, W, 3) in [0, 1]

    if model_name == "rgb":
        batch = orig_batch  # (1, H, W, 3)
    elif model_name == "hsv":
        batch = rgb_2_hsv(orig_batch)  # (1, H, W, 3)
    elif model_name == "gray":
        batch = rgb_2_gray(orig_batch)[..., np.newaxis]  # (1, H, W, 1)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    batch = np.transpose(batch, (0, 3, 1, 2))  # (1, H, W, C) -> (1, C, H, W)
    return torch.tensor(batch).float().to(DEVICE)


@app.post("/predict")
async def predict_all(file: UploadFile = File(...)):

    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")

    results = {}
    raw = await file.read()
    for model_name in MODELS:
        results[model_name] = predict(model_name, preprocess(raw, model_name))
    return {"filename": file.filename, "results": results}
