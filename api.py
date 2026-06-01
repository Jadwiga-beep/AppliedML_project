import io

import numpy as np
import torch
import torch.nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from preprocessing import IMG_SIZE, rgb_2_gray, rgb_2_hsv
from train import DEVICE, load_class_names, load_model

MODELS = {}


class Prediction(BaseModel):
    """
    Class representing the prediction results for a single model, including the predicted class name,
    confidence score, and probabilities for all classes.
    """

    predicted_class_name: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    """
    Class representing the response from the prediction endpoint, including the original filename,
    and a mapping of model names to their respective predictions.
    """

    filename: str
    results: dict[str, Prediction]


class RootResponse(BaseModel):
    """
    Class representing the response from the root endpoint, including the list of loaded models.
    """

    loaded_models: list[str]


# Initialize the FastAPI app and load the models and class names
app = FastAPI(
    title="Fruit & Vegetable Classifier",
    description="Classifies images using three CNNs (RGB, HSV, grayscale color spaces).",
    version="1.0.0",
)

# Load models and class names
try:
    MODELS["rgb"] = load_model("./models/CNN_rgb.zip", "rgb")
    MODELS["hsv"] = load_model("./models/CNN_hsv.zip", "hsv")
    MODELS["gray"] = load_model("./models/CNN_gray.zip", "gray")
    CLASS_NAMES = load_class_names("./models/class_names.json")

# Raise an error if loading fails
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}") from e


# Define API endpoints
@app.get(
    "/",
    response_model=RootResponse,
    summary="List loaded models",
    tags=["Status"],
)

async def root() -> dict:
    """
    Returns the CNN models currently loaded in memory.

    Args:
        None

    Returns:
        dict: A dictionary with a single key "loaded_models" mapping to a list of the loaded model names.
    """
    return {"loaded_models": list(MODELS.keys())}


def api_preprocess(raw: bytes, model_name: str) -> torch.Tensor:
    """
    This method preprocesses raw image bytes into a normalized PyTorch tensor.

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

    orig_batch = np.array(img)[np.newaxis, ...].astype(np.float32) / 255.0

    if model_name == "rgb":
        batch = orig_batch
    elif model_name == "hsv":
        batch = rgb_2_hsv(orig_batch)
    elif model_name == "gray":
        batch = rgb_2_gray(orig_batch)[..., np.newaxis]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.tensor(batch).float().to(DEVICE)


def api_predict(model_name: str, tensor: torch.Tensor) -> dict:
    """
    Makes a prediction using the specified model and input tensor.

    Args:
        model_name (str): The name of the model to use for prediction ("rgb", "hsv", or "gray").
        tensor (torch.Tensor): A 4D tensor of shape (1, C, H, W)
            representing the preprocessed input image.

    Returns:
        dict: A dictionary containing the predicted class name, confidence score,
            and probabilities for all classes.
    """
    if model_name not in ("rgb", "hsv", "gray"):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    model = MODELS[model_name]

    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]

    index = int(probs.argmax().item())
    class_name = CLASS_NAMES[index]

    sorted_probs, sorted_idx = torch.topk(probs, probs.shape[0])
    probabilities = {
        CLASS_NAMES[int(i)]: round(float(p), 2)
        for p, i in zip(sorted_probs, sorted_idx)
    }

    return {
        "predicted_class_name": class_name,
        "confidence": float(probs[index].item()),
        "probabilities": probabilities,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify an image",
    tags=["Inference"],
    responses={
        400: {"description": "Invalid image or unknown model"},
        503: {"description": "No models loaded"},
    },
)

async def predict_all(
    file: UploadFile = File(..., description="Image file (jpg, jpeg, png) to classify"),
) -> dict:
    """
    Runs the uploaded image through all loaded models and returns the
    predicted class and confidence for each color space.

    Args:
        file (UploadFile): The uploaded image file (jpg, jpeg, or png) to classify.

    Returns:
        dict: A dictionary with the original "filename" and a "results" mapping where
              each model name (rgb, hsv, gray) maps to its predicted class
             name and confidence score.

    Raises:
        HTTPException:
            - 503 error if no models are loaded.
            - 400 error if the uploaded image is invalid
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")

    results = {}
    raw = await file.read()

    for model_name in MODELS:
        results[model_name] = api_predict(model_name, api_preprocess(raw, model_name))

    return {"filename": file.filename, "results": results}
