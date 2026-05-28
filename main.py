from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from preprocessing import load_images, normalize, rgb_2_gray, rgb_2_hsv, split
from train import load_model


class ModelInput(BaseModel):
    pass


app = FastAPI()
MODELS = {}

try:
    MODELS["rgb"] = load_model("./models/CNN_rgb.zip", "rgb")
    MODELS["hsv"] = load_model("./models/CNN_hsv.zip", "hsv")
    MODELS["gray"] = load_model("./models/CNN_gray.zip", "gray")
except Exception as e:
    raise HTTPException(status_code=400, detail=f"{e}")


@app.get("/")
async def root():
    return {"loaded_models": list(MODELS.keys())}


def predict(name, data):
    return f"len={len(data)}"


def preprocess(raw, name):
    


@app.post("/predict")
async def predict_all(file: UploadFile = File(...)):

    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")

    results = {}
    raw = await file.read()
    for name in MODELS:
        results[name] = predict(MODELS[name], preprocess(raw, name))
    return {"filename": file.filename, "results": results}
