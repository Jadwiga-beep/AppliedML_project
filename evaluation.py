import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from CNN import CNN
from preprocessing import process_data
from train import DEVICE, load_class_names, load_model


def _load_models() -> dict:
    """
    Loads the saved CNN models.

    Args:
        None

    Returns:
        dict: The saved CNN models.

    Raises:
        RuntimeError: If there is an error loading the models or data.
    """
    MODELS = {}
    try:
        MODELS["rgb"] = load_model("./models/CNN_rgb.zip", "rgb")
        MODELS["hsv"] = load_model("./models/CNN_hsv.zip", "hsv")
        MODELS["gray"] = load_model("./models/CNN_gray.zip", "gray")
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}") from e

    return MODELS


def evaluate_validation(model: CNN, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Evaluates the trained CNN model on the validation dataset.

    Args:
        model (CNN): The trained CNN model to evaluate.
        X_val (np.ndarray): The validation images as a numpy array.
        y_val (np.ndarray): The validation labels as a numpy array.

    Returns:
        float: The validation accuracy of the model.
    """
    device = next(model.parameters()).device

    X_val_t = torch.tensor(X_val).float().to(device)
    y_val_t = torch.tensor(y_val).long().to(device)

    model.eval()
    with torch.no_grad():
        val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()

    return val_acc


def evaluate_test(model: CNN, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Evaluates the trained CNN model on the test dataset.

    Args:
        model (CNN): The trained CNN model to evaluate.
        X_test (np.ndarray): The test images as a numpy array.
        y_test (np.ndarray): The test labels as a numpy array.

    Returns:
        float: The test accuracy of the model.
    """
    device = next(model.parameters()).device

    X_test_t = torch.tensor(X_test).float().to(device)
    y_test_t = torch.tensor(y_test).long().to(device)

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()

    return test_acc


def heat_map_conf_matrix(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Generates and saves a heatmap of the confusion matrix for all models and test dataset.

    Args:
        X_train (np.ndarray): The training images as a numpy array.
        X_val (np.ndarray): The validation images as a numpy array.
        X_test (np.ndarray): The test images as a numpy array.
        y_test (np.ndarray): The test labels as a numpy array.

    Returns:
        None: Saves the heat map as an image.
    """
    MODELS = _load_models()

    class_names = load_class_names("./models/class_names.json")

    rgb_X_test, hsv_X_test, gray_X_test = [
        split[2] for split in process_data(X_train, X_val, X_test)
    ]

    X_test_versions = {
        "rgb": torch.tensor(rgb_X_test).float().to(DEVICE),
        "hsv": torch.tensor(hsv_X_test).float().to(DEVICE),
        "gray": torch.tensor(gray_X_test).float().to(DEVICE),
    }

    os.makedirs("./images", exist_ok=True)

    for model_name, model in MODELS.items():
        with torch.no_grad():
            y_pred = model(X_test_versions[model_name]).argmax(1).cpu().numpy()

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(11, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Heat Map — {model_name.upper()}")
        plt.savefig(f"./images/heat_map_{model_name}.png")
        plt.close()


def evaluation_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Loads the saved CNN models, evaluates them on the validation and test sets,
    prints the accuracy comparison across the three color spaces,
    and generates the heat maps.

    Args:
        X_train (np.ndarray): The training images as a numpy array.
        X_val (np.ndarray): The validation images as a numpy array.
        X_test (np.ndarray): The test images as a numpy array.
        y_val (np.ndarray): The validation labels as a numpy array.
        y_test (np.ndarray): The test labels as a numpy array.


    Returns:
        None - Performs the whole evaluation pipeline.
    """
    rgb, hsv, gray = process_data(X_train, X_val, X_test)
    X_train_rgb, X_val_rgb, X_test_rgb = rgb
    X_train_hsv, X_val_hsv, X_test_hsv = hsv
    X_train_gray, X_val_gray, X_test_gray = gray

    MODELS = _load_models()

    # Validation accuracy
    rgb_val_acc = evaluate_validation(MODELS["rgb"], X_val_rgb, y_val)
    hsv_val_acc = evaluate_validation(MODELS["hsv"], X_val_hsv, y_val)
    gray_val_acc = evaluate_validation(MODELS["gray"], X_val_gray, y_val)

    # Test accuracy
    rgb_test_acc = evaluate_test(MODELS["rgb"], X_test_rgb, y_test)
    hsv_test_acc = evaluate_test(MODELS["hsv"], X_test_hsv, y_test)
    gray_test_acc = evaluate_test(MODELS["gray"], X_test_gray, y_test)

    print("\n--- Comparison between CNN models on the validation set ---")
    print(f"RGB validation accuracy: {rgb_val_acc:.4f}")
    print(f"HSV validation accuracy: {hsv_val_acc:.4f}")
    print(f"Grayscale validation accuracy: {gray_val_acc:.4f}")

    print("\n--- Comparison between CNN models on the test set ---")
    print(f"RGB test accuracy: {rgb_test_acc:.4f}")
    print(f"HSV test accuracy: {hsv_test_acc:.4f}")
    print(f"Grayscale test accuracy: {gray_test_acc:.4f}")

    heat_map_conf_matrix(X_train, X_val, X_test, y_test)
