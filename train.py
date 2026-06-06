import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from CNN import CNN
from preprocessing import process_data

EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5
LR = 1e-3
INPUT_SHAPES = {
    "rgb": (64, 64, 3),
    "hsv": (64, 64, 3),
    "gray": (64, 64, 1),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across numpy, torch, and cuda.

    Args:
        seed (int = 42): The random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    input_shape: tuple[int, int, int],
    name: str,
) -> CNN:
    """
    Trains a CNN model on the given training dataset.

    Args:
        X_train (np.ndarray): The training images as a numpy array.
        y_train (np.ndarray): The training labels as a numpy array.
        X_val (np.ndarray): The validation images as a numpy array.
        y_val (np.ndarray): The validation labels as a numpy array.
        num_classes (int): The number of classes for classification.
        input_shape (tuple[int, int, int]): The shape of the input images (height, width, channels).
        name (str): The name of the model (for logging purposes).

    Returns:
        CNN: The trained CNN model.
    """
    set_seed()

    X_train_t = torch.tensor(X_train).float().to(DEVICE)
    X_val_t = torch.tensor(X_val).float().to(DEVICE)
    y_train_t = torch.tensor(y_train).long().to(DEVICE)
    y_val_t = torch.tensor(y_val).long().to(DEVICE)

    model = CNN(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Training loop with early stopping based on validation loss
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train_t), device=DEVICE)
        X_train_t, y_train_t = X_train_t[perm], y_train_t[perm]

        total_loss = 0.0
        for i in range(0, len(X_train_t), BATCH_SIZE):
            xb, yb = X_train_t[i : i + BATCH_SIZE], y_train_t[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
            val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()

        print(
            f"[{name}] Epoch {epoch + 1}/{EPOCHS} — loss: {total_loss / len(X_train_t):.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[{name}] Early stopping.")
                break

    model.load_state_dict(best_state)

    return model


def save_model(model: CNN, file_name: str) -> bool:
    """
    This method saves a trained model in a new file.

    Args:
        model (CNN): The trained CNN model to save.
        file_name(str): The name of the file under which the model is saved.

    Returns:
        bool: True if the model was successfully saved, otherwise False.
    """
    folder = os.path.dirname(file_name)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), file_name)

    print(f"Successfully saved {file_name}")
    return True


def load_model(file_name: str, model_name: str) -> CNN:
    """
    This method loads a trained model from a zip file.

    Args:
        file_name(str): The name of the file under which the model is saved.

    Returns:
        CNN: The saved CNN model.
    """
    if model_name not in INPUT_SHAPES:
        raise ValueError(f"Unknown model name '{model_name}'")

    state = torch.load(file_name, map_location=DEVICE)
    num_classes = state["fc2.bias"].shape[0]
    model = CNN(input_shape=INPUT_SHAPES[model_name], num_classes=num_classes).to(
        DEVICE
    )
    model.load_state_dict(state)
    model.eval()

    print(f"Successfully loaded {file_name}")
    return model


def save_class_names(class_names: list, file_name: str) -> bool:
    """
    This method saves the list of class names to a .json file.

    Args:
        class_names(list): The list of class names
        file_name(str): The name of the file under which the class list is saved.

    Returns:
        bool: True if the class list was successfully saved, otherwise False.
    """
    folder = os.path.dirname(file_name)

    if folder:
        os.makedirs(folder, exist_ok=True)
    json.dump(class_names, open(file_name, "w"))

    print(f"Successfully saved {file_name}")
    return True


def load_class_names(file_name: str) -> list:
    """
    This method loads class names from a json file.

    Args:
        file_name(str): The name of the file under which the class names are stored.

    Returns:
        list: The list of the class names.
    """

    if not os.path.exists(file_name):
        raise Exception(f"Can not find {file_name} to load")

    loaded_names = json.load(open(file_name))
    print(f"Successfully loaded {file_name}")

    return [name[5:] for name in loaded_names]


def train_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
) -> None:
    """
    Main pipeline function: loads and processes the data, trains the CNN models,
    and saves the best models and class names.

    Args:
        X_train (np.ndarray): The training images as a numpy array.
        X_val (np.ndarray): The validation images as a numpy array.
        X_test (np.ndarray): The test images as a numpy array.
        y_train (np.ndarray): The training labels as a numpy array.
        y_val (np.ndarray): The validation labels as a numpy array.
        num_classes (int): NUmber of classes.

    Returns:
        None - the function performs the entire training pipeline.
    """
    rgb, hsv, gray = process_data(X_train, X_val, X_test)
    X_train_rgb, X_val_rgb, X_test_rgb = rgb
    X_train_hsv, X_val_hsv, X_test_hsv = hsv
    X_train_gray, X_val_gray, X_test_gray = gray

    # RGB
    model_rgb = train(
        X_train_rgb, y_train, X_val_rgb, y_val, num_classes, INPUT_SHAPES["rgb"], "RGB"
    )
    print("Successfully trained the RGB model.")

    # HSV
    model_hsv = train(
        X_train_hsv, y_train, X_val_hsv, y_val, num_classes, INPUT_SHAPES["hsv"], "HSV"
    )
    print("Successfully trained the HSV model.")

    # Grayscale
    model_gray = train(
        X_train_gray,
        y_train,
        X_val_gray,
        y_val,
        num_classes,
        INPUT_SHAPES["gray"],
        "Grayscale",
    )
    print("Successfully trained the Grayscale model.")

    # Saving the models
    save_model(model_rgb, "./models/CNN_rgb.zip")
    save_model(model_hsv, "./models/CNN_hsv.zip")
    save_model(model_gray, "./models/CNN_gray.zip")
