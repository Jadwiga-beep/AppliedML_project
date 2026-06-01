import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from CNN import CNN
from preprocessing import (
    add_dimension,
    change_order,
    load_images,
    normalize,
    rgb_2_gray,
    rgb_2_hsv,
    split,
)

# Hyperparameters for training the CNN model
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

    X_train = torch.tensor(X_train).float().to(DEVICE)
    X_val = torch.tensor(X_val).float().to(DEVICE)
    y_train = torch.tensor(y_train).long().to(DEVICE)
    y_val = torch.tensor(y_val).long().to(DEVICE)

    model = CNN(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Training loop with early stopping based on validation loss
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train), device=DEVICE)
        X_train, y_train = X_train[perm], y_train[perm]

        total_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            xb, yb = X_train[i : i + BATCH_SIZE], y_train[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
            val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

        print(
            f"[{name}] Epoch {epoch + 1}/{EPOCHS} — loss: {total_loss / len(X_train):.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
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


def evaluate_validation(
    model: CNN, X_val: np.ndarray, y_val: np.ndarray, name: str
) -> float:
    """
    Evaluates the trained CNN model on the validation dataset.

    Args:
        model (CNN): The trained CNN model to evaluate.
        X_val (np.ndarray): The validation images as a numpy array.
        y_val (np.ndarray): The validation labels as a numpy array.
        name (str): The name of the model (for logging purposes).

    Returns:
        float: The validation accuracy of the model.
    """
    device = next(model.parameters()).device

    X_val = torch.tensor(X_val).float().to(device)
    y_val = torch.tensor(y_val).long().to(device)

    model.eval()
    with torch.no_grad():
        val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

    print(f"[{name}] Validation accuracy: {val_acc:.4f}")
    return val_acc


def retrain_with_best_val(
    model: CNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
) -> CNN:
    """
    Retrains the CNN model using both the training and validation datasets.

    Args:
        model (CNN): The CNN model to retrain.
        X_train (np.ndarray): The training images as a numpy array.
        y_train (np.ndarray): The training labels as a numpy array.
        X_val (np.ndarray): The validation images as a numpy array.
        y_val (np.ndarray): The validation labels as a numpy array.
        name (str): The name of the model (for logging purposes).

    Returns:
        CNN: The retrained CNN model.
    """
    device = next(model.parameters()).device

    X_combined = torch.tensor(np.concatenate([X_train, X_val])).float().to(device)
    y_combined = torch.tensor(np.concatenate([y_train, y_val])).long().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Retraining loop with early stopping based on combined training and validation loss
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_combined), device=device)
        X_combined, y_combined = X_combined[perm], y_combined[perm]

        total_loss = 0.0
        for i in range(0, len(X_combined), BATCH_SIZE):
            xb, yb = X_combined[i : i + BATCH_SIZE], y_combined[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(X_combined)
        print(f"[{name}] Retrain Epoch {epoch + 1}/{EPOCHS} — loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[{name}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_test(
    model: CNN, X_test: np.ndarray, y_test: np.ndarray, name: str
) -> float:
    """
    Evaluates the trained CNN model on the test dataset.

    Args:
        model (CNN): The trained CNN model to evaluate.
        X_test (np.ndarray): The test images as a numpy array.
        y_test (np.ndarray): The test labels as a numpy array.
        name (str): The name of the model (for logging purposes).

    Returns:
        float: The test accuracy of the model.
    """
    device = next(model.parameters()).device

    X_test = torch.tensor(X_test).float().to(device)
    y_test = torch.tensor(y_test).long().to(device)

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"[{name}] Test accuracy: {test_acc:.4f}")
    return test_acc


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
    This method saves the list of class names to a json file.

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

    pretty_names = []
    for name in loaded_names:
        name = name[5:]
        pretty_names.append(name)

    return pretty_names


def train_and_save():
    # Loading and preprocessing the data
    images, labels, class_names = load_images()
    images = normalize(images)
    num_classes = len(class_names)

    # RGB
    X_train, X_val, X_test, y_train, y_val, y_test = split(images, labels)
    model_rgb = train(
        change_order(X_train),
        y_train,
        change_order(X_val),
        y_val,
        num_classes,
        (64, 64, 3),
        "RGB",
    )
    optimized_model_rgb = retrain_with_best_val(
        model_rgb,
        change_order(X_train),
        y_train,
        change_order(X_val),
        y_val,
        "RGB",
    )

    # HSV
    X_train, X_val, X_test, y_train, y_val, y_test = split(rgb_2_hsv(images), labels)
    model_hsv = train(
        change_order(X_train),
        y_train,
        change_order(X_val),
        y_val,
        num_classes,
        (64, 64, 3),
        "HSV",
    )
    optimized_model_hsv = retrain_with_best_val(
        model_hsv,
        change_order(X_train),
        y_train,
        change_order(X_val),
        y_val,
        "HSV",
    )

    # Grayscale
    X_train, X_val, X_test, y_train, y_val, y_test = split(rgb_2_gray(images), labels)
    model_gray = train(
        add_dimension(X_train),
        y_train,
        add_dimension(X_val),
        y_val,
        num_classes,
        (64, 64, 1),
        "Grayscale",
    )
    optimized_model_gray = retrain_with_best_val(
        model_gray,
        add_dimension(X_train),
        y_train,
        add_dimension(X_val),
        y_val,
        "Grayscale",
    )


    # Saving the optimized models and class names to disk.
    save_model(optimized_model_rgb, "./models/CNN_rgb.zip")
    save_model(optimized_model_hsv, "./models/CNN_hsv.zip")
    save_model(optimized_model_gray, "./models/CNN_gray.zip")
    save_class_names(class_names, "./models/class_names.json")


if __name__ == "__train__":
    train_and_save()
