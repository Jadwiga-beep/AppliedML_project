import io
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from sklearn.model_selection import train_test_split

IMG_SIZE = (64, 64)
DATA_DIRS = {
    "fruits": "data/fruits",
    "vegetables": "data/vegetables",
}


def load_images() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads images from the specified directories, resizes them,
    and returns the image data, labels, and class names.

    Returns:
        images (np.ndarray): Array of image data.
        labels (np.ndarray): Array of corresponding labels for the images.
        class_names (list[str]): List of class names corresponding to the labels.
    """
    images = []
    labels = []
    class_names = []

    for folder in DATA_DIRS.values():
        for class_name in sorted(os.listdir(folder)):
            class_path = os.path.join(folder, class_name)
            if not os.path.isdir(class_path):
                continue
            label = len(class_names)
            class_names.append(class_name)

            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(class_path, fname)

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMG_SIZE)
                    images.append(np.array(img))
                    labels.append(label)
                except Exception:
                    continue

    return np.array(images), np.array(labels), class_names


def rgb_2_hsv(images: np.ndarray) -> np.ndarray:
    """
    Converts RGB images to HSV color space.

    Args:
        images (np.ndarray): Array of RGB images with shape (N, H, W, 3).

    Returns:
        np.ndarray: Array of HSV images with shape (N, H, W, 3).
    """
    hsv_images = []

    for image in images:
        hsv = rgb2hsv(image)
        hsv_images.append(hsv)

    return np.array(hsv_images)


def normalize(images: np.ndarray) -> np.ndarray:
    """
    Normalizes image pixel values to [0, 1].

    Args:
        images (np.ndarray): Array of images with integer or float pixel values.

    Returns:
        np.ndarray: Array of images with float pixel values in [0, 1].
    """
    return images.astype(np.float32) / 255.0


def rgb_2_gray(images: np.ndarray) -> np.ndarray:
    """
    Converts RGB images to grayscale using skimage.

    Args:
        images (np.ndarray): Array of RGB images with shape (N, H, W, 3).

    Returns:
        np.ndarray: Array of grayscale images with shape (N, H, W).
    """
    grayscale_images = []

    for image in images:
        gray = rgb2gray(image)
        grayscale_images.append(gray)

    return np.array(grayscale_images)


def split(
    images: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits images and labels into train, validation, and test sets (70:15:15).

    Args:
        images (np.ndarray): Array of images.
        labels (np.ndarray): Array of corresponding labels.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(raw, name):
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    img = np.array(img)
    # XXX
