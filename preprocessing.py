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

    Args:
        None

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


def normalize(images: np.ndarray) -> np.ndarray:
    """
    Normalizes image pixel values to [0, 1].

    Args:
        images (np.ndarray): Array of images with integer or float pixel values.

    Returns:
        np.ndarray: Array of images with float pixel values in [0, 1].
    """
    return images.astype(np.float32) / 255.0


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


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Flattens images to 1D vectors for SVM input.

    Args:
        images (np.ndarray): Array of images with shape (N, H, W) or (N, H, W, C).

    Returns:
        np.ndarray: Flattened array of shape (N, H*W) or (N, H*W*C).
    """
    return images.reshape(images.shape[0], -1)


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


def change_order(X: np.ndarray) -> np.ndarray:
    """
    Changes the order of dimensions from (N, H, W, C) to
    (N, C, H, W) for PyTorch compatibility.

    Args:
        X (np.ndarray): The input images with shape (N, H, W, C).

    Returns:
        np.ndarray: The images with shape (N, C, H, W).
    """
    return np.transpose(X, (0, 3, 1, 2))


def add_dimension(X: np.ndarray) -> np.ndarray:
    """
    Adds a new dimension to the input array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        np.ndarray: The array with an additional dimension.
    """
    return X[:, np.newaxis, :, :]


def load_and_prepare_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads the image dataset, normalizes pixel values to [0, 1], and returns
    the images, labels, and class names.

    Args:
        None

    Returns:
        tuple: (images, labels, class_names)
    """
    images, labels, class_names = load_images()
    images = normalize(images)

    return images, labels, class_names


def process_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Processes the data to be in the correct color space and format.

    Args:
        X_train (np.ndarray): The training images as a numpy array.
        X_val (np.ndarray): The validation images as a numpy array.
        X_test (np.ndarray): The test images as a numpy array.

    Returns:
        tuple: The processed images in 3 colour spaces.
    """
    # RGB
    X_train_rgb, X_val_rgb, X_test_rgb = (
        change_order(X_train),
        change_order(X_val),
        change_order(X_test),
    )
    rgb = [X_train_rgb, X_val_rgb, X_test_rgb]

    X_train_hsv, X_val_hsv, X_test_hsv = (
        change_order(rgb_2_hsv(X_train)),
        change_order(rgb_2_hsv(X_val)),
        change_order(rgb_2_hsv(X_test)),
    )
    hsv = X_train_hsv, X_val_hsv, X_test_hsv

    X_train_gray, X_val_gray, X_test_gray = (
        add_dimension(rgb_2_gray(X_train)),
        add_dimension(rgb_2_gray(X_val)),
        add_dimension(rgb_2_gray(X_test)),
    )
    gray = X_train_gray, X_val_gray, X_test_gray

    return rgb, hsv, gray
