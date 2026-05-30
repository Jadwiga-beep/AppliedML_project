import numpy as np

from preprocessing import load_images, normalize, rgb_2_gray, rgb_2_hsv, split
from train import (
    evaluate_test,
    evaluate_validation,
    retrain_with_best_val,
    save_class_names,
    save_model,
    train,
)


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


# Main function to execute the training and evaluation process
def main():
    # Loading and preprocessing the data
    images, labels, class_names = load_images()
    images = normalize(images)
    num_classes = len(class_names)

    # Splitting the data, training, evaluating on the validation set,
    # retraining with the best validation model,
    # and evaluating on the test set for each color space representation.

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
    rgb_val_acc = evaluate_validation(model_rgb, change_order(X_val), y_val, "RGB")
    optimized_model_rgb = retrain_with_best_val(
        model_rgb,
        change_order(X_train),
        y_train,
        change_order(X_val),
        y_val,
        "RGB",
    )

    rgb_test_acc = evaluate_test(
        optimized_model_rgb, change_order(X_test), y_test, "RGB"
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

    hsv_val_acc = evaluate_validation(model_hsv, change_order(X_val), y_val, "HSV")
    optimized_model_hsv = retrain_with_best_val(
        model_hsv,
        change_order(X_train),
        y_train,
        change_order(X_val),
        y_val,
        "HSV",
    )
    hsv_test_acc = evaluate_test(
        optimized_model_hsv, change_order(X_test), y_test, "HSV"
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
    gray_val_acc = evaluate_validation(
        model_gray, add_dimension(X_val), y_val, "Grayscale"
    )
    optimized_model_gray = retrain_with_best_val(
        model_gray,
        add_dimension(X_train),
        y_train,
        add_dimension(X_val),
        y_val,
        "Grayscale",
    )
    gray_test_acc = evaluate_test(
        optimized_model_gray, add_dimension(X_test), y_test, "Grayscale"
    )

    save_model(optimized_model_rgb, "./models/CNN_rgb.zip")
    save_model(optimized_model_hsv, "./models/CNN_hsv.zip")
    save_model(optimized_model_gray, "./models/CNN_gray.zip")
    save_class_names(class_names, "./models/class_names.json")

    # Printing the comparison of validation and test accuracies for all models.
    print("\n--- Comparison between models on the validation set ---")
    print(f"RGB validation accuracy: {rgb_val_acc:.4f}")
    print(f"HSV validation accuracy: {hsv_val_acc:.4f}")
    print(f"Grayscale validation accuracy: {gray_val_acc:.4f}")

    print("\n--- Comparison between models on the test set ---")
    print(f"RGB test accuracy: {rgb_test_acc:.4f}")
    print(f"HSV test accuracy: {hsv_test_acc:.4f}")
    print(f"Grayscale test accuracy: {gray_test_acc:.4f}")


# Running the main function
if __name__ == "__main__":
    main()
