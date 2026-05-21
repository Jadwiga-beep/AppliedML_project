from preprocessing import load_images, normalize, split, rgb_2_hsv, rgb_2_gray
from train import train, evaluate
import numpy as np


def change_order(X):
    return np.transpose(X, (0, 3, 1, 2))

def add_dimension(X):
    return X[:, np.newaxis, :, :]


def main():
    images, labels, class_names = load_images()
    images = normalize(images)
    num_classes = len(class_names)

    # RGB
    X_train, X_val, X_test, y_train, y_val, y_test = split(images, labels)
    model_rgb = train(change_order(X_train), y_train, change_order(X_val), y_val, num_classes, (64, 64, 3), "RGB")
    rgb_acc = evaluate(model_rgb, change_order(X_test), y_test, "RGB")

    # HSV
    X_train, X_val, X_test, y_train, y_val, y_test = split(rgb_2_hsv(images), labels)
    model_hsv = train(change_order(X_train), y_train, change_order(X_val), y_val, num_classes, (64, 64, 3), "HSV")
    hsv_acc = evaluate(model_hsv, change_order(X_test), y_test, "HSV")

    # Grayscale
    X_train, X_val, X_test, y_train, y_val, y_test = split(rgb_2_gray(images), labels)
    model_gray = train(add_dimension(X_train), y_train, add_dimension(X_val), y_val, num_classes, (64, 64, 1), "Grayscale")
    gray_acc = evaluate(model_gray, add_dimension(X_test), y_test, "Grayscale")

    print("\n--- Comparison ---")
    print(f"RGB test accuracy: {rgb_acc:.4f}")
    print(f"HSV test accuracy: {hsv_acc:.4f}")
    print(f"Grayscale test accuracy: {gray_acc:.4f}")


if __name__ == "__main__":
    main()