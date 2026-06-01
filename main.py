from preprocessing import (
    add_dimension,
    change_order,
    load_images,
    normalize,
    rgb_2_gray,
    rgb_2_hsv,
    split,
)
from train import (
    evaluate_test,
    evaluate_validation,
    load_model,
    retrain_with_best_val,
    save_class_names,
    save_model,
    train, load_class_names,
)
from SVM import run_svm_baseline


def main() -> None:
    """
    Main function to execute the training and evaluation process for CNN models.

    Args:
        None
    
    Returns:
        None
    """
    # Loading and preprocessing the data
    images, labels, class_names = load_images()
    images = normalize(images)
    num_classes = len(class_names)


    MODELS = {}
    try:
        MODELS["rgb"] = load_model("./models/CNN_rgb.zip", "rgb")
        MODELS["hsv"] = load_model("./models/CNN_hsv.zip", "hsv")
        MODELS["gray"] = load_model("./models/CNN_gray.zip", "gray")
        CLASS_NAMES = load_class_names("./models/class_names.json")

    # Raise an error if loading fails
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}") from e

    # Splitting the data, training, evaluating on the validation set,
    # retraining with the best validation model,
    # and evaluating on the test set for each color space representation.

    # RGB
    X_train, X_val, X_test, y_train, y_val, y_test = split(images, labels)

    # Evaluating on the validation set
    rgb_val_acc = evaluate_validation(MODELS["rgb"], change_order(X_val), y_val, "RGB")
    hsv_val_acc = evaluate_validation(MODELS["hsv"], change_order(rgb_2_hsv(X_val)), y_val, "HSV")
    gray_val_acc = evaluate_validation(
        MODELS["gray"], add_dimension(rgb_2_gray(X_val)), y_val, "Grayscale"
    )

    # Evaluating on the test set
    rgb_test_acc = evaluate_test(
        MODELS["rgb"], change_order(X_test), y_test, "RGB"
    )
    hsv_test_acc = evaluate_test(
        MODELS["hsv"], change_order(rgb_2_hsv(X_test)), y_test, "HSV"
    )
    gray_test_acc = evaluate_test(
        MODELS["gray"], add_dimension(rgb_2_gray(X_test)), y_test, "Grayscale"
    )

    # Printing the comparison of validation and test accuracies for all models.
    print("\n--- Comparison between CNN models on the validation set ---")
    print(f"RGB validation accuracy: {rgb_val_acc:.4f}")
    print(f"HSV validation accuracy: {hsv_val_acc:.4f}")
    print(f"Grayscale validation accuracy: {gray_val_acc:.4f}")

    # Evaluating on the test set and printing the results for all models.
    print("\n--- Comparison between CNN models on the test set ---")
    print(f"RGB test accuracy: {rgb_test_acc:.4f}")
    print(f"HSV test accuracy: {hsv_test_acc:.4f}")
    print(f"Grayscale test accuracy: {gray_test_acc:.4f}")

    print("\n--- Training SVM baseline models ---")
    run_svm_baseline(images, labels, "./models")


# Running the main function
if __name__ == "__main__":
    main()
