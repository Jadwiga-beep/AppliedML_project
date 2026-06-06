from evaluation import evaluation_pipeline
from preprocessing import load_and_prepare_data, split
from SVM import run_svm
from train import set_seed, train_pipeline


def main() -> None:
    """
    Hub script: trains the three CNN models, saves them to ./models/,
    then evaluates them, and prints the accuracy comparison.
    Trains and evaluates SVM baseline models for RGB, HSV, and grayscale.

    Run this once before launching the API (uvicorn api:app --reload).

    Args:
        None

    Returns:
        None
    """
    set_seed()

    # Data preprocessing
    print("\n--- Preprocessing the data ---")
    images, labels, class_names = load_and_prepare_data()
    num_classes = len(class_names)
    X_train, X_val, X_test, y_train, y_val, y_test = split(images, labels)

    # Training the CNN models
    print("\n--- Training the CNN models ---")
    train_pipeline(X_train, X_val, X_test, y_train, y_val, num_classes)

    # Evaluating the CNN models
    evaluation_pipeline(X_train, X_val, X_test, y_val, y_test)

    print("\n--- Training the SVM baseline models ---")
    run_svm(images, labels, class_names)


if __name__ == "__main__":
    main()
