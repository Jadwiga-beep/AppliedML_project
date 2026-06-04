from SVM import run_svm_baseline
from train import (
    evaluate_all,
    heat_map_conf_matrix,
    load_and_prepare_data,
    train_and_save,
)


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
    train_and_save()
    evaluate_all()
    heat_map_conf_matrix()

    print("\n--- Training SVM baseline models ---")
    images, labels, _ = load_and_prepare_data()
    run_svm_baseline(images, labels, "./models")


if __name__ == "__main__":
    main()
