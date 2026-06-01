from train import train_and_save, evaluate_all, load_and_prepare_data
from SVM import run_svm_baseline


def main() -> None:
    """
    Hub script: trains and evaluates the three CNN models (RGB, HSV, grayscale),
    saves them to ./models/, then trains and evaluates the SVM baseline models.
    Run this once before launching the API (uvicorn api:app --reload).
    """
    train_and_save()
    evaluate_all()

    print("\n--- Training SVM baseline models ---")
    images, labels, _ = load_and_prepare_data()
    run_svm_baseline(images, labels, "./models")


if __name__ == "__main__":
    main()