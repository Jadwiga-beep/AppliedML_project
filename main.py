from train import train_and_save, evaluate_all


def main() -> None:
    """
    Hub script: trains the three CNN models, saves them to ./models/,
    then evaluates them and prints the accuracy comparison.
    Run this once before launching the API (uvicorn api:app --reload).
    """
    train_and_save()
    evaluate_all()


if __name__ == "__main__":
    main()