import json
import time
from pathlib import Path

from joblib import dump, load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    hinge_loss,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing import flatten_images, rgb_2_gray, rgb_2_hsv, split


def build_svm_pipeline(
    kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", standardize: bool = True
) -> Pipeline:
    """Builds an SVM pipeline with optional standard scaling."""
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svc", SVC(kernel=kernel, C=C, gamma=gamma)))
    return Pipeline(steps)


def grid_search_svm(
    X_train,
    y_train,
    param_grid=None,
    cv: int = 3,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Runs grid search over SVM hyperparameters and returns the fitted search object."""
    if param_grid is None:
        param_grid = {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", "auto"],
        }
    pipeline = build_svm_pipeline()
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_svm(model, X_train, y_train, X_test, y_test) -> dict:
    """Evaluates a trained SVM pipeline on train and test data."""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "train_hinge_loss": hinge_loss(y_train, model.decision_function(X_train)),
        "test_hinge_loss": hinge_loss(y_test, model.decision_function(X_test)),
    }


def save_svm(model, file_path: str) -> None:
    """Saves a trained SVM pipeline to disk."""
    folder = Path(file_path).parent
    folder.mkdir(parents=True, exist_ok=True)
    dump(model, file_path)


def load_svm(file_path: str):
    """Loads a saved SVM pipeline from disk."""
    return load(file_path)


def run_svm_baseline(images, labels, output_dir: str = "./models") -> dict:
    """Trains and evaluates SVM baseline models for RGB, HSV, and grayscale."""
    color_spaces = {
        "rgb": images,
        "hsv": rgb_2_hsv(images),
        "gray": rgb_2_gray(images),
    }
    results = {}

    for color_space, X in color_spaces.items():
        X_train, _, X_test, y_train, _, y_test = split(X, labels)
        svm_train = flatten_images(X_train)
        svm_test = flatten_images(X_test)

        t0 = time.time()
        search = grid_search_svm(
            svm_train,
            y_train,
            param_grid={"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", "auto"]},
            cv=3,
            n_jobs=-1,
        )
        training_time = time.time() - t0

        best_model = search.best_estimator_
        save_svm(best_model, f"{output_dir}/svm_{color_space}.joblib")

        metrics = evaluate_svm(best_model, svm_train, y_train, svm_test, y_test)
        metrics["best_params"] = search.best_params_
        metrics["training_time_seconds"] = training_time
        results[color_space] = metrics
        print(f"{color_space.upper()} SVM accuracy: {metrics['accuracy']:.4f}  "
              f"(train hinge: {metrics['train_hinge_loss']:.4f}, "
              f"test hinge: {metrics['test_hinge_loss']:.4f}, "
              f"time: {training_time:.1f}s)")

    output_path = Path(output_dir) / "svm_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results
