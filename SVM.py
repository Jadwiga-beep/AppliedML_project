import json
import time
from pathlib import Path

import numpy as np
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


def svm_pipeline(C: float = 1.0, standardize: bool = True) -> Pipeline:
    """
    Builds an SVM pipeline with optional standard scaling.

    Args:
        C (float): Regularization parameter. The strength of the regularization is inversely proportional to C.
        standardize (bool): Whether to include a StandardScaler in the pipeline.

    Returns:
        Pipeline: A scikit-learn Pipeline object with the specified SVM configuration.
    """
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svc", SVC(kernel="rbf", C=C, gamma="scale")))

    return Pipeline(steps)


def grid_search_svm(
    X_train: list,
    y_train: list,
    param_grid: dict | None = None,
    cv: int = 3,
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Performs a grid search to find the best hyperparameters for an SVM pipeline.

    Args:
        X_train (list): Training feature data.
        y_train (list): Training labels.
        param_grid (dict|None): A dictionary with parameters names (str) as keys and lists of parameter
            settings to try as values. If None, a default grid will be used.
        cv (int): Number of folds for cross-validation.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.

    Returns:
        GridSearchCV: The fitted GridSearchCV object containing the results of the search.
    """
    if param_grid is None:
        param_grid = {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", "auto"],
        }

    pipeline = svm_pipeline()
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


def evaluate_svm(
    model: Pipeline, X_train: list, y_train: list, X_test: list, y_test: list
) -> dict:
    """
    Evaluates a trained SVM model on the training and test datasets, returning various performance metrics.

    Args:
        model (Pipeline): The trained SVM model to evaluate.
        X_train (list): Training feature data.
        y_train (list): Training labels.
        X_test (list): Test feature data.
        y_test (list): Test labels.

    Returns:
        dict: A dictionary containing the accuracy, classification report, confusion matrix, and hinge loss for
    """
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


def save_svm(model: Pipeline, file_path: str) -> None:
    """
    Saves a trained SVM pipeline to disk.

    Args:
        model (Pipeline): The trained SVM pipeline to save.
        file_path (str): The path where the model should be saved. The directory will be created if
                     it does not exist. The file will be overwritten if it already exists.

    Returns:
        None
    """
    folder = Path(file_path).parent
    folder.mkdir(parents=True, exist_ok=True)
    dump(model, file_path)


def load_svm(file_path: str) -> Pipeline:
    """
    Loads a saved SVM pipeline from disk.

    Args:
        file_path (str): The path to the saved SVM model file. The file must exist and
                   be a valid joblib file containing a scikit-learn Pipeline object.
    Returns:
        Pipeline: The loaded SVM pipeline object.

    Raises:
        RuntimeError: If the model file cannot be loaded.
    """
    return load(file_path)


def run_svm_baseline(
    images: np.ndarray, labels: np.ndarray, output_dir: str = "./models"
) -> dict:
    """
    Trains and evaluates SVM baseline models for RGB, HSV, and grayscale.

    Args:
        images (np.ndarray): A numpy array of shape (n_samples, height, width, channels)
            containing the input images in RGB format.
        labels (np.ndarray): A numpy array of shape (n_samples,) containing the class labels for each image.
        output_dir (str): The directory where the trained SVM models and
                    results will be saved. The directory will be created if it does not exist.

    Returns:
        dict: A dictionary containing the evaluation metrics for each color space (RGB, HSV, grayscale).
    """
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
        print(
            f"{color_space.upper()} SVM accuracy: {metrics['accuracy']:.4f}  "
            f"(train hinge: {metrics['train_hinge_loss']:.4f}, "
            f"test hinge: {metrics['test_hinge_loss']:.4f}, "
            f"time: {training_time:.1f}s)"
        )

    output_path = Path(output_dir) / "svm_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results
