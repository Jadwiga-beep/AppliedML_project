import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
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

    steps.append(("pca", PCA(n_components=50, whiten=True, random_state=42)))
    steps.append(("svc", SVC(kernel="rbf", C=C, gamma="scale")))

    return Pipeline(steps)


def grid_search_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict | None = None,
    cv: int = 3,
    n_jobs: int = 2,
) -> GridSearchCV:
    """
    Performs a grid search to find the best hyperparameters for an SVM pipeline.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        param_grid (dict|None): A dictionary with parameters names (str) as keys and lists of parameter
            settings to try as values. If None, a default grid will be used.
        cv (int): Number of folds for cross-validation.
        n_jobs (int): Number of jobs to run in parallel. Defaults to 2.

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
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluates a trained SVM model on the training and test datasets, returning various performance metrics.

    Args:
        model (Pipeline): The trained SVM model to evaluate.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test labels.

    Returns:
        dict: A dictionary containing the accuracy, classification report, confusion matrix,
        and hinge loss for train and test data.
    """
    predictions = model.predict(X_test)

    train_decisions = model.decision_function(X_train)
    test_decisions = model.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "train_hinge_loss": hinge_loss(
            y_train, train_decisions, labels=np.unique(y_train)
        ),
        "test_hinge_loss": hinge_loss(y_test, test_decisions, labels=np.unique(y_test)),
    }


def svm_conf_matrix(
    cm: np.ndarray,
    class_names: list[str],
    color_space: str,
    output_dir: str = "./images",
) -> None:
    """
    Generates and saves a confusion matrix for an SVM model.

    Args:
        cm (np.ndarray): The confusion matrix as a numpy array.
        class_names (list[str]): List of class names.
        color_space (str): Name of  the color space.
        output_dir (str): Directory to save the heatmap image.

    Returns:
        None: Saves the heatmap as an image.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(11, 10))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {color_space.upper()} SVM")
    plt.savefig(f"{output_dir}/conf_matrix_svm_{color_space}.png")
    plt.close()


def plot_svm_hinge_losses(
    results: dict,
    save_path: str = "images/hinge_loss_svm.png",
) -> None:
    """
    Bar chart comparing train vs test hinge loss for each SVM color space.

    Args:
        results (dict): Keys are color-space names, values contain
            train_hinge_loss and test_hinge_loss.
        save_path (str): File path for the saved figure.

    Returns:
        None: Saves the plot as an image file.
    """
    color_spaces = list(results.keys())
    train_losses = [results[cs]["train_hinge_loss"] for cs in color_spaces]
    test_losses = [results[cs]["test_hinge_loss"] for cs in color_spaces]

    x = np.arange(len(color_spaces))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, train_losses, width, label="Train", color="#4a90d9")
    ax.bar(x + width / 2, test_losses, width, label="Test", color="#e05252")
    ax.set_xticks(x)
    ax.set_xticklabels([cs.upper() for cs in color_spaces])
    ax.set_xlabel("Color space")
    ax.set_ylabel("Hinge loss")
    ax.set_title("SVM train vs test hinge loss by color space", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved SVM hinge loss plot to {save_path}")


def run_svm(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Trains and evaluates SVM baseline models for RGB, HSV, and grayscale.

    Args:
        images (np.ndarray): A numpy array of shape (n_samples, height, width, channels)
            containing the input images in RGB format.
        labels (np.ndarray): A numpy array of shape (n_samples,) containing the class labels for each image.
        class names (list[str]): A list of names of data classes.

    Returns:
        dict: A dictionary containing the evaluation metrics for each color space (RGB, HSV, Grayscale).
    """
    class_names = [name[5:] for name in class_names]

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

        t_0 = time.time()
        search = grid_search_svm(
            svm_train,
            y_train,
            param_grid={"svc__C": [0.1, 1.0], "svc__gamma": ["scale"]},
            cv=3,
            n_jobs=-1,
        )
        training_time = time.time() - t_0

        best_model = search.best_estimator_

        metrics = evaluate_svm(best_model, svm_train, y_train, svm_test, y_test)
        metrics["best_params"] = search.best_params_
        metrics["training_time_seconds"] = training_time
        results[color_space] = metrics
        print(f"\n--- {color_space.upper()} SVM Results ---")
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Train hinge loss: {metrics['train_hinge_loss']:.4f}")
        print(f"Test hinge loss:  {metrics['test_hinge_loss']:.4f}")
        print(f"Training time:   {training_time:.1f}s")

        svm_conf_matrix(
            cm=metrics["confusion_matrix"],
            class_names=class_names,
            color_space=color_space,
            output_dir="./images",
        )

    plot_svm_hinge_losses(results, "images/hinge_loss_svm.png")

    return results
