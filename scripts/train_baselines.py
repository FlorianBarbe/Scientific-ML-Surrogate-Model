"""Entraîne un baseline scikit-learn reproductible sur le dataset thermique."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scientific_ml_surrogate_model.dataset import load_dataset, split_dataset
from scientific_ml_surrogate_model.metrics import compute_all_metrics, relative_l2_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "generated" / "stationary_dataset.npz"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne un baseline PolynomialFeatures + Ridge."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0],
    )
    return parser.parse_args()


def build_model(degree: int, alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "polynomial_features",
                PolynomialFeatures(degree=degree, include_bias=False),
            ),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def plot_predictions(
    references: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    selected_indices = np.array([438, 88, 653, 772])
    selected_indices = selected_indices[selected_indices < len(references)]

    figure, axes = plt.subplots(
        len(selected_indices),
        3,
        figsize=(12, 3.2 * len(selected_indices)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row, sample_index in enumerate(selected_indices):
        reference = np.squeeze(references[sample_index])
        prediction = np.squeeze(predictions[sample_index])
        error = np.abs(prediction - reference)
        common_min = min(float(reference.min()), float(prediction.min()))
        common_max = max(float(reference.max()), float(prediction.max()))

        reference_image = axes[row, 0].imshow(
            reference,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="inferno",
            vmin=common_min,
            vmax=common_max,
        )
        axes[row, 1].imshow(
            prediction,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="inferno",
            vmin=common_min,
            vmax=common_max,
        )
        error_image = axes[row, 2].imshow(
            error,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="magma",
        )

        axes[row, 0].set_title(f"Référence — exemple {sample_index}")
        axes[row, 1].set_title("Ridge polynomial")
        axes[row, 2].set_title("Erreur absolue")
        figure.colorbar(reference_image, ax=axes[row, :2], shrink=0.75)
        figure.colorbar(error_image, ax=axes[row, 2], shrink=0.75)

        for axis in axes[row]:
            axis.set_xlabel("x")
            axis.set_ylabel("y")

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    dataset_path = arguments.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    run_id = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + "_polynomial_ridge"
    )
    run_directory = PROJECT_ROOT / "artifacts" / "runs" / run_id
    evaluation_directory = run_directory / "evaluation"
    evaluation_directory.mkdir(parents=True, exist_ok=False)

    parameters, temperatures = load_dataset(dataset_path)
    splits = split_dataset(
        parameters,
        temperatures,
        train_fraction=0.8,
        validation_fraction=0.1,
        seed=arguments.seed,
    )

    train_targets = splits.train_temperatures.reshape(
        len(splits.train_temperatures), -1
    )
    validation_targets = splits.validation_temperatures.reshape(
        len(splits.validation_temperatures), -1
    )
    splits.test_temperatures.reshape(
        len(splits.test_temperatures), -1
    )

    candidates = []
    print("Sélection de la régularisation Ridge...")
    for alpha in arguments.alphas:
        candidate = build_model(arguments.degree, alpha)
        candidate.fit(splits.train_parameters, train_targets)
        validation_predictions = candidate.predict(
            splits.validation_parameters
        )
        validation_l2 = relative_l2_error(
            validation_targets,
            validation_predictions,
        )
        candidates.append((validation_l2, alpha, candidate))
        print(f"alpha={alpha:g} — L2 validation={validation_l2:.6f}")

    best_validation_l2, best_alpha, model = min(
        candidates,
        key=lambda item: item[0],
    )

    # Réentraînement final sur train + validation avec l'alpha sélectionné.
    final_parameters = np.concatenate(
        [splits.train_parameters, splits.validation_parameters], axis=0
    )
    final_targets = np.concatenate(
        [train_targets, validation_targets], axis=0
    )
    model = build_model(arguments.degree, best_alpha)
    training_start = time.perf_counter()
    model.fit(final_parameters, final_targets)
    training_duration = time.perf_counter() - training_start

    # Préchauffage et mesure médiane de l'inférence par lot.
    model.predict(splits.test_parameters[:64])
    inference_durations = []
    for _ in range(5):
        inference_start = time.perf_counter()
        flat_predictions = model.predict(splits.test_parameters)
        inference_durations.append(time.perf_counter() - inference_start)
    inference_duration = float(np.median(inference_durations))

    predictions = flat_predictions.reshape(
        splits.test_temperatures.shape
    ).astype(np.float32)
    metrics = compute_all_metrics(splits.test_temperatures, predictions)
    metrics.update(
        {
            "validation_relative_l2": float(best_validation_l2),
            "training_time_seconds": float(training_duration),
            "total_inference_time_seconds": inference_duration,
            "mean_inference_time_seconds": inference_duration
            / len(predictions),
            "simulations_per_second": len(predictions)
            / inference_duration,
        }
    )

    configuration = {
        "run_id": run_id,
        "model_name": "polynomial_ridge",
        "polynomial_degree": arguments.degree,
        "selected_alpha": best_alpha,
        "candidate_alphas": arguments.alphas,
        "dataset_path": str(dataset_path.resolve()),
        "number_of_samples": len(parameters),
        "number_of_train_samples": len(splits.train_parameters),
        "number_of_validation_samples": len(splits.validation_parameters),
        "number_of_test_samples": len(splits.test_parameters),
        "split_seed": arguments.seed,
    }

    joblib.dump(model, run_directory / "model.joblib")
    with (run_directory / "config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4, ensure_ascii=False)
    with (evaluation_directory / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "run_id": run_id,
                "model_path": str(run_directory / "model.joblib"),
                "dataset_path": str(dataset_path.resolve()),
                "metrics": metrics,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    np.savez_compressed(
        evaluation_directory / "test_predictions.npz",
        predictions=predictions,
        references=splits.test_temperatures,
    )
    plot_predictions(
        splits.test_temperatures,
        predictions,
        evaluation_directory / "prediction_comparison.png",
    )

    print("\n=== BASELINE SCIKIT-LEARN ===")
    print("Run :", run_id)
    print("Degré polynomial :", arguments.degree)
    print("Alpha sélectionné :", best_alpha)
    for name, value in metrics.items():
        print(f"{name}: {value:.6e}")
    print("Résultats :", run_directory)


if __name__ == "__main__":
    main()
