"""Génère les figures versionnées utilisées dans le README."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scientific_ml_surrogate_model.physics_stationary import solve_stationary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "stationary_dataset.npz"

MLP_RUN = PROJECT_ROOT / "artifacts" / "runs" / "20261028-162521_mlp_baseline"
ENCODER_RUN = PROJECT_ROOT / "artifacts" / "runs" / "20261028-162120_encoder_decoder"
ENCODER_HISTORY_RUN = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "20261028-161846_encoder_decoder_existing"
)
RIDGE_RUN = PROJECT_ROOT / "artifacts" / "runs" / "20261028-164859_polynomial_ridge"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATASET_PATH) as dataset:
        return dataset["parameters"], dataset["temperatures"]


def add_geometry(axis: plt.Axes, parameters: np.ndarray) -> None:
    _, xc, yc, radius, _, xs, ys = parameters
    axis.add_patch(
        plt.Circle(
            (xc, yc),
            radius,
            fill=False,
            color="white",
            linewidth=1.3,
        )
    )
    axis.scatter(xs, ys, color="cyan", marker="x", s=45)


def generate_solver_overview(parameters: np.ndarray) -> None:
    sample_index = 438
    sample_parameters = parameters[sample_index].astype(np.float64)
    log_kappa, xc, yc, radius, amplitude, xs, ys = sample_parameters
    temperature, _, _, algebraic_residual = solve_stationary(sample_parameters)

    grid = np.linspace(0.0, 1.0, 34)
    x_grid, y_grid = np.meshgrid(grid, grid)
    conductivity = np.ones_like(x_grid)
    inclusion = (x_grid - xc) ** 2 + (y_grid - yc) ** 2 <= radius**2
    conductivity[inclusion] = np.exp(log_kappa)
    source = amplitude * np.exp(
        -(
            (x_grid[1:-1, 1:-1] - xs) ** 2
            + (y_grid[1:-1, 1:-1] - ys) ** 2
        )
        / (2.0 * 0.08**2)
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    panels = [
        (conductivity, "Conductivité $k(x,y)$", "viridis"),
        (source, "Source gaussienne $q(x,y)$", "magma"),
        (temperature[1:-1, 1:-1], "Température stationnaire $T(x,y)$", "inferno"),
    ]

    for axis, (field, title, colormap) in zip(axes, panels, strict=True):
        image = axis.imshow(
            field,
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap=colormap,
        )
        add_geometry(axis, sample_parameters)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.82)

    figure.suptitle(
        "Solveur SciPy de référence — "
        f"résidu algébrique relatif : {algebraic_residual:.2e}",
        fontsize=13,
    )
    figure.savefig(ASSET_DIRECTORY / "solver_overview.png", dpi=180)
    plt.close(figure)


def generate_dataset_overview(
    parameters: np.ndarray,
    temperatures: np.ndarray,
) -> None:
    selected_indices = [438, 88, 653, 772]
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), constrained_layout=True)

    for axis, sample_index in zip(axes.ravel(), selected_indices, strict=True):
        sample_parameters = parameters[sample_index]
        log_kappa, _, _, radius, amplitude, _, _ = sample_parameters
        temperature = np.squeeze(temperatures[sample_index])
        image = axis.imshow(
            temperature,
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="inferno",
        )
        add_geometry(axis, sample_parameters)
        axis.set_title(
            f"Simulation {sample_index} — "
            f"κ={np.exp(log_kappa):.2f}, A={amplitude:.2f}, r={radius:.2f}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.82, label="Température")

    figure.suptitle("Exemples du dataset de 10 000 simulations", fontsize=14)
    figure.savefig(ASSET_DIRECTORY / "dataset_examples.png", dpi=180)
    plt.close(figure)


def generate_learning_curves() -> None:
    mlp_history = pd.read_csv(MLP_RUN / "training_history.csv")
    encoder_history = pd.read_csv(ENCODER_HISTORY_RUN / "training_history.csv")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    histories = [
        (mlp_history, "MLP — 200 époques"),
        (encoder_history, "Encodeur-décodeur — run archivé, 158 époques"),
    ]

    for axis, (history, title) in zip(axes, histories, strict=True):
        epochs = history["epoch"] + 1
        axis.semilogy(epochs, history["loss"], label="Entraînement")
        axis.semilogy(epochs, history["val_loss"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Époque")
        axis.set_ylabel("MSE normalisée")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Courbes d’apprentissage", fontsize=14)
    figure.savefig(ASSET_DIRECTORY / "learning_curves.png", dpi=180)
    plt.close(figure)


def read_metrics(run_directory: Path) -> dict[str, float]:
    with (run_directory / "evaluation" / "metrics.json").open(
        encoding="utf-8"
    ) as file:
        payload = json.load(file)
    return payload.get("metrics", payload)


def generate_model_comparison() -> None:
    labels = ["Ridge\npolynomial", "Encodeur-\ndécodeur*", "MLP"]
    runs = [RIDGE_RUN, ENCODER_RUN, MLP_RUN]
    metrics = [read_metrics(run) for run in runs]
    relative_l2 = [100.0 * item["relative_l2"] for item in metrics]
    r_squared = [item["r2"] for item in metrics]
    colors = ["#94a3b8", "#38bdf8", "#f97316"]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bars = axes[0].bar(labels, relative_l2, color=colors)
    axes[0].bar_label(bars, fmt="%.2f %%", padding=3)
    axes[0].set_ylabel("Erreur L2 relative")
    axes[0].set_title("Précision — plus bas est meilleur")
    axes[0].grid(axis="y", alpha=0.25)

    bars = axes[1].bar(labels, r_squared, color=colors)
    axes[1].bar_label(bars, fmt="%.4f", padding=3)
    axes[1].set_ylim(0.94, 1.002)
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("Variance expliquée — plus haut est meilleur")
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("Comparaison sur le même jeu de test (1 000 simulations)")
    figure.savefig(ASSET_DIRECTORY / "model_comparison.png", dpi=180)
    plt.close(figure)


def copy_evaluation_figures() -> None:
    copies = {
        MLP_RUN / "evaluation" / "prediction_comparison.png": (
            ASSET_DIRECTORY / "mlp_worst_case_predictions.png"
        ),
        ENCODER_RUN / "evaluation" / "prediction_comparison.png": (
            ASSET_DIRECTORY / "encoder_decoder_predictions.png"
        ),
        RIDGE_RUN / "evaluation" / "prediction_comparison.png": (
            ASSET_DIRECTORY / "ridge_predictions.png"
        ),
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)


def main() -> None:
    ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parameters, temperatures = load_data()
    generate_solver_overview(parameters)
    generate_dataset_overview(parameters, temperatures)
    generate_learning_curves()
    generate_model_comparison()
    copy_evaluation_figures()
    print(f"Figures README générées dans : {ASSET_DIRECTORY}")


if __name__ == "__main__":
    main()
