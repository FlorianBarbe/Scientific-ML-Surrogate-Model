import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import tensorflow as tf

from scientific_ml_surrogate_model.physics_stationary import (
    solve_stationary,
)

from scientific_ml_surrogate_model.dataset import (
    denormalize_temperatures,
    load_dataset,
    load_normalization,
    normalize_parameters,
    split_dataset,
)
from scientific_ml_surrogate_model.metrics import (
    compute_all_metrics,
    relative_pde_residual,

)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évalue un run entraîné."
    )

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help=(
            "Dossier complet du run ou identifiant "
            "présent dans artifacts/runs."
        ),
    )

    return parser.parse_args()


arguments = parse_arguments()

# accepte soit artifacts/runs/..., soit uniquement l'identifiant
if arguments.run.is_absolute():
    RUN_DIRECTORY = arguments.run
elif (PROJECT_ROOT / arguments.run).exists():
    RUN_DIRECTORY = PROJECT_ROOT / arguments.run
else:
    RUN_DIRECTORY = (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / arguments.run
    )

if not RUN_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Run introuvable : {RUN_DIRECTORY}"
    )

CONFIG_PATH = RUN_DIRECTORY / "config.json"
NORMALIZATION_PATH = RUN_DIRECTORY / "normalization.npz"

MODEL_PATH = RUN_DIRECTORY / "model.keras"

# permet d'évaluer un entraînement interrompu
if not MODEL_PATH.exists():
    MODEL_PATH = RUN_DIRECTORY / "checkpoint.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Aucun model.keras ou checkpoint.keras "
        f"dans {RUN_DIRECTORY}."
    )

if not NORMALIZATION_PATH.exists():
    raise FileNotFoundError(
        f"Normalisation introuvable : {NORMALIZATION_PATH}"
    )

# configuration du run
if CONFIG_PATH.exists():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)
else:
    configuration = {}

DATASET_PATH = Path(
    configuration.get(
        "dataset_path",
        PROJECT_ROOT
        / "data"
        / "generated"
        / "stationary_dataset.npz",
    )
)

SPLIT_SEED = int(
    configuration.get(
        "split_seed",
        42,
    )
)

TRAIN_FRACTION = float(
    configuration.get(
        "train_fraction",
        0.8,
    )
)

VALIDATION_FRACTION = float(
    configuration.get(
        "validation_fraction",
        0.1,
    )
)

BATCH_SIZE = int(
    configuration.get(
        "batch_size",
        64,
    )
)

OUTPUT_DIRECTORY = RUN_DIRECTORY / "evaluation"

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

# chargement du même dataset que pendant l'entraînement
parameters, temperatures = load_dataset(
    DATASET_PATH
)

splits = split_dataset(
    parameters,
    temperatures,
    train_fraction=TRAIN_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
    seed=SPLIT_SEED,
)

normalization = load_normalization(
    NORMALIZATION_PATH
)

test_parameters = normalize_parameters(
    splits.test_parameters,
    normalization,
)

true_temperatures = splits.test_temperatures

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Run :", RUN_DIRECTORY.name)
print("Modèle chargé :", MODEL_PATH)
print("Dataset :", DATASET_PATH)
print(
    "Nombre de simulations de test :",
    len(test_parameters),
)

# conversion en tenseur pour éviter le surcoût de model.predict
test_tensor = tf.convert_to_tensor(
    test_parameters,
    dtype=tf.float32,
)

# préchauffage du GPU
model(
    test_tensor[:BATCH_SIZE],
    training=False,
).numpy()

# mesure répétée de l'inférence sur tout le test
inference_durations = []

for _ in range(5):
    start_time = time.perf_counter()

    prediction_tensor = model(
        test_tensor,
        training=False,
    )

    normalized_predictions = (
        prediction_tensor.numpy()
    )

    inference_durations.append(
        time.perf_counter() - start_time
    )

# utilisation de la médiane pour limiter le bruit
inference_duration = float(
    np.median(inference_durations)
)

predicted_temperatures = denormalize_temperatures(
    normalized_predictions,
    normalization,
)

# métriques globales
metrics = compute_all_metrics(
    true_temperatures,
    predicted_temperatures,
)

# erreur L2 calculée simulation par simulation
true_flat = true_temperatures.reshape(
    len(true_temperatures),
    -1,
)

predicted_flat = predicted_temperatures.reshape(
    len(predicted_temperatures),
    -1,
)

relative_l2_per_sample = (
    np.linalg.norm(
        predicted_flat - true_flat,
        axis=1,
    )
    / np.maximum(
        np.linalg.norm(
            true_flat,
            axis=1,
        ),
        1.0e-12,
    )
)

metrics["relative_l2_median"] = float(
    np.median(relative_l2_per_sample)
)

metrics["relative_l2_p90"] = float(
    np.percentile(
        relative_l2_per_sample,
        90,
    )
)

metrics["relative_l2_p99"] = float(
    np.percentile(
        relative_l2_per_sample,
        99,
    )
)

metrics["relative_l2_maximum"] = float(
    relative_l2_per_sample.max()
)

# calcul du résidu physique et temps du solveur
pde_residuals = []
reference_pde_residuals = []
solver_durations = []

print("\nCalcul des résidus physiques...")

for sample_index, (physical_parameters, predicted_temperature) in enumerate(
    zip(
        splits.test_parameters,
        predicted_temperatures,
        strict=True,
    )
):
    solver_start = time.perf_counter()

    _, stiffness_matrix, source, _ = solve_stationary(
        physical_parameters,
        N=32,
        sigma=0.08,
    )

    solver_durations.append(
        time.perf_counter() - solver_start
    )

    # résidu physique de la prédiction du réseau
    pde_residuals.append(
        relative_pde_residual(
            stiffness_matrix,
            predicted_temperature,
            source,
        )
    )

    # résidu physique de la solution de référence
    reference_temperature = np.asarray(
        splits.test_temperatures[sample_index]
    ).squeeze()

    reference_pde_residuals.append(
        relative_pde_residual(
            stiffness_matrix,
            reference_temperature,
            source,
        )
    )

    if (
        (sample_index + 1) % 100 == 0
        or sample_index + 1 == len(test_parameters)
    ):
        print(
            f"{sample_index + 1}/"
            f"{len(test_parameters)}"
        )

pde_residuals = np.asarray(
    pde_residuals,
    dtype=np.float64,
)

reference_pde_residuals = np.asarray(
    reference_pde_residuals,
    dtype=np.float64,
)

solver_durations = np.asarray(
    solver_durations,
    dtype=np.float64,
)

print("\n=== VÉRIFICATION DU RÉSIDU ===")
print(
    "Résidu moyen des références :",
    reference_pde_residuals.mean(),
)
print(
    "Résidu maximal des références :",
    reference_pde_residuals.max(),
)

metrics["reference_pde_residual_mean"] = float(
    reference_pde_residuals.mean()
)

metrics["reference_pde_residual_maximum"] = float(
    reference_pde_residuals.max()
)
mean_solver_time = float(
    solver_durations.mean()
)

mean_inference_time = (
    inference_duration
    / len(test_parameters)
)

metrics["pde_residual_mean"] = float(
    pde_residuals.mean()
)

metrics["pde_residual_median"] = float(
    np.median(pde_residuals)
)

metrics["pde_residual_p90"] = float(
    np.percentile(
        pde_residuals,
        90,
    )
)

metrics["pde_residual_maximum"] = float(
    pde_residuals.max()
)

metrics["total_inference_time_seconds"] = (
    inference_duration
)

metrics["mean_inference_time_seconds"] = (
    mean_inference_time
)

metrics["simulations_per_second"] = (
    len(test_parameters)
    / inference_duration
)

metrics["mean_solver_time_seconds"] = (
    mean_solver_time
)

metrics["measured_speedup"] = (
    mean_solver_time
    / mean_inference_time
)

print("\n=== ÉVALUATION SCIENTIFIQUE ===")

for name, value in metrics.items():
    print(f"{name}: {value:.6e}")

# sauvegarde des métriques
metrics_with_metadata = {
    "run_id": RUN_DIRECTORY.name,
    "model_path": str(MODEL_PATH),
    "dataset_path": str(DATASET_PATH),
    "metrics": metrics,
}

with (
    OUTPUT_DIRECTORY
    / "metrics.json"
).open("w", encoding="utf-8") as file:
    json.dump(
        metrics_with_metadata,
        file,
        indent=4,
    )

# sauvegarde des prédictions
np.savez_compressed(
    OUTPUT_DIRECTORY
    / "test_predictions.npz",
    parameters=splits.test_parameters,
    true_temperatures=true_temperatures,
    predicted_temperatures=predicted_temperatures,
    relative_l2_per_sample=relative_l2_per_sample,
    pde_residuals=pde_residuals,
    solver_durations=solver_durations,
)

number_of_examples = min(
    4,
    len(test_parameters),
)

selected_indices = np.argsort(
    relative_l2_per_sample
)[-number_of_examples:][::-1]

print("\nPires simulations :", selected_indices)

for sample_index in selected_indices:
    print(
        f"Exemple {sample_index}: "
        f"L2={relative_l2_per_sample[sample_index]:.4f}, "
        f"PDE={pde_residuals[sample_index]:.4e}"
    )

selected_errors = [
    np.abs(
        np.squeeze(
            predicted_temperatures[index]
        )
        - np.squeeze(
            true_temperatures[index]
        )
    )
    for index in selected_indices
]

common_error_maximum = max(
    error.max()
    for error in selected_errors
)

figure, axes = plt.subplots(
    number_of_examples,
    3,
    figsize=(12, 4 * number_of_examples),
    constrained_layout=True,
)

if number_of_examples == 1:
    axes = axes[np.newaxis, :]

for row, sample_index in enumerate(
    selected_indices
):
    true_field = np.squeeze(
        true_temperatures[sample_index]
    )

    predicted_field = np.squeeze(
        predicted_temperatures[sample_index]
    )

    absolute_error = np.abs(
        predicted_field
        - true_field
    )

    minimum_temperature = min(
        true_field.min(),
        predicted_field.min(),
    )

    maximum_temperature = max(
        true_field.max(),
        predicted_field.max(),
    )

    true_image = axes[row, 0].imshow(
        true_field,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="inferno",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
    )

    axes[row, 0].set_title(
        f"Référence — exemple {sample_index}"
    )

    predicted_image = axes[row, 1].imshow(
        predicted_field,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="inferno",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
    )

    axes[row, 1].set_title(
        "Prédiction du modèle"
    )

    error_image = axes[row, 2].imshow(
        absolute_error,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=common_error_maximum,
    )

    axes[row, 2].set_title(
        "Erreur absolue"
    )

    figure.colorbar(
        true_image,
        ax=axes[row, 0],
        label="Température",
    )

    figure.colorbar(
        predicted_image,
        ax=axes[row, 1],
        label="Température",
    )

    figure.colorbar(
        error_image,
        ax=axes[row, 2],
        label="Erreur",
    )

    for column in range(3):
        axes[row, column].set_xlabel("x")
        axes[row, column].set_ylabel("y")

comparison_path = (
    OUTPUT_DIRECTORY
    / "prediction_comparison.png"
)

figure.savefig(
    comparison_path,
    dpi=200,
)

plt.show()

print("\nRésultats sauvegardés dans :")
print(OUTPUT_DIRECTORY)