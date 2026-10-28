import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from scientific_ml_surrogate_model.dataset import (
    fit_normalization,
    load_dataset,
    make_tensorflow_dataset,
    normalize_splits,
    save_normalization,
    split_dataset,
)
from scientific_ml_surrogate_model.models import (
    build_encoder_decoder,
    build_mlp_baseline,
    compile_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "stationary_dataset.npz"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne un surrogate thermique."
    )

    parser.add_argument(
        "--model",
        choices=[
            "encoder_decoder",
            "mlp_baseline",
        ],
        default="encoder_decoder",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


arguments = parse_arguments()

DATASET_PATH = arguments.dataset

if not DATASET_PATH.is_absolute():
    DATASET_PATH = PROJECT_ROOT / DATASET_PATH

MODEL_NAME = re.sub(
    r"[^a-zA-Z0-9_-]+",
    "-",
    arguments.model,
).strip("-")

RUN_ID = (
    datetime.now().strftime("%Y%m%d-%H%M%S")
    + "_"
    + MODEL_NAME
)

RUN_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / RUN_ID
)

RUN_DIRECTORY.mkdir(
    parents=True,
    exist_ok=False,
)

MODEL_PATH = RUN_DIRECTORY / "model.keras"
CHECKPOINT_PATH = RUN_DIRECTORY / "checkpoint.keras"
NORMALIZATION_PATH = RUN_DIRECTORY / "normalization.npz"
CONFIG_PATH = RUN_DIRECTORY / "config.json"
HISTORY_CSV_PATH = RUN_DIRECTORY / "training_history.csv"
HISTORY_NPZ_PATH = RUN_DIRECTORY / "training_history.npz"
HISTORY_IMAGE_PATH = RUN_DIRECTORY / "training_history.png"
SUMMARY_PATH = RUN_DIRECTORY / "model_summary.txt"

SEED = arguments.seed
BATCH_SIZE = arguments.batch_size
NUMBER_OF_EPOCHS = arguments.epochs
LEARNING_RATE = arguments.learning_rate

keras.utils.set_random_seed(SEED)

# configuration du GPU
gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(
            gpu,
            True,
        )
    except RuntimeError:
        pass

print("TensorFlow :", tf.__version__)
print("GPU détectés :", gpus)
print("Run :", RUN_ID)
print("Dossier :", RUN_DIRECTORY)

# chargement des données
parameters, temperatures = load_dataset(
    DATASET_PATH
)

splits = split_dataset(
    parameters,
    temperatures,
    train_fraction=0.8,
    validation_fraction=0.1,
    seed=SEED,
)

print("Dataset :", DATASET_PATH)
print("Paramètres :", parameters.shape)
print("Températures :", temperatures.shape)
print("Entraînement :", len(splits.train_parameters))
print("Validation :", len(splits.validation_parameters))
print("Test :", len(splits.test_parameters))

# normalisation calculée uniquement sur le train
normalization = fit_normalization(
    splits.train_parameters,
    splits.train_temperatures,
)

save_normalization(
    NORMALIZATION_PATH,
    normalization,
)

normalized_splits = normalize_splits(
    splits,
    normalization,
)

train_dataset = make_tensorflow_dataset(
    normalized_splits.train_parameters,
    normalized_splits.train_temperatures,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

validation_dataset = make_tensorflow_dataset(
    normalized_splits.validation_parameters,
    normalized_splits.validation_temperatures,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# choix du modèle
if arguments.model == "encoder_decoder":
    model = build_encoder_decoder(
        number_of_parameters=7,
    )
else:
    model = build_mlp_baseline(
        number_of_parameters=7,
    )

compile_model(
    model,
    learning_rate=LEARNING_RATE,
)

model.summary()

# sauvegarde de l'architecture
with SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    model.summary(
        print_fn=lambda line: file.write(
            line + "\n"
        )
    )

# sauvegarde de la configuration du run
configuration = {
    "run_id": RUN_ID,
    "model_name": arguments.model,
    "dataset_path": str(DATASET_PATH.resolve()),
    "number_of_samples": len(parameters),
    "number_of_train_samples": len(
        splits.train_parameters
    ),
    "number_of_validation_samples": len(
        splits.validation_parameters
    ),
    "number_of_test_samples": len(
        splits.test_parameters
    ),
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "split_seed": SEED,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": NUMBER_OF_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "grid_size": 32,
    "tensorflow_version": tf.__version__,
    "gpu_devices": [
        gpu.name for gpu in gpus
    ],
}

with CONFIG_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        configuration,
        file,
        indent=4,
    )

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1.0e-6,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        HISTORY_CSV_PATH
    ),
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=NUMBER_OF_EPOCHS,
    callbacks=callbacks,
)

# sauvegarde du meilleur modèle restauré
model.save(MODEL_PATH)

np.savez(
    HISTORY_NPZ_PATH,
    **history.history,
)

# courbes d'entraînement
figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

axes[0].plot(
    history.history["loss"],
    label="Entraînement",
)

axes[0].plot(
    history.history["val_loss"],
    label="Validation",
)

axes[0].set_xlabel("Époque")
axes[0].set_ylabel("MSE normalisée")
axes[0].set_title("Fonction de coût")
axes[0].set_yscale("log")
axes[0].grid()
axes[0].legend()

axes[1].plot(
    history.history["mae"],
    label="Entraînement",
)

axes[1].plot(
    history.history["val_mae"],
    label="Validation",
)

axes[1].set_xlabel("Époque")
axes[1].set_ylabel("MAE normalisée")
axes[1].set_title("Erreur absolue moyenne")
axes[1].set_yscale("log")
axes[1].grid()
axes[1].legend()

figure.tight_layout()

figure.savefig(
    HISTORY_IMAGE_PATH,
    dpi=200,
)

plt.show()

print("\n=== ENTRAÎNEMENT TERMINÉ ===")
print("Run :", RUN_ID)
print("Résultats :", RUN_DIRECTORY)
print("\nCommande d'évaluation :")
print(
    "python scripts/evaluate.py "
    f"--run artifacts/runs/{RUN_ID}"
)
