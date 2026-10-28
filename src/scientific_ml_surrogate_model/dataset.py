from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class NormalizationStats:
    parameter_mean: np.ndarray
    parameter_std: np.ndarray
    temperature_mean: float
    temperature_std: float


@dataclass(frozen=True)
class DatasetSplits:
    train_parameters: np.ndarray
    train_temperatures: np.ndarray

    validation_parameters: np.ndarray
    validation_temperatures: np.ndarray

    test_parameters: np.ndarray
    test_temperatures: np.ndarray


def load_dataset(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge les paramètres et les champs de température."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Le dataset n'existe pas : {path}"
        )

    with np.load(path) as data:
        if "parameters" not in data:
            raise KeyError(
                "Le dataset doit contenir 'parameters'."
            )

        if "temperatures" not in data:
            raise KeyError(
                "Le dataset doit contenir 'temperatures'."
            )

        parameters = data["parameters"].astype(
            np.float32
        )

        temperatures = data["temperatures"].astype(
            np.float32
        )

    if parameters.ndim != 2:
        raise ValueError(
            "parameters doit avoir la forme "
            "(nombre_simulations, 7)."
        )

    if parameters.shape[1] != 7:
        raise ValueError(
            "Chaque simulation doit contenir 7 paramètres."
        )

    # Ajout du canal nécessaire aux couches Conv2D
    if temperatures.ndim == 3:
        temperatures = temperatures[..., np.newaxis]

    if temperatures.ndim != 4:
        raise ValueError(
            "temperatures doit avoir la forme "
            "(nombre_simulations, 32, 32, 1)."
        )

    if temperatures.shape[1:] != (32, 32, 1):
        raise ValueError(
            "Chaque température doit avoir la forme "
            "(32, 32, 1)."
        )

    if len(parameters) != len(temperatures):
        raise ValueError(
            "Le nombre de paramètres et de températures "
            "doit être identique."
        )

    if not np.isfinite(parameters).all():
        raise ValueError(
            "Les paramètres contiennent NaN ou Inf."
        )

    if not np.isfinite(temperatures).all():
        raise ValueError(
            "Les températures contiennent NaN ou Inf."
        )

    return parameters, temperatures


def split_dataset(
    parameters: np.ndarray,
    temperatures: np.ndarray,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    """Sépare aléatoirement les données."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError(
            "Les proportions doivent être positives."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "Il faut conserver des données pour le test."
        )

    number_of_samples = len(parameters)

    if number_of_samples < 10:
        raise ValueError(
            "Le dataset doit contenir au moins 10 simulations."
        )

    random_generator = np.random.default_rng(seed)
    indices = random_generator.permutation(
        number_of_samples
    )

    number_of_train_samples = int(
        train_fraction * number_of_samples
    )

    number_of_validation_samples = int(
        validation_fraction * number_of_samples
    )

    train_end = number_of_train_samples

    validation_end = (
        number_of_train_samples
        + number_of_validation_samples
    )

    train_indices = indices[:train_end]

    validation_indices = indices[
        train_end:validation_end
    ]

    test_indices = indices[validation_end:]

    return DatasetSplits(
        train_parameters=parameters[train_indices],
        train_temperatures=temperatures[train_indices],

        validation_parameters=parameters[
            validation_indices
        ],
        validation_temperatures=temperatures[
            validation_indices
        ],

        test_parameters=parameters[test_indices],
        test_temperatures=temperatures[test_indices],
    )


def fit_normalization(
    train_parameters: np.ndarray,
    train_temperatures: np.ndarray,
) -> NormalizationStats:
    """Calcule les statistiques uniquement sur le train."""

    parameter_mean = train_parameters.mean(
        axis=0
    )

    parameter_std = train_parameters.std(
        axis=0
    )

    # Évite une division par zéro
    parameter_std = np.maximum(
        parameter_std,
        1.0e-8,
    )

    temperature_mean = float(
        train_temperatures.mean()
    )

    temperature_std = float(
        train_temperatures.std()
    )

    temperature_std = max(
        temperature_std,
        1.0e-8,
    )

    return NormalizationStats(
        parameter_mean=parameter_mean,
        parameter_std=parameter_std,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
    )


def normalize_parameters(
    parameters: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    normalized = (
        parameters - stats.parameter_mean
    ) / stats.parameter_std

    return normalized.astype(np.float32)


def normalize_temperatures(
    temperatures: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    normalized = (
        temperatures - stats.temperature_mean
    ) / stats.temperature_std

    return normalized.astype(np.float32)


def denormalize_temperatures(
    temperatures: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    denormalized = (
        temperatures * stats.temperature_std
        + stats.temperature_mean
    )

    return denormalized.astype(np.float32)


def normalize_splits(
    splits: DatasetSplits,
    stats: NormalizationStats,
) -> DatasetSplits:
    """Normalise toutes les séparations avec le train."""

    return DatasetSplits(
        train_parameters=normalize_parameters(
            splits.train_parameters,
            stats,
        ),
        train_temperatures=normalize_temperatures(
            splits.train_temperatures,
            stats,
        ),

        validation_parameters=normalize_parameters(
            splits.validation_parameters,
            stats,
        ),
        validation_temperatures=normalize_temperatures(
            splits.validation_temperatures,
            stats,
        ),

        test_parameters=normalize_parameters(
            splits.test_parameters,
            stats,
        ),
        test_temperatures=normalize_temperatures(
            splits.test_temperatures,
            stats,
        ),
    )


def make_tensorflow_dataset(
    parameters: np.ndarray,
    temperatures: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Crée un pipeline de données TensorFlow."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            parameters,
            temperatures,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(parameters),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def save_normalization(
    path: str | Path,
    stats: NormalizationStats,
) -> None:
    """Sauvegarde les statistiques de normalisation."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        path,
        parameter_mean=stats.parameter_mean,
        parameter_std=stats.parameter_std,
        temperature_mean=stats.temperature_mean,
        temperature_std=stats.temperature_std,
    )


def load_normalization(
    path: str | Path,
) -> NormalizationStats:
    """Recharge les statistiques de normalisation."""

    with np.load(path) as data:
        return NormalizationStats(
            parameter_mean=data[
                "parameter_mean"
            ],
            parameter_std=data[
                "parameter_std"
            ],
            temperature_mean=float(
                data["temperature_mean"]
            ),
            temperature_std=float(
                data["temperature_std"]
            ),
        )