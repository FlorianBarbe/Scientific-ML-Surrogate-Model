import numpy as np
import scipy.sparse as sp


def _prepare_batches(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transforme les champs en tableaux (batch, pixels)."""

    true_values = np.asarray(
        true_values,
        dtype=np.float64,
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=np.float64,
    )

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "Les températures réelles et prédites "
            "doivent avoir la même forme."
        )

    # Cas d'un seul champ 32 x 32
    if true_values.ndim == 2:
        true_values = true_values[np.newaxis, ...]
        predicted_values = predicted_values[
            np.newaxis, ...
        ]

    # Cas d'un seul champ 32 x 32 x 1
    elif (
        true_values.ndim == 3
        and true_values.shape[-1] == 1
    ):
        true_values = true_values[np.newaxis, ...]
        predicted_values = predicted_values[
            np.newaxis, ...
        ]

    number_of_samples = true_values.shape[0]

    true_values = true_values.reshape(
        number_of_samples,
        -1,
    )

    predicted_values = predicted_values.reshape(
        number_of_samples,
        -1,
    )

    return true_values, predicted_values


def mean_absolute_error(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Erreur absolue moyenne."""

    true_values, predicted_values = _prepare_batches(
        true_values,
        predicted_values,
    )

    errors = np.abs(
        predicted_values - true_values
    )

    sample_errors = errors.mean(axis=1)

    return float(sample_errors.mean())


def root_mean_squared_error(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Racine de l'erreur quadratique moyenne."""

    true_values, predicted_values = _prepare_batches(
        true_values,
        predicted_values,
    )

    squared_errors = (
        predicted_values - true_values
    ) ** 2

    sample_errors = np.sqrt(
        squared_errors.mean(axis=1)
    )

    return float(sample_errors.mean())


def relative_l2_error(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Erreur L2 relative moyenne."""

    true_values, predicted_values = _prepare_batches(
        true_values,
        predicted_values,
    )

    error_norms = np.linalg.norm(
        predicted_values - true_values,
        axis=1,
    )

    true_norms = np.linalg.norm(
        true_values,
        axis=1,
    )

    relative_errors = error_norms / np.maximum(
        true_norms,
        epsilon,
    )

    return float(relative_errors.mean())


def maximum_temperature_error(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Erreur moyenne sur la température maximale."""

    true_values, predicted_values = _prepare_batches(
        true_values,
        predicted_values,
    )

    true_maximums = true_values.max(axis=1)

    predicted_maximums = predicted_values.max(
        axis=1
    )

    errors = np.abs(
        predicted_maximums - true_maximums
    )

    return float(errors.mean())


def maximum_absolute_error(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Plus grande erreur locale observée."""

    true_values, predicted_values = _prepare_batches(
        true_values,
        predicted_values,
    )

    return float(
        np.max(
            np.abs(
                predicted_values - true_values
            )
        )
    )


def coefficient_of_determination(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Coefficient de détermination R² global."""

    true_values, predicted_values = _prepare_batches(
        true_values,
        predicted_values,
    )

    true_flat = true_values.ravel()
    predicted_flat = predicted_values.ravel()

    residual_sum = np.sum(
        (true_flat - predicted_flat) ** 2
    )

    total_sum = np.sum(
        (
            true_flat
            - true_flat.mean()
        ) ** 2
    )

    return float(
        1.0
        - residual_sum
        / max(total_sum, epsilon)
    )


def relative_pde_residual(
    stiffness_matrix: sp.spmatrix,
    predicted_temperature: np.ndarray,
    source: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Calcule ||K T - q||₂ / ||q||₂ pour une simulation."""

    predicted_temperature = np.asarray(
        predicted_temperature,
        dtype=np.float64,
    ).reshape(-1)

    source = np.asarray(
        source,
        dtype=np.float64,
    ).reshape(-1)

    residual = (
        stiffness_matrix
        @ predicted_temperature
        - source
    )

    return float(
        np.linalg.norm(residual)
        / max(
            np.linalg.norm(source),
            epsilon,
        )
    )


def compute_all_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict[str, float]:
    """Calcule les principales métriques du projet."""

    return {
        "mae": mean_absolute_error(
            true_values,
            predicted_values,
        ),
        "rmse": root_mean_squared_error(
            true_values,
            predicted_values,
        ),
        "relative_l2": relative_l2_error(
            true_values,
            predicted_values,
        ),
        "maximum_temperature_error":
            maximum_temperature_error(
                true_values,
                predicted_values,
            ),
        "maximum_absolute_error":
            maximum_absolute_error(
                true_values,
                predicted_values,
            ),
        "r2": coefficient_of_determination(
            true_values,
            predicted_values,
        ),
    }