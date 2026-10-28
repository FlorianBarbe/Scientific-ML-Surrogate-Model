import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from scientific_ml_surrogate_model.physics_stationary import (
    solve_stationary,
)


# racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "stationary_dataset.npz"
)

# paramètres numériques
N = 32
SIGMA = 0.08

# noms des paramètres dans l'ordre utilisé par le solveur
PARAMETER_NAMES = np.array(
    [
        "log_kappa",
        "xc",
        "yc",
        "r",
        "A",
        "xs",
        "ys",
    ]
)

# limites inférieures des paramètres
LOWER_BOUNDS = np.array(
    [
        np.log(0.25),
        0.25,
        0.25,
        0.10,
        0.50,
        0.20,
        0.20,
    ],
    dtype=np.float64,
)

# limites supérieures des paramètres
UPPER_BOUNDS = np.array(
    [
        np.log(4.0),
        0.75,
        0.75,
        0.20,
        2.00,
        0.80,
        0.80,
    ],
    dtype=np.float64,
)


def sample_parameters(
    number_of_samples: int,
    seed: int,
) -> np.ndarray:
    """Échantillonne les paramètres avec Latin Hypercube."""

    sampler = qmc.LatinHypercube(
        d=7,
        seed=seed,
    )

    samples_in_unit_cube = sampler.random(
        n=number_of_samples
    )

    parameters = qmc.scale(
        samples_in_unit_cube,
        LOWER_BOUNDS,
        UPPER_BOUNDS,
    )

    return parameters.astype(np.float64)


def extract_interior_temperature(
    temperature: np.ndarray,
) -> np.ndarray:
    """Extrait le champ intérieur 32 x 32."""

    temperature = np.asarray(
        temperature,
        dtype=np.float64,
    )

    # grille complète 34 x 34 avec les bords
    if temperature.shape == (N + 2, N + 2):
        return temperature[1:-1, 1:-1]

    # grille intérieure déjà extraite
    if temperature.shape == (N, N):
        return temperature

    raise ValueError(
        "Le solveur doit renvoyer une grille "
        f"{(N + 2, N + 2)} ou {(N, N)}, "
        f"mais a renvoyé {temperature.shape}."
    )


def save_dataset(
    output_path: Path,
    parameters: np.ndarray,
    temperatures: np.ndarray,
    seed: int,
) -> None:
    """Sauvegarde le dataset et ses métadonnées."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        parameters=parameters.astype(np.float32),
        temperatures=temperatures.astype(np.float32),
        parameter_names=PARAMETER_NAMES,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        grid_size=N,
        sigma=SIGMA,
        seed=seed,
    )


def generate_dataset(
    number_of_samples: int,
    output_path: Path,
    seed: int,
    checkpoint_every: int = 250,
) -> None:
    """Génère les simulations stationnaires."""

    parameters = sample_parameters(
        number_of_samples,
        seed,
    )

    temperatures = np.zeros(
        (
            number_of_samples,
            N,
            N,
            1,
        ),
        dtype=np.float32,
    )

    start_time = time.perf_counter()

    for sample_index, parameter_values in enumerate(
        parameters
    ):
        # résolution du problème stationnaire
        temperature, _, _, residual = solve_stationary(
            parameter_values,
            N=N,
            sigma=SIGMA,
        )

        interior_temperature = (
            extract_interior_temperature(
                temperature
            )
        )

        # vérifications physiques et numériques
        if not np.isfinite(
            interior_temperature
        ).all():
            raise RuntimeError(
                "La simulation "
                f"{sample_index} contient NaN ou Inf."
            )

        if interior_temperature.min() < -1.0e-10:
            raise RuntimeError(
                "La simulation "
                f"{sample_index} contient une "
                "température négative."
            )

        if residual > 1.0e-10:
            raise RuntimeError(
                "Le résidu de la simulation "
                f"{sample_index} est trop élevé : "
                f"{residual:.3e}"
            )

        temperatures[
            sample_index,
            :,
            :,
            0,
        ] = interior_temperature

        completed_samples = sample_index + 1

        # affichage de la progression
        if (
            completed_samples % 50 == 0
            or completed_samples == number_of_samples
        ):
            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            simulations_per_second = (
                completed_samples
                / elapsed_time
            )

            remaining_samples = (
                number_of_samples
                - completed_samples
            )

            remaining_time = (
                remaining_samples
                / simulations_per_second
            )

            print(
                f"{completed_samples:6d}/"
                f"{number_of_samples} simulations"
                f" — {simulations_per_second:.2f} sim/s"
                f" — reste environ "
                f"{remaining_time / 60:.1f} min"
            )

        # sauvegarde intermédiaire
        if (
            checkpoint_every > 0
            and completed_samples % checkpoint_every == 0
        ):
            checkpoint_path = (
                output_path.parent
                / "stationary_dataset_partial.npz"
            )

            save_dataset(
                checkpoint_path,
                parameters[:completed_samples],
                temperatures[:completed_samples],
                seed,
            )

    # sauvegarde finale
    save_dataset(
        output_path,
        parameters,
        temperatures,
        seed,
    )

    total_duration = (
        time.perf_counter()
        - start_time
    )

    print("\n=== DATASET GÉNÉRÉ ===")
    print("Chemin :", output_path)
    print("Paramètres :", parameters.shape)
    print("Températures :", temperatures.shape)
    print(
        "Durée totale :",
        f"{total_duration / 60:.2f} min",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère le dataset du problème "
            "thermique stationnaire."
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Nombre de simulations à générer.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Chemin du fichier NPZ.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autorise le remplacement d'un dataset existant.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.samples < 10:
        raise ValueError(
            "Il faut générer au moins 10 simulations."
        )

    if (
        arguments.output.exists()
        and not arguments.overwrite
    ):
        raise FileExistsError(
            f"{arguments.output} existe déjà. "
            "Ajoute --overwrite pour le remplacer."
        )

    generate_dataset(
        number_of_samples=arguments.samples,
        output_path=arguments.output,
        seed=arguments.seed,
    )