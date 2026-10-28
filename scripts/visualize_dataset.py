from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "stationary_dataset.npz"
)

# chargement du dataset
with np.load(DATASET_PATH) as data:
    parameters = data["parameters"]
    temperatures = data["temperatures"]

print("Forme des paramètres :", parameters.shape)
print("Forme des températures :", temperatures.shape)
print("Température minimale :", temperatures.min())
print("Température maximale :", temperatures.max())

# sélection aléatoire de plusieurs simulations
random_generator = np.random.default_rng(42)

number_of_examples = min(
    6,
    len(parameters),
)

selected_indices = random_generator.choice(
    len(parameters),
    size=number_of_examples,
    replace=False,
)

figure, axes = plt.subplots(
    2,
    3,
    figsize=(14, 9),
    constrained_layout=True,
)

axes = axes.ravel()

# même échelle de couleur pour toutes les simulations
minimum_temperature = temperatures.min()
maximum_temperature = temperatures.max()

for axis, sample_index in zip(
    axes,
    selected_indices,
):
    params = parameters[sample_index]

    log_kappa, xc, yc, r, A, xs, ys = params
    kappa = np.exp(log_kappa)

    temperature = np.squeeze(
        temperatures[sample_index]
    )

    image = axis.imshow(
        temperature,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="inferno",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
    )

    # centre de la source
    axis.scatter(
        xs,
        ys,
        color="cyan",
        marker="x",
        s=60,
        label="Source",
    )

    # inclusion conductrice
    axis.add_patch(
        plt.Circle(
            (xc, yc),
            r,
            fill=False,
            color="white",
            linewidth=1.5,
        )
    )

    axis.set_title(
        f"Exemple {sample_index}\n"
        f"κ={kappa:.2f}, A={A:.2f}, r={r:.2f}"
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")

# suppression des axes inutilisés
for axis in axes[number_of_examples:]:
    axis.set_visible(False)

figure.colorbar(
    image,
    ax=axes.tolist(),
    label="Température",
    shrink=0.8,
)

plt.show()