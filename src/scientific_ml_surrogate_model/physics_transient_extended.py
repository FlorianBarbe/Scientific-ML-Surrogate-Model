# les bibliothèques
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# dimensions physiques de la plaque en mètres
Lx, Ly = 0.10, 0.10
thickness = 0.005

# grille de 32 x 32 cellules
N = 32
dx, dy = Lx / N, Ly / N
cell_volume = dx * dy * thickness

# centres des cellules
x = (np.arange(N) + 0.5) * dx
y = (np.arange(N) + 0.5) * dy
X, Y = np.meshgrid(x, y)

# propriétés du matériau principal
k_background = 205.0       # conductivité en W/(m.K)
rho_background = 2700.0    # masse volumique en kg/m³
cp_background = 900.0      # capacité thermique en J/(kg.K)

# propriétés du matériau de l'inclusion
k_inclusion = 16.0
rho_inclusion = 8000.0
cp_inclusion = 500.0

# géométrie de l'inclusion
xc, yc = 0.04, 0.05
radius = 0.015

# paramètres de la source thermique
xs, ys = 0.065, 0.055
sigma = 0.006
q_peak = 1.0e8             # puissance volumique maximale en W/m³
heating_duration = 5.0     # durée d'activation de la source en secondes

# échange thermique avec l'air
T_ambient = 20.0            # température ambiante en °C
h_edges = 10.0              # convection sur les bords en W/(m².K)
h_surfaces = 8.0            # convection sur les faces en W/(m².K)

# paramètres temporels
dt = 0.1
final_time = 30.0
save_every = 2
number_of_steps = int(final_time / dt)

# position de l'inclusion
inside_inclusion = (
    (X - xc) ** 2
    + (Y - yc) ** 2
    <= radius**2
)

# champs physiques des matériaux
conductivity = np.where(
    inside_inclusion,
    k_inclusion,
    k_background,
)

density = np.where(
    inside_inclusion,
    rho_inclusion,
    rho_background,
)

specific_heat = np.where(
    inside_inclusion,
    cp_inclusion,
    cp_background,
)

# capacité thermique de chaque cellule
thermal_capacity = (
    density
    * specific_heat
    * cell_volume
).ravel()

# profil spatial de la source gaussienne
source_profile = np.exp(
    -(
        (X - xs) ** 2
        + (Y - ys) ** 2
    )
    / (2.0 * sigma**2)
)


# amplitude temporelle de la source
def source_amplitude(time: float) -> float:
    if time <= heating_duration:
        return q_peak
    return 0.0


# moyenne harmonique entre deux conductivités
def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b)


# conductance entre une cellule et l'air extérieur
def convective_conductance(
    k_cell: float,
    distance: float,
    area: float,
    h_convection: float,
) -> float:
    conduction_resistance = distance / (k_cell * area)
    convection_resistance = 1.0 / (h_convection * area)
    return 1.0 / (
        conduction_resistance
        + convection_resistance
    )


# construction de la matrice de conduction
number_of_unknowns = N**2
K = sp.lil_matrix(
    (number_of_unknowns, number_of_unknowns),
    dtype=np.float64,
)

# contribution de la température ambiante
ambient_rhs = np.zeros(number_of_unknowns)

# assemblage des flux entre les cellules
for i in range(N):
    for j in range(N):
        p = i * N + j

        # interface avec la cellule de droite
        if j < N - 1:
            east = p + 1
            k_face = harmonic_mean(
                conductivity[i, j],
                conductivity[i, j + 1],
            )
            conductance = (
                k_face
                * dy
                * thickness
                / dx
            )

            K[p, p] += conductance
            K[east, east] += conductance
            K[p, east] -= conductance
            K[east, p] -= conductance

        # interface avec la cellule supérieure
        if i < N - 1:
            north = p + N
            k_face = harmonic_mean(
                conductivity[i, j],
                conductivity[i + 1, j],
            )
            conductance = (
                k_face
                * dx
                * thickness
                / dy
            )

            K[p, p] += conductance
            K[north, north] += conductance
            K[p, north] -= conductance
            K[north, p] -= conductance

# convection sur les bords gauche et droit
for i in range(N):
    for j in (0, N - 1):
        p = i * N + j
        conductance = convective_conductance(
            conductivity[i, j],
            dx / 2.0,
            dy * thickness,
            h_edges,
        )

        K[p, p] += conductance
        ambient_rhs[p] += conductance * T_ambient

# convection sur les bords inférieur et supérieur
for j in range(N):
    for i in (0, N - 1):
        p = i * N + j
        conductance = convective_conductance(
            conductivity[i, j],
            dy / 2.0,
            dx * thickness,
            h_edges,
        )

        K[p, p] += conductance
        ambient_rhs[p] += conductance * T_ambient

# convection sur les deux grandes faces de la plaque
surface_conductance = (
    2.0
    * h_surfaces
    * dx
    * dy
)

for p in range(number_of_unknowns):
    K[p, p] += surface_conductance
    ambient_rhs[p] += (
        surface_conductance
        * T_ambient
    )

K = K.tocsr()

# condition initiale : toute la plaque est à la température ambiante
T_current = np.full(
    number_of_unknowns,
    T_ambient,
)

# matrice d'Euler implicite
mass_over_dt = sp.diags(
    thermal_capacity / dt,
    format="csr",
)

time_matrix = mass_over_dt + K

# factorisation calculée une seule fois
solve_time_step = spla.factorized(
    time_matrix.tocsc()
)

# stockage des résultats pour l'animation
times = [0.0]
temperature_frames = [
    T_current.reshape(N, N).copy()
]

# résolution temporelle
for step in range(1, number_of_steps + 1):
    time = step * dt

    # source volumique en watts dans chaque cellule
    source_vector = (
        source_amplitude(time)
        * source_profile
        * cell_volume
    ).ravel()

    right_hand_side = (
        thermal_capacity / dt
        * T_current
        + ambient_rhs
        + source_vector
    )

    T_current = solve_time_step(
        right_hand_side
    )

    if step % save_every == 0 or step == number_of_steps:
        times.append(time)
        temperature_frames.append(
            T_current.reshape(N, N).copy()
        )

times = np.asarray(times)
temperature_frames = np.asarray(
    temperature_frames
)

maximum_temperatures = (
    temperature_frames.max(axis=(1, 2))
)

print("Nombre de pas de temps :", number_of_steps)
print("Température initiale :", temperature_frames[0].max())
print("Température maximale :", temperature_frames.max())
print("Température finale :", temperature_frames[-1].max())

# création de la figure
figure, (
    temperature_axis,
    curve_axis,
) = plt.subplots(
    1,
    2,
    figsize=(13, 5),
)

minimum_temperature = T_ambient
maximum_temperature = temperature_frames.max()

image = temperature_axis.imshow(
    temperature_frames[0],
    extent=[0, Lx, 0, Ly],
    origin="lower",
    cmap="inferno",
    vmin=minimum_temperature,
    vmax=maximum_temperature,
)

figure.colorbar(
    image,
    ax=temperature_axis,
    label="Température (°C)",
)

temperature_axis.scatter(
    xs,
    ys,
    color="cyan",
    marker="x",
    s=80,
    label="Source",
)

temperature_axis.add_patch(
    plt.Circle(
        (xc, yc),
        radius,
        fill=False,
        color="white",
        linewidth=2,
        label="Inclusion",
    )
)

temperature_axis.set_xlabel("x (m)")
temperature_axis.set_ylabel("y (m)")
temperature_axis.set_aspect("equal")
temperature_axis.legend()

temperature_title = (
    temperature_axis.set_title(
        "Température à t = 0.0 s"
    )
)

# courbe de température maximale
temperature_curve, = curve_axis.plot(
    [],
    [],
    color="red",
    linewidth=2,
)

curve_axis.axvline(
    heating_duration,
    color="black",
    linestyle="--",
    label="Arrêt de la source",
)

curve_axis.set_xlim(0, final_time)
curve_axis.set_ylim(
    T_ambient,
    maximum_temperature * 1.02,
)

curve_axis.set_xlabel("Temps (s)")
curve_axis.set_ylabel("Température maximale (°C)")
curve_axis.set_title("Chauffage puis refroidissement")
curve_axis.grid()
curve_axis.legend()


# mise à jour de l'animation
def update(frame_index):
    image.set_data(
        temperature_frames[frame_index]
    )

    temperature_title.set_text(
        f"Température à t = {times[frame_index]:.1f} s"
    )

    temperature_curve.set_data(
        times[:frame_index + 1],
        maximum_temperatures[:frame_index + 1],
    )

    return (
        image,
        temperature_title,
        temperature_curve,
    )


# animation temporelle
animation = FuncAnimation(
    figure,
    update,
    frames=len(times),
    interval=60,
    repeat=True,
)

plt.tight_layout()
plt.show()

# enregistrement optionnel en GIF
# animation.save(
#     "temperature_extended.gif",
#     writer="pillow",
#     fps=20,
# )