
# les bibliothèques
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 32  # taille de la grille spatiale intérieure
h = 1 / (N + 1)  # pas de discrétisation

# les paramètres physiques
log_conductivity_ratio = np.log(0.1)
xc, yc = 0.3, 0.5  # centre de l'inclusion
r = 0.3  # rayon de l'inclusion
sigma = 0.08  # largeur de la source gaussienne
A = 1.0  # amplitude de la source
xs, ys = 0.6, 0.6  # centre de la source

# le vecteur de paramètres
params = np.array([log_conductivity_ratio, xc, yc, r, A, xs, ys])

# les paramètres temporels
rho_c = 1.0  # capacité thermique volumique adimensionnée
dt = 0.001  # pas de temps
final_time = 0.5  # temps final
save_every = 5  # sauvegarde d'une image tous les 5 pas de temps
number_of_steps = int(final_time / dt)

# la grille des positions adimensionnées entre 0 et 1
coordinates = np.linspace(0, 1, N + 2)
X, Y = np.meshgrid(coordinates, coordinates)

# conductivité du matériau
k = np.ones((N + 2, N + 2))
for i in range(N + 2):
    for j in range(N + 2):
        if (X[i, j] - params[1])**2 + (Y[i, j] - params[2])**2 <= params[3]**2:
            k[i, j] = np.exp(params[0])
        else:
            k[i, j] = 1.0

# calcul de la source de chaleur sur la grille spatiale intérieure
X_interior = X[1:-1, 1:-1]
Y_interior = Y[1:-1, 1:-1]
q_grid = params[4] * np.exp(-((X_interior - params[5])**2 + (Y_interior - params[6])**2) / (2 * sigma**2))
q = q_grid.ravel()

# fonction de moyenne harmonique
def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b)

# construction de la matrice creuse K
K = sp.lil_matrix((N**2, N**2), dtype=np.float64)

for i in range(N):
    for j in range(N):
        ii, jj = i + 1, j + 1
        p = i * N + j

        kw = harmonic_mean(k[ii, jj], k[ii, jj - 1])
        ke = harmonic_mean(k[ii, jj], k[ii, jj + 1])
        kn = harmonic_mean(k[ii, jj], k[ii + 1, jj])
        ks = harmonic_mean(k[ii, jj], k[ii - 1, jj])

        K[p, p] = (ke + kw + kn + ks) / h**2
        if j < N - 1: K[p, p + 1] = -ke / h**2
        if j > 0: K[p, p - 1] = -kw / h**2
        if i < N - 1: K[p, p + N] = -kn / h**2
        if i > 0: K[p, p - N] = -ks / h**2

K = K.tocsr()

# condition initiale : la plaque est initialement froide
T_current = np.zeros(N**2)

# matrice du schéma d'Euler implicite
identity = sp.eye(N**2, format="csr")
K_temporal = rho_c * identity + dt * K

# factorisation effectuée une seule fois
solve_time_step = spla.factorized(K_temporal.tocsc())

# stockage des températures pour l'animation
times = [0.0]
temperature_frames = [np.zeros((N + 2, N + 2))]

# résolution temporelle
for step in range(1, number_of_steps + 1):
    right_hand_side = rho_c * T_current + dt * q
    T_current = solve_time_step(right_hand_side)

    if step % save_every == 0 or step == number_of_steps:
        T_grid = np.zeros((N + 2, N + 2))
        T_grid[1:-1, 1:-1] = T_current.reshape(N, N)
        temperature_frames.append(T_grid)
        times.append(step * dt)

temperature_frames = np.asarray(temperature_frames)
times = np.asarray(times)
maximum_temperatures = temperature_frames.max(axis=(1, 2))

print("Nombre de pas de temps :", number_of_steps)
print("Forme des températures :", temperature_frames.shape)
print("Température maximale finale :", temperature_frames[-1].max())

# création de l'affichage temporel
figure, (axis_temperature, axis_curve) = plt.subplots(1, 2, figsize=(13, 5))

image = axis_temperature.imshow(
    temperature_frames[0],
    extent=[0, 1, 0, 1],
    origin="lower",
    cmap="inferno",
    vmin=0,
    vmax=temperature_frames.max(),
)

figure.colorbar(image, ax=axis_temperature, label="Température")
axis_temperature.scatter(xs, ys, color="cyan", marker="x", s=80, label="Source")
axis_temperature.add_patch(
    plt.Circle(
        (xc, yc),
        r,
        fill=False,
        color="white",
        linewidth=2,
        label="Inclusion",
    )
)
axis_temperature.set_xlabel("x")
axis_temperature.set_ylabel("y")
axis_temperature.legend()
temperature_title = axis_temperature.set_title("Température à t = 0.000")

# courbe de la température maximale en fonction du temps
temperature_line, = axis_curve.plot([], [], color="red", linewidth=2)
axis_curve.set_xlim(0, final_time)
axis_curve.set_ylim(0, maximum_temperatures.max() * 1.05)
axis_curve.set_xlabel("Temps")
axis_curve.set_ylabel("Température maximale")
axis_curve.set_title("Évolution de la température maximale")
axis_curve.grid()

# fonction appelée à chaque image de l'animation
def update(frame_index):
    image.set_data(temperature_frames[frame_index])
    temperature_title.set_text(f"Température à t = {times[frame_index]:.3f}")
    temperature_line.set_data(times[:frame_index + 1], maximum_temperatures[:frame_index + 1])
    return image, temperature_title, temperature_line

# création de l'animation
animation = FuncAnimation(
    figure,
    update,
    frames=len(times),
    interval=80,
    repeat=True,
)

plt.tight_layout()
plt.show()

# pour enregistrer l'animation en GIF si Pillow est installé
# animation.save("evolution_temperature.gif", writer="pillow", fps=20)