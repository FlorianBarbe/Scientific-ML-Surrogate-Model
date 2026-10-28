# les bibliothèques

import numpy as np
# Import de scipy pour résoudre le problème stationnaire matriciel de températures
import scipy.sparse as sp
import scipy.sparse.linalg as spla
#pour l'affichage des résultats
import matplotlib.pyplot as plt

N = 100  # taille de la grille spatiale intérieure
h = 1 / (N + 1)  # pas de discrétisation

# les paramètres
log_conductivity_ratio = np.log(0.1)  # logarithme du rapport de conductivité entre l'inclusion et le matériau de base
xc, yc = 0.3, 0.5  # centre de l'inclusion
r = 0.3  # rayon de l'inclusion
sigma = 0.08  # largeur de la source gaussienne
A = 1  # amplitude de la source
xs, ys = 0.6, 0.6  # centre de la source

# le vecteur de paramètres
params = np.array([log_conductivity_ratio, xc, yc, r, A, xs, ys])

# la grille de sortie des températures avec les bords
T = np.zeros((N + 2, N + 2))

# la grille des positions adimensionnées entre 0 et 1
l = np.linspace(0, 1, N + 2)
X, Y = np.meshgrid(l, l)

# conductivité du matériau
k = np.ones((N + 2, N + 2))
for i in range(N + 2):
    for j in range(N + 2):
        if (X[i, j] - params[1])**2 + (Y[i, j] - params[2])**2 <= params[3]**2:  # si on est dans la zone d'inclusion
            k[i, j] = np.exp(params[0])  # alors on a la conductivité kappa (on stocke log(kappa) dans les params)
        else:
            k[i, j] = 1  # conductivité du matériau de base

# calcul de la source de chaleur sur la grille spatiale intérieure
q_grid = params[4] * np.exp(-((X[1:-1, 1:-1] - params[5])**2 + (Y[1:-1, 1:-1] - params[6])**2) / (2 * sigma**2))
q = q_grid.ravel()

# fonction de moyenne harmonique
def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b)

# construction de la matrice creuse du système
K = sp.lil_matrix((N**2, N**2), dtype=np.float64)

for i in range(N):
    for j in range(N):
        ii, jj = i + 1, j + 1  # indices correspondants dans la grille contenant les bords
        p = i * N + j  # passage de l'indice 2D à l'indice 1D

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

# résolution du système linéaire
T_vector = spla.spsolve(K, q)
T[1:-1, 1:-1] = T_vector.reshape(N, N)

# vérification du résultat
residu_relatif = np.linalg.norm(K @ T_vector - q) / np.linalg.norm(q)

print("Forme de K :", K.shape)
print("Température minimale :", T.min())
print("Température maximale :", T.max())
print("Résidu relatif :", residu_relatif)

# affichage du champ de température
plt.figure(figsize=(7, 6))
plt.imshow(T, extent=[0, 1, 0, 1], origin="lower", cmap="inferno")
plt.colorbar(label="Température")
plt.scatter(xs, ys, color="cyan", marker="x", s=80, label="Source")
plt.gca().add_patch(plt.Circle((xc, yc), r, fill=False, color="white", linewidth=2, label="Inclusion"))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Champ de température stationnaire")
plt.legend()
plt.tight_layout()
plt.show()