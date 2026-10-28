# les bibliothèques

import numpy as np
# Import de scipy pour résoudre le problème stationnaire matriciel de températures
import scipy.sparse as sp
import scipy.sparse.linalg as spla
#pour l'affichage des résultats
import matplotlib.pyplot as plt


def solve_stationary(params: np.ndarray, N: int = 32, sigma: float = 0.08):  # Implementation for solving the stationary temperature problem

    params = np.asarray(params, dtype=np.float64)

    if params.shape != (7,):
        raise ValueError(
            "params doit avoir la forme (7,)."
        )

    # les paramètres
    log_conductivity_ratio, xc, yc, r, A, xs, ys = params
    kappa = np.exp(log_conductivity_ratio)
    h = 1.0 / (N + 1)

    # le vecteur de paramètres
    # params = [log(kappa), xc, yc, r, A, xs, ys]

    # la grille de sortie des températures avec les bords
    T = np.zeros((N + 2, N + 2))

    # la grille des positions adimensionnées entre 0 et 1
    l = np.linspace(0, 1, N + 2)
    X, Y = np.meshgrid(l, l)

    # conductivité du matériau
    k = np.ones((N + 2, N + 2))
    for i in range(N + 2):
        for j in range(N + 2):
            if (X[i, j] - xc)**2 + (Y[i, j] - yc)**2 <= r**2:  # si on est dans la zone d'inclusion
                k[i, j] = kappa  # alors on a la conductivité kappa (on stocke log(kappa) dans les params)
            else:
                k[i, j] = 1  # conductivité du matériau de base

    # calcul de la source de chaleur sur la grille spatiale intérieure
    q_grid = A * np.exp(
        -(
            (X[1:-1, 1:-1] - xs)**2
            + (Y[1:-1, 1:-1] - ys)**2
        )
        / (2 * sigma**2)
    )
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

    return T, K, q, residu_relatif