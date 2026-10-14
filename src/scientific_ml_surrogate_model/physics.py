# les bibliothèques
import numpy as np
import tensorflow as tf

# les paramètres
log_conductivity_ratio = 0
inclusion_center_x = 0
inclusion_center_y = 0
inclusion_radius = 0
source_amplitude = 0
source_center_x = 0
source_center_y = 0


# le vecteur de paramètres
params = np.array(
    [
        log_conductivity_ratio,
        inclusion_center_x,
        inclusion_center_y,
        inclusion_radius,
        source_amplitude,
        source_center_x,
        source_center_y,
    ]
)

# la grille de sortie des températures
T = np.zeros((32, 32))
