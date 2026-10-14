# Problem Definition — Parameterized 2D Heat Conduction

## English version

### 1. Objective

The objective of this project is to build a machine-learning surrogate model for a parameterized numerical simulation.

The reference problem describes steady-state heat conduction in a heterogeneous two-dimensional plate. Given a small set of physical parameters, the surrogate model must predict the complete temperature field significantly faster than the numerical solver.

The problem is dimensionless in the initial MVP.

### 2. Governing equation

The computational domain is the unit square:

$$
\Omega = (0,1)^2.
$$

The temperature field $T(x,y)$ satisfies the steady-state heat equation:

$$
-\nabla \cdot \left(k(x,y;\theta)\nabla T(x,y)\right)
= q(x,y;\theta),
\qquad (x,y)\in\Omega.
$$

Homogeneous Dirichlet boundary conditions are imposed:

$$
T(x,y)=0,
\qquad (x,y)\in\partial\Omega.
$$

The plate boundaries are therefore maintained at a fixed reference temperature.

### 3. Heterogeneous conductivity

The plate contains one circular inclusion with a conductivity different from the background material.

The background conductivity is fixed to 1:

$$
k_{\mathrm{background}}=1.
$$

The conductivity field is defined by:

$$
k(x,y)=
\begin{cases}
\kappa,
& (x-c_x)^2+(y-c_y)^2\leq r^2,\\
1,
& \text{otherwise}.
\end{cases}
$$

Here:

- $\kappa$ is the conductivity ratio;
- $(c_x,c_y)$ is the center of the inclusion;
- $r$ is the inclusion radius.

### 4. Heat source

The plate is heated by a localized Gaussian source:

$$
q(x,y)
=
A\exp\left(
-\frac{(x-s_x)^2+(y-s_y)^2}{2\sigma^2}
\right).
$$

Here:

- $A$ is the source amplitude;
- $(s_x,s_y)$ is the source center;
- the source width is fixed to $\sigma=0.08$.

The source is allowed to overlap with the conductive inclusion.

### 5. Input parameters

The surrogate model receives the following seven-dimensional parameter vector:

$$
\theta =
\left(
\log\kappa,
c_x,
c_y,
r,
A,
s_x,
s_y
\right).
$$

| Parameter | Range | Description |
|---|---:|---|
| $\kappa$ | $[0.25,4.0]$ | Conductivity ratio |
| $c_x$ | $[0.25,0.75]$ | Inclusion center, horizontal coordinate |
| $c_y$ | $[0.25,0.75]$ | Inclusion center, vertical coordinate |
| $r$ | $[0.10,0.20]$ | Inclusion radius |
| $A$ | $[0.5,2.0]$ | Heat-source amplitude |
| $s_x$ | $[0.20,0.80]$ | Source center, horizontal coordinate |
| $s_y$ | $[0.20,0.80]$ | Source center, vertical coordinate |

The conductivity ratio will be sampled uniformly in logarithmic space. This gives balanced coverage of insulating and conductive inclusions.

### 6. Surrogate output

The reference temperature field is computed on a grid containing 32 by 32 interior points.

The surrogate model therefore predicts:

$$
32\times32=1024
$$

interior temperature values.

The zero-valued boundary is reconstructed explicitly and is not predicted by the model. This guarantees exact satisfaction of the prescribed boundary condition.

### 7. Reference numerical method

The reference solver will use a conservative finite-difference discretization.

Conductivity at cell interfaces will be computed using the harmonic mean:

$$
k_{i+1/2,j}
=
\frac{2k_{i,j}k_{i+1,j}}
{k_{i,j}+k_{i+1,j}}.
$$

This formulation preserves heat flux across discontinuous material interfaces.

The resulting linear system will be:

- sparse;
- symmetric;
- positive definite;
- assembled and solved in double precision;
- solved with SciPy sparse linear-algebra routines.

### 8. Solver validation

The numerical solver must be validated before dataset generation.

The validation procedure will include:

1. A relative algebraic residual below $10^{-10}$:

   $$
   \frac{\|AT-b\|_2}{\|b\|_2}<10^{-10}.
   $$

2. Non-negative temperatures for a non-negative heat source.

3. A symmetry test using a centered inclusion and centered source.

4. A grid-convergence test using the manufactured solution:

   $$
   T(x,y)=\sin(\pi x)\sin(\pi y).
   $$

   For constant conductivity $k=1$, the corresponding source is:

   $$
   q(x,y)=2\pi^2\sin(\pi x)\sin(\pi y).
   $$

   The measured convergence rate should be close to the expected second-order accuracy of the finite-difference scheme.

### 9. Scientific objective

The final surrogate model will be compared with the reference solver using:

- root mean squared error;
- mean absolute error;
- relative $L^2$ error;
- maximum temperature error;
- spatial error maps;
- discrete PDE residual;
- inference time;
- reference-solver time;
- measured speedup.

No performance or accuracy claim will be made without reproducible experimental measurements.

---

## Version française

### 1. Objectif

L’objectif de ce projet est de construire un modèle de substitution fondé sur le machine learning pour une simulation numérique paramétrée.

Le problème de référence décrit la conduction thermique stationnaire dans une plaque bidimensionnelle hétérogène. À partir d’un petit nombre de paramètres physiques, le modèle doit prédire le champ de température complet beaucoup plus rapidement que le solveur numérique.

Le problème est adimensionné dans la première version du projet.

### 2. Équation physique

Le domaine de calcul est le carré unité :

$$
\Omega = (0,1)^2.
$$

Le champ de température $T(x,y)$ vérifie l’équation thermique stationnaire :

$$
-\nabla \cdot \left(k(x,y;\theta)\nabla T(x,y)\right)
= q(x,y;\theta),
\qquad (x,y)\in\Omega.
$$

Des conditions de Dirichlet homogènes sont imposées :

$$
T(x,y)=0,
\qquad (x,y)\in\partial\Omega.
$$

Les bords de la plaque sont donc maintenus à une température de référence fixe.

### 3. Conductivité hétérogène

La plaque contient une inclusion circulaire dont la conductivité diffère de celle du matériau environnant.

La conductivité du matériau de fond est fixée à 1 :

$$
k_{\mathrm{fond}}=1.
$$

Le champ de conductivité est défini par :

$$
k(x,y)=
\begin{cases}
\kappa,
& (x-c_x)^2+(y-c_y)^2\leq r^2,\\
1,
& \text{ailleurs}.
\end{cases}
$$

Dans cette expression :

- $\kappa$ est le rapport de conductivité ;
- $(c_x,c_y)$ est le centre de l’inclusion ;
- $r$ est le rayon de l’inclusion.

### 4. Source thermique

La plaque est chauffée par une source gaussienne localisée :

$$
q(x,y)
=
A\exp\left(
-\frac{(x-s_x)^2+(y-s_y)^2}{2\sigma^2}
\right).
$$

Dans cette expression :

- $A$ est l’intensité de la source ;
- $(s_x,s_y)$ est le centre de la source ;
- la largeur de la source est fixée à $\sigma=0.08$.

La source peut recouvrir partiellement ou totalement l’inclusion conductrice.

### 5. Paramètres d’entrée

Le modèle de substitution reçoit le vecteur de sept paramètres suivant :

$$
\theta =
\left(
\log\kappa,
c_x,
c_y,
r,
A,
s_x,
s_y
\right).
$$

| Paramètre | Intervalle | Description |
|---|---:|---|
| $\kappa$ | $[0.25,4.0]$ | Rapport de conductivité |
| $c_x$ | $[0.25,0.75]$ | Coordonnée horizontale du centre de l’inclusion |
| $c_y$ | $[0.25,0.75]$ | Coordonnée verticale du centre de l’inclusion |
| $r$ | $[0.10,0.20]$ | Rayon de l’inclusion |
| $A$ | $[0.5,2.0]$ | Intensité de la source thermique |
| $s_x$ | $[0.20,0.80]$ | Coordonnée horizontale du centre de la source |
| $s_y$ | $[0.20,0.80]$ | Coordonnée verticale du centre de la source |

Le rapport de conductivité sera échantillonné uniformément en échelle logarithmique. Cela permettra de représenter équitablement les inclusions isolantes et conductrices.

### 6. Sortie du modèle

Le champ de température de référence est calculé sur une grille de 32 par 32 points intérieurs.

Le modèle prédit donc :

$$
32\times32=1024
$$

valeurs de température intérieures.

Les valeurs nulles du bord sont reconstruites explicitement et ne sont pas prédites par le modèle. La condition aux limites est ainsi respectée exactement.

### 7. Méthode numérique de référence

Le solveur utilisera une discrétisation conservative par différences finies.

La conductivité entre deux cellules sera calculée avec la moyenne harmonique :

$$
k_{i+1/2,j}
=
\frac{2k_{i,j}k_{i+1,j}}
{k_{i,j}+k_{i+1,j}}.
$$

Cette formulation préserve correctement le flux thermique aux interfaces entre les matériaux.

Le système linéaire obtenu sera :

- creux ;
- symétrique ;
- défini positif ;
- assemblé et résolu en double précision ;
- résolu avec les outils d’algèbre linéaire creuse de SciPy.

### 8. Validation du solveur

Le solveur numérique devra être validé avant toute génération de données.

La validation comprendra :

1. Un résidu algébrique relatif inférieur à $10^{-10}$ :

   $$
   \frac{\|AT-b\|_2}{\|b\|_2}<10^{-10}.
   $$

2. Des températures non négatives pour une source thermique positive.

3. Un test de symétrie avec une inclusion et une source centrées.

4. Un test de convergence de grille utilisant la solution manufacturée :

   $$
   T(x,y)=\sin(\pi x)\sin(\pi y).
   $$

   Pour une conductivité constante $k=1$, la source correspondante est :

   $$
   q(x,y)=2\pi^2\sin(\pi x)\sin(\pi y).
   $$

   Le taux de convergence mesuré devra être proche de l’ordre 2 attendu pour le schéma de différences finies.

### 9. Objectif scientifique

Le modèle final sera comparé au solveur de référence selon :

- la RMSE ;
- la MAE ;
- l’erreur relative $L^2$ ;
- l’erreur sur la température maximale ;
- les cartes d’erreur spatiale ;
- le résidu discret de l’équation ;
- le temps d’inférence ;
- le temps du solveur de référence ;
- le facteur d’accélération mesuré.

Aucune affirmation concernant la précision ou les performances ne sera formulée sans mesures expérimentales reproductibles.