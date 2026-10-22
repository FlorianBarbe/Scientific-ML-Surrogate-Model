# Scientific ML Surrogate Model

Ce projet vise à construire un modèle de substitution fondé sur le machine learning pour simuler la conduction thermique stationnaire dans une plaque bidimensionnelle hétérogène.

Un solveur physique de référence génère des champs de température à partir de plusieurs paramètres physiques. Ces données serviront ensuite à entraîner un réseau de neurones avec **TensorFlow/Keras**, afin de prédire le champ de température plus rapidement que le solveur numérique.

La définition complète du problème, les équations, les paramètres et la méthode numérique sont présentés dans la [définition du problème](docs/problem_definition.md).

## Technologies utilisées

- Python 3.13 ou version ultérieure ;
- NumPy et SciPy pour le calcul scientifique et le solveur physique ;
- TensorFlow/Keras pour le réseau de neurones ;
- pandas et scikit-learn pour la préparation des données ;
- Matplotlib pour la visualisation.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/FlorianBarbe/Scientific-ML-Surrogate-Model.git
cd Scientific-ML-Surrogate-Model
```

### 2. Créer l’environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l’environnement

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

### 4. Installer le projet et ses dépendances

```bash
python -m pip install --upgrade pip
pip install -e .
```

### 5. Vérifier l’installation de TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 6. Quitter l’environnement virtuel

```bash
deactivate
```

---

# Scientific ML Surrogate Model — English Version

This project aims to build a machine-learning surrogate model for the steady-state heat conduction simulation of a heterogeneous two-dimensional plate.

A reference physics-based solver generates temperature fields from several physical parameters. These data will then be used to train a neural network with **TensorFlow/Keras** to predict the temperature field faster than the numerical solver.

The equations, physical parameters, numerical method and complete problem description are available in the [problem definition](docs/problem_definition.md).

## Technologies

- Python 3.13 or later;
- NumPy and SciPy for scientific computing and the physics-based solver;
- TensorFlow/Keras for the neural network;
- pandas and scikit-learn for data preparation;
- Matplotlib for visualization.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/FlorianBarbe/Scientific-ML-Surrogate-Model.git
cd Scientific-ML-Surrogate-Model
```

### 2. Create the virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 4. Install the project and its dependencies

```bash
python -m pip install --upgrade pip
pip install -e .
```

### 5. Check the TensorFlow installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 6. Leave the virtual environment

```bash
deactivate
```
