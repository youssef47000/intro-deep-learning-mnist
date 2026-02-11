# Classification MNIST

**Auteurs :** Youssef Bouamama & Amine Itji

Ce projet compare l'efficacité de différentes architectures de réseaux de neurones (Deep Learning) sur le dataset de chiffres manuscrits **MNIST**.

Il a été réalisé avec **PyTorch** et met en évidence la supériorité des réseaux convolutifs pour le traitement d'images par rapport aux architectures basées uniquement sur des **couches pleinement connectées**.

## Objectifs

* Implémenter des réseaux de neurones (Shallow, Deep, CNN).
* Comparer l'impact de la **profondeur** et de la **structure** (Convolutions vs Fully Connected).
* Optimiser les hyperparamètres (Learning rate, Batch size, Nombre d'époques, Taille des couches, Profondeur) pour chaque modèle.

## Architectures Implémentées

Le code source dans `src/` propose trois modèles distincts :

### 1. Shallow Network
* **Type :** MLP (Multi-Layer Perceptron)
* **Structure :** Une seule **couche pleinement connectée** cachée.
* **Rôle :** Sert de référence (baseline). Montre qu'une simple projection linéaire suivie d'une non-linéarité capture déjà l'essentiel.

### 2. Deep Network (Réseau Profond)
* **Type :** MLP Profond
* **Structure :** Empilement de plusieurs **couches pleinement connectées**.
* **Observation :** L'ajout de profondeur en "Fully Connected" n'améliore pas significativement la performance sur MNIST et augmente le risque de sur-apprentissage.

### 3. CNN (LeNet-5)
* **Type :** Convolutional Neural Network
* **Structure :** Architecture hybride : Extraction de features (Convolutions) $\to$ Classification (**Couches pleinement connectées**).
* **Avantage :** Exploite la structure spatiale de l'image. Atteint une meilleure précision avec **3x moins de paramètres** que les réseaux pleinement connectés.

## Synthèse des Résultats

### 📊 Synthèse des Résultats

Nos expériences ont permis d'obtenir les précisions suivantes sur le jeu de test :

| Modèle | Configuration Optimale | Précision (Test) | Paramètres | Conclusion |
| :--- | :--- | :---: | :---: | :--- |
| **Shallow** | Hidden=256, LR=0.001, Batch=32 | 98.14% | ~200k | Efficace mais limité par l'aplatissement. |
| **Deep** | [256→128→64], LR=0.001, Batch=32 | 98.03% | ~300k | Pas de gain via la profondeur. |
| **CNN** | LeNet-5, LR=0.001, Batch=32 | **98.99%** | **~60k** | **Excellent.** Quasi aucun overfitting. |


## Project Structure

```
projet_M2_deep_learning/
├── dataset/
│   └── mnist.pkl.gz
├── doc/
│   └── doc.md
├── src/
│   ├── main.py
│   ├── perceptron_pytorch.py
│   ├── perceptron_pytorch_data_auto_layer_optim.py
│   ├── shallow_network.py
│   ├── deep_network.py
│   └── cnn_network.py
├── .gitignore
├── projet.pdf
├── README.md
└── requirements.txt
```

## Installation

### 1. Setup Environment

```bash
python3 -m venv env
source env/bin/activate        # Linux/Mac
# or
env\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print('PyTorch OK')"
```

## Run

### Interactive Menu

```bash
cd src/
python main.py
```

### Individual Scripts

```bash
# Part 1 - Perceptron
python perceptron_pytorch.py

# Part 2 - Shallow Network
python shallow_network.py

# Part 3 - Deep Network
python deep_network.py

# Part 4 - CNN
python cnn_network.py
```

### Generate PDF 

```bash
pandoc RAPPORT.md -o RAPPORT.pdf --css style.css --pdf-engine=weasyprint --metadata title=""
```