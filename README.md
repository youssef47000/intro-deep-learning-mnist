# projet_M2_deep_learning
# INTRODUCTION TO DEEP LEARNING
    - Amine ITJI (p2018984)
    - Youssef BOUAMAMA (p2306151)

# Part 1: Perceptrons

## Analyse des tenseurs dans perceptron_pytorch.py

### Weight tensor (w)
```python
w = torch.empty((data_train.shape[1],label_train.shape[1]),dtype=torch.float)
```
Le tenseur des poids w a une dimension de **[784, 10]**.
- **784 lignes** : correspondent aux 784 pixels de chaque image MNIST (28×28 pixels)
- **10 colonnes** : représentent les 10 classes de sortie (chiffres 0 à 9)

Chaque élément w[i,j] représente le poids de connexion entre le pixel i et la classe j.

### Bias tensor (b)
```python
b = torch.empty((1,label_train.shape[1]),dtype=torch.float)
```
Le tenseur des biais b a une dimension de **[1, 10]**.
- **1 ligne** : un seul vecteur de biais partagé pour tous les échantillons
- **10 colonnes** : un biais pour chaque classe de sortie

Les biais permettent d'ajuster le seuil d'activation pour chaque classe.

### Input tensor (x)
```python
x = data_train[indices[i:i+batch_size]]
```
Le tenseur d'entrée x a une dimension de **[5, 784]**
- **5 lignes** : correspond à la taille du batch (batch_size = 5)
- **784 colonnes** : les 784 pixels de chaque image, mis à plat

Chaque ligne représente une image complète sous forme vectorielle.

### Output tensor (y)
```python
y = torch.mm(x,w)+b
```
Le tenseur de sortie y a une dimension de **[5, 10]**
- **5 lignes** : une prédiction pour chaque image du batch
- **10 colonnes** : les scores pour chaque classe (0 à 9)

Chaque ligne contient les scores non normalisés pour les 10 classes d'une image donnée.

### Target tensor (t)
```python
t = label_train[indices[i:i+batch_size]]
```
Le tenseur des target labels t a une dimension de **[5, 10]**
- **5 lignes** : les target labels pour chaque image du batch
- **10 colonnes** : encodage one-hot des classes

Par exemple, pour le chiffre 5, le target label sera [0,0,0,0,0,1,0,0,0,0].

### Gradient tensor (grad)
```python
grad = (t-y)
```
Le tenseur du gradient grad a une dimension de **[5, 10]**
- **5 lignes** : l'erreur pour chaque image du batch
- **10 colonnes** : l'erreur pour chaque classe

Ce gradient représente la différence entre les prédictions et les target labels, utilisée pour la mise à jour des poids.

# Part 2: Shallow network

## Implémentation du shallow network

Un MLP avec **une seule couche cachée** et une couche de sortie linéaire est implémenté en utilisant les outils PyTorch :

**Architecture :** 784 (input) → hidden_size (ReLU) → 10 (linear output)

**Outils PyTorch utilisés :**
- `nn.Linear` pour les couches fully-connected
- `nn.ReLU` comme fonction d'activation pour la couche cachée
- `nn.CrossEntropyLoss` comme fonction de perte
- `optim.Adam` comme optimiseur
- `DataLoader` pour le chargement par batch

## Méthodologie pour trouver les hyperparamètres

### Création du dataset de validation (évitement de l'overfitting)
- **Split train/validation** : 80% / 20% des données d'entraînement originales  
- **Mélange aléatoire** avec `torch.randperm()` pour éviter les biais
- **Critère de sélection** : validation accuracy (pas test accuracy) pour éviter l'overfitting
- **Poids aléatoires** : initialisation par défaut PyTorch (Xavier/Glorot normale)

### Stratégie de test des hyperparamètres
**Tests d'un paramètre à la fois** (approche one-factor-at-a-time) :
1. **η (learning rate)** : [0.0001, 0.001, 0.01] avec hidden=128, batch=64
2. **Nombre de neurones cachés** : [64, 128, 256, 512] avec lr=0.001, batch=64  
3. **Batch size** : [32, 64, 128] avec hidden=128, lr=0.001

### Justification du nombre d'expériences
**Choix : 11 expériences au total** (au lieu de grid search exhaustif 4×3×3 = 36)
- **15 époques par test** au lieu de 20 pour accélérer les expériences
- **Approche séquentielle** plutôt que grid search complet (trop coûteux en temps)
- **Tests ciblés** sur les hyperparamètres les plus influents selon la littérature
- **Durée réelle mesurée** : 2.5 minutes au total, moyenne de 14.8s par expérience

## Résultats expérimentaux

### Test initial
```
Test avec hidden_size=128, lr=0.001, batch_size=64
Epoch  1/20 | Train Acc:  89.94% | Val Acc:  93.52% | Test Acc:  94.10% | Time: 0.9s
Epoch  6/20 | Train Acc:  98.29% | Val Acc:  97.21% | Test Acc:  97.46% | Time: 0.8s
Epoch 11/20 | Train Acc:  99.43% | Val Acc:  97.21% | Test Acc:  97.56% | Time: 0.8s
Epoch 16/20 | Train Acc:  99.82% | Val Acc:  97.45% | Test Acc:  98.09% | Time: 0.8s
Epoch 20/20 | Train Acc:  99.93% | Val Acc:  97.38% | Test Acc:  97.61% | Time: 0.8s

Résultats de base: Val=97.45%, Test=97.61% | Durée: 17.8s
```

### Recherche d'hyperparamètres

#### 1. Influence du nombre de neurones cachés (Section 1: 66.5s)
| Hidden Size | Validation Acc | Test Acc | Temps |
|-------------|----------------|----------|-------|
| 64          | 97.39%         | 97.50%   | 12.5s |
| 128         | 97.44%         | 97.74%   | 13.6s |
| **256**     | **97.94%**     | **98.14%** | 17.5s |
| 512         | 97.86%         | 98.06%   | 22.7s |

#### 2. Influence du learning rate (Section 2: 39.8s)
| Learning Rate | Validation Acc | Test Acc | Temps |
|---------------|----------------|----------|-------|
| 0.0001        | 95.38%         | 95.79%   | 13.5s |
| **0.001**     | **97.70%**     | **97.84%** | 13.0s |
| 0.01          | 96.81%         | 96.69%   | 13.3s |

#### 3. Influence du batch size (Section 3: 42.2s)
| Batch Size | Validation Acc | Test Acc | Temps |
|------------|----------------|----------|-------|
| **32**     | **97.77%**     | **98.04%** | 20.0s |
| 64         | 97.63%         | 97.96%   | 12.9s |
| 128        | 97.44%         | 97.53%   | 9.2s  |

### Résumé complet des expériences
| Paramètre   | Valeur | Validation | Test    | Temps |
|-------------|--------|------------|---------|-------|
| hidden_size | 64     | 97.39%     | 97.50%  | 12.5s |
| hidden_size | 128    | 97.44%     | 97.74%  | 13.6s |
| hidden_size | **256** | **97.94%** | **98.14%** | 17.5s |
| hidden_size | 512    | 97.86%     | 98.06%  | 22.7s |
| lr          | 0.0001 | 95.38%     | 95.79%  | 13.5s |
| lr          | **0.001** | **97.70%** | **97.84%** | 13.0s |
| lr          | 0.01   | 96.81%     | 96.69%  | 13.3s |
| batch_size  | **32** | **97.77%** | **98.04%** | 20.0s |
| batch_size  | 64     | 97.63%     | 97.96%  | 12.9s |
| batch_size  | 128    | 97.44%     | 97.53%  | 9.2s  |

## Meilleur résultat
- **Paramètre optimal** : hidden_size = 256
- **Validation accuracy** : 97.94%
- **Test accuracy** : 98.14%
- **Temps d'entraînement** : 17.5s

## Bilan temporel
- **Moyenne par expérience** : 14.8s
- **Temps total** : 148.4s (2.5 minutes)

## Analyse de l'influence de chaque hyperparamètre

### 1. η (Learning Rate) - Impact : ±2.32%
- **lr=0.0001** : 95.38% → **Apprentissage lent**, convergence insuffisante en 15 époques
- **lr=0.001** : 97.70% → **Optimal**, bon équilibre vitesse/stabilité  
- **lr=0.01** : 96.81% → **Instabilité**, oscillations autour du minimum

**Explication** : Le learning rate contrôle la taille des pas dans l'espace des poids. Trop faible → sous-apprentissage, trop élevé → instabilité.

### 2. Nombre de neurones cachés - Impact : +0.55%
- **64 neurones** : 97.39% → **Capacité limitée** du modèle
- **256 neurones** : 97.94% → **Capacité optimale** pour MNIST
- **512 neurones** : 97.86% → **Overfitting**, performance dégradée

**Explication** : 256 neurones offrent le bon équilibre capacité/généralisation. Au-delà, on observe un début d'overfitting sur ce dataset.

### 3. Batch Size - Impact : +0.33%
- **batch=32** : 97.77% → **Optimal**, bon équilibre bruit/exploration mais plus lent (20.0s)
- **batch=64** : 97.63% → **Compromis** performance/vitesse (12.9s)
- **batch=128** : 97.44% → **Plus rapide** (9.2s) mais gradients lisses

**Explication** : Petit batch → gradients bruités (meilleure exploration) mais coût plus élevé. Grand batch → convergence plus rapide mais risque de minima locaux.

## Conclusion

La configuration optimale trouvée est **hidden_size=256, lr=0.001, batch_size=32**. Le learning rate reste l'hyperparamètre le plus critique (±2.32% d'impact), suivi du nombre de neurones cachés (+0.55%) puis de la batch size (+0.33%).

# Part 3: Deep network

## Implémentation du deep network

Le réseau multi-couches est implémenté avec au moins deux couches cachées en utilisant `nn.Sequential`. L'architecture utilise `nn.Linear` pour les connexions fully-connected, `nn.ReLU` comme fonction d'activation, `nn.CrossEntropyLoss` pour la fonction de perte et `optim.Adam` comme optimiseur.

## Méthodologie pour trouver les hyperparamètres

**Contraintes de calcul :** 11 expériences au total, 15 époques par test, approche séquentielle testant un paramètre à la fois.

**Tests effectués :** 5 architectures différentes (2-3 couches), 3 valeurs de learning rate, 3 tailles de batch. Durée totale mesurée : 3.6 minutes.

## Résultats expérimentaux

### Test initial
Configuration de base : hidden_layers=[256, 128], lr=0.001, batch_size=64

```
Epoch  1/15 | Train Acc:  91.22% | Val Acc:  94.64% | Test Acc:  95.29%
Epoch 15/15 | Train Acc:  99.63% | Val Acc:  97.39% | Test Acc:  97.87%
Résultats de base: Val=98.06% (maximum), Test=97.87% | Durée: 17.3s
```

### Recherche d'hyperparamètres

#### 1. Architecture (Impact : +0.42%)
| Structure | Validation Acc | Test Acc | Temps |
|-----------|----------------|----------|-------|
| 784 → 128→64 → 10 | 97.71% | 97.79% | 14.8s |
| 784 → 256→128 → 10 | 98.05% | 98.06% | 17.3s |
| 784 → 512→256 → 10 | 98.03% | 98.31% | 25.5s |
| **784 → 256→128→64 → 10** | **98.13%** | **98.03%** | **19.3s** |
| 784 → 512→256→128 → 10 | 97.99% | 98.21% | 29.3s |

#### 2. Learning Rate (Impact : ±0.78%)
| Learning Rate | Validation Acc | Test Acc |
|---------------|----------------|----------|
| 0.0001 | 97.10% | 97.23% |
| **0.001** | **97.88%** | **97.73%** |
| 0.01 | 97.22% | 97.43% |

#### 3. Batch Size (Impact : +0.20%)
| Batch Size | Validation Acc | Test Acc |
|------------|----------------|----------|
| **32** | **97.89%** | **98.21%** |
| 64 | 97.79% | 98.01% |
| 128 | 97.84% | 97.69% |

## Résultats optimaux

**Configuration optimale :** 256→128→64 (3 couches cachées), lr=0.001, batch_size=32
- **Validation accuracy :** 98.13%
- **Test accuracy :** 98.03%
- **Temps d'entraînement :** 19.3s

## Comparaison avec le shallow network

| Architecture | Meilleure Test Acc | Temps moyen |
|--------------|-------------------|-------------|
| Shallow Network | 98.14% | 14.8s |
| Deep Network | 98.03% | 19.7s |
| **Différence** | -0.11% | +33% |

Le deep network n'apporte pas d'amélioration par rapport au shallow network sur MNIST, avec un surcoût de 33% pour un gain négligeable.

# Part 4: CNN

## Implémentation du CNN

Deux architectures CNN sont implémentées et comparées pour la classification MNIST. La transformation des données vectorielles [784] vers le format image [1, 28, 28] permet l'utilisation des convolutions.

### Architectures implémentées

#### 1. LeNet-5 inspiré (Architecture classique)
```
1 → Conv2d(6) → MaxPool → Conv2d(16) → MaxPool → FC(120) → FC(84) → FC(10)
- Conv1: 1→6 channels, kernel 5x5, padding 2 (28x28 → 28x28)
- MaxPool1: 28x28 → 14x14
- Conv2: 6→16 channels, kernel 5x5 (14x14 → 10x10)
- MaxPool2: 10x10 → 5x5
- FC: 400 → 120 → 84 → 10
```

#### 2. CNN Simple (Architecture moderne)
```
1 → Conv2d(32) → MaxPool → Conv2d(64) → MaxPool → FC(128) → FC(10)
- Conv1: 1→32 channels, kernel 3x3, padding 1 (28x28 → 28x28)
- MaxPool1: 28x28 → 14x14
- Conv2: 32→64 channels, kernel 3x3, padding 1 (14x14 → 14x14)
- MaxPool2: 14x14 → 7x7
- FC: 3136 → 128 → 10
```

### Outils PyTorch utilisés
- `nn.Conv2d` pour les couches convolutionnelles
- `nn.MaxPool2d` pour le pooling
- `F.relu` comme fonction d'activation
- `nn.Dropout(0.5)` pour la régularisation
- `nn.CrossEntropyLoss` et `optim.Adam`

## Méthodologie

**Contraintes de calcul :** 8 expériences au total, 5 époques pour tests d'architecture, 10 époques pour hyperparamètres.

**Tests effectués :** 2 architectures, 3 learning rates, 3 batch sizes. Durée totale : 4.7 minutes (35.0s par expérience).

## Résultats expérimentaux

### Test initial (LeNet-5, 15 époques)
```
Transformation des données: [N, 784] -> [N, 1, 28, 28]

Epoch  1/15 | Train Acc:  85.52% | Val Acc:  96.24% | Test Acc:  96.91%
Epoch 15/15 | Train Acc:  98.87% | Val Acc:  99.03% | Test Acc:  99.10%
Résultats CNN de base: Val=99.03%, Test=99.10% | Durée: 48.9s
```

### Recherche d'hyperparamètres

#### 1. Architecture (Impact : +0.66%)
| Modèle | Validation Acc | Test Acc | Temps |
|--------|----------------|----------|-------|
| **CNN Simple** | **99.02%** | **98.94%** | 60.3s |
| LeNet-5 | 98.36% | 98.77% | 17.1s |

#### 2. Learning Rate (Impact : ±1.75%)
| Learning Rate | Validation Acc | Test Acc |
|---------------|----------------|----------|
| 0.0001 | 97.22% | 97.64% |
| **0.001** | **98.97%** | **98.80%** |
| 0.01 | 98.31% | 98.43% |

#### 3. Batch Size (Impact : +0.17%)
| Batch Size | Validation Acc | Test Acc |
|------------|----------------|----------|
| 32 | 98.79% | 98.94% |
| **64** | **98.86%** | **98.87%** |
| 128 | 98.69% | 98.83% |

## Résultats optimaux

**Configuration optimale :** CNN Simple, lr=0.001, batch_size=64
- **Validation accuracy :** 99.02%
- **Test accuracy :** 98.94%
- **Temps d'entraînement :** 60.3s

## Comparaison avec les architectures précédentes

| Architecture | Meilleure Test Acc | Temps moyen |
|--------------|-------------------|-------------|
| Perceptron | ~92% | ~5s |
| Shallow Network | 98.14% | 14.8s |
| Deep Network | 98.03% | 19.7s |
| **CNN** | **98.94%** | 35.0s |

### Analyse comparative
- **Gain CNN vs MLP :** +0.80% à +0.91% de précision
- **Surcoût temporel :** +78% par rapport au Deep Network
- **Trade-off :** CNN Simple 3.5× plus lent que LeNet-5 pour +0.66% de précision

## Conclusion

Les CNN démontrent leur efficacité pour la classification d'images avec une amélioration des performances sur MNIST. Le CNN Simple atteint les meilleures performances (99.02% validation) mais nécessite un temps de calcul plus élevé que LeNet-5.
