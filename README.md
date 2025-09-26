Voici la version révisée sans les adverbes et exagérations :

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






















































## Partie 4 : CNN (Convolutional Neural Network)

### Implémentation

On a implémenté deux architectures CNN à comparer. LeNet-5, avec deux couches convolutionnelles et trois couches fully connected, et un CNN simple avec deux convolutions et deux couches fully connected.

LeNet-5 utilise les configurations suivantes :  
- Conv1 : 1 canal → 6 cartes, kernel 5×5, padding 2, taille image maintenue à 28×28  
- MaxPool2d réduit la taille à 14×14  
- Conv2 : 6 → 16 cartes, kernel 5×5, taille réduite à 10×10  
- MaxPool2d réduit à 5×5  
- Fully connected : 400 → 120 → 84 → 10

Le CNN simple utilise :  
- Conv1 : 1 → 32 filtres, kernel 3×3, padding 1, taille 28×28  
- MaxPool2d réduit à 14×14  
- Conv2 : 32 → 64 filtres, kernel 3×3, padding 1, taille maintenue à 14×14  
- MaxPool2d réduit à 7×7  
- Fully connected : 3136 → 128 → 10

On a appliqué ReLU, Dropout 0.5, CrossEntropyLoss et optimiseur Adam.

### Méthodologie

Les images sont transformées du format vecteur (784) au format image (1×28×28). On fait varier architecture, learning rate (0.0001, 0.001, 0.01) et batch sizes (32, 64, 128). Chaque configuration est testée sur validation puis évaluation test.

### Résultats

Le test initial avec LeNet-5 (lr=0.001, batch=64) donne à la 15e époque 99.04% de précision test, avec un temps total d'environ 59 secondes (dont 1.68s pour charger les données et 55.27s d'entraînement). Dès la première époque, la précision test est déjà à 97.44%.

| Époque | Train Acc (%) | Val Acc (%) | Test Acc (%) | Temps (s) |
|--------|---------------|-------------|--------------|-----------|
| 1      | 87.09         | 97.16       | 97.44        | 6.2       |
| 4      | 97.51         | 98.45       | 98.53        | 3.6       |
| 7      | 98.29         | 98.73       | 98.71        | 3.5       |
| 10     | 98.54         | 98.82       | 98.73        | 3.4       |
| 13     | 98.93         | 98.97       | 98.96        | 3.4       |
| 15     | 98.96         | 98.94       | 99.04        | 3.4       |

En explorant les architectures, le CNN simple atteint 98.87% en test (batch=64, lr=0.001) après 5 époques, mais en 68.7 secondes d’entraînement, contre 98.40% en 19.4 secondes pour LeNet-5 (mêmes hyperparamètres).

Pour l’influence du learning rate :  
- lr=0.0001 → val=97.29%, test=97.79%, entraînement 36s  
- lr=0.001 → val=98.87%, test=98.91%, temps 35.2s  
- lr=0.01 → val=97.60%, test=97.64%, 34.5s

La taille de batch montre un faible effet sur la précision finale, avec un léger avantage à batch=32 (val=98.98%, test=98.99%) contre batch=128 (val=98.75%, test=98.99%), le temps d’entraînement diminuant avec la taille du batch.

| Paramètre     | Valeur | Validation Acc (%) | Test Acc (%) | Temps (s) |
|---------------|---------|-------------------|-------------|-----------|
| model_type    | simple  | 98.71             | 98.87       | 68.7      |
| model_type    | lenet5  | 98.27             | 98.40       | 19.4      |
| lr            | 0.0001  | 97.29             | 97.79       | 36.0      |
| lr            | 0.001   | 98.87             | 98.91       | 35.2      |
| lr            | 0.01    | 97.60             | 97.64       | 34.5      |
| batch_size    | 32      | 98.98             | 98.99       | 49.6      |
| batch_size    | 64      | 98.93             | 98.90       | 40.5      |
| batch_size    | 128     | 98.75             | 98.99       | 31.5      |

### Analyse

On constate que LeNet-5 et CNN simple sont proches en précision, mais avec un compromis fort sur le temps d’entraînement : LeNet-5 est environ 3 fois plus rapide pour une différence de précision final d’environ 0.5%.

Le learning rate 0.001 donne un bon équilibre entre stabilité et vitesse d’apprentissage. La taille de batch, elle, influence plus le temps que la précision.