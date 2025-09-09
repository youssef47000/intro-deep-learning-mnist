# INTRODUCTION TO DEEP LEARNING
    - Amine ITJI (p2018984)


# Part 1: Perceptron

## Indicate and explain the size of each tensor of the provided file perceptron_pytorch.py

tensors: w (weight), b (bias), x (input), y (output), t (target labels), grad (gradient).


## Weight tensor

```python
w = torch.empty((data_train.shape[1],label_train.shape[1]),dtype=torch.float)
```
    - data_train.shape[1] = 784 (=28*28)
    - label_train.shape[1] = 10 (output classes: 0-9, example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [784,10]

## Bias tensor 

```python
b = torch.empty((1,label_train.shape[1]),dtype=torch.float)
```
    - 1
    - label_train.shape[1] = 10

    Size => [1,10]


## Input batch tensor 

```python
x = data_train[indices[i:i+batch_size]]
```
    - batch_size = 5
    - nb_data_train = 784

    Size => [5,784]

## Output tensor 

```python
y = torch.mm(x,w)+b
```

    - batch_size = 5
    - output classes size = 10 (example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [5,10]

## Target labels tensor 

```python
t = label_train[indices[i:i+batch_size]]
```

    - batch_size = 5
    - output classes size = 10 (example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [5,10]

## Gradient tensor 

```python
grad = (t-y)
```

    - batch_size = 5
    - output classes size = 10 (example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [5,10]

# Part 2: Shallow network

## Implémentation du shallow network

J'ai implémenté un MLP avec **une seule couche cachée** et une couche de sortie linéaire en utilisant les outils PyTorch :

**Architecture :** 784 (input) → hidden_size (ReLU) → 10 (linear output)

**Outils PyTorch utilisés :**
- `nn.Linear` pour les couches fully-connected
- `nn.ReLU` comme fonction d'activation pour la couche cachée
- `nn.CrossEntropyLoss` comme fonction de perte
- `optim.Adam` comme optimiseur
- `DataLoader` pour le chargement par batch

## Méthodologie précise pour trouver les hyperparamètres

### Création du dataset de validation (évitement de l'overfitting)
- **Split train/validation** : 80% / 20% des données d'entraînement originales  
- **Mélange aléatoire** avec `torch.randperm()` pour éviter les biais
- **Critère de sélection** : validation accuracy (pas test accuracy) pour éviter l'overfitting
- **Poids aléatoires** : initialisation par défaut PyTorch (Xavier/Glorot normale)

### Stratégie de test des hyperparamètres
**Tests systématiques d'un paramètre à la fois** (approche one-factor-at-a-time) :
1. **η (learning rate)** : [0.0001, 0.001, 0.01] avec hidden=128, batch=64
2. **Nombre de neurones cachés** : [64, 128, 256, 512] avec lr=0.001, batch=64  
3. **Batch size** : [32, 64, 128] avec hidden=128, lr=0.001

### Justification du nombre d'expériences adapté à la puissance computationnelle
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
- **lr=0.0001** : 95.38% → **Apprentissage trop lent**, convergence insuffisante en 15 époques
- **lr=0.001** : 97.70% → **Optimal**, bon équilibre vitesse/stabilité  
- **lr=0.01** : 96.81% → **Instabilité**, oscillations autour du minimum, dépassements

**Explication** : Le learning rate contrôle la taille des pas dans l'espace des poids. Trop faible → sous-apprentissage, trop élevé → instabilité.

### 2. Nombre de neurones cachés - Impact : +0.55%
- **64 neurones** : 97.39% → **Capacité limitée** du modèle
- **256 neurones** : 97.94% → **Capacité optimale** pour MNIST
- **512 neurones** : 97.86% → **Léger overfitting**, performance légèrement dégradée

**Explication** : 256 neurones offrent le bon équilibre capacité/généralisation. Au-delà, on observe un début d'overfitting sur ce dataset.

### 3. Batch Size - Impact : +0.33%
- **batch=32** : 97.77% → **Optimal**, bon équilibre bruit/exploration mais plus lent (20.0s)
- **batch=64** : 97.63% → **Compromis** performance/vitesse (12.9s)
- **batch=128** : 97.44% → **Plus rapide** (9.2s) mais gradients trop lisses

**Explication** : Petit batch → gradients bruités (meilleure exploration) mais coût computationnel plus élevé. Grand batch → convergence plus rapide mais risque de minima locaux.

## Conclusion

La configuration optimale trouvée est **hidden_size=256, lr=0.001, batch_size=32** qui pourrait théoriquement donner les meilleures performances, bien que le meilleur résultat individuel soit obtenu avec hidden_size=256. Le learning rate reste l'hyperparamètre le plus critique (±2.32% d'impact), suivi du nombre de neurones cachés (+0.55%) puis de la batch size (+0.33%).


# Part 3: Deep network

## Implémentation du deep network

J'ai implémenté un MLP avec **au moins deux couches cachées** en utilisant les outils PyTorch :

**Architecture flexible :** Construction dynamique avec `nn.Sequential` permettant de spécifier une liste de tailles de couches cachées.

**Outils PyTorch utilisés :**
- `nn.Sequential` pour la construction dynamique du réseau
- `nn.Linear` pour les couches fully-connected
- `nn.ReLU` comme fonction d'activation entre les couches cachées
- `nn.CrossEntropyLoss` comme fonction de perte
- `optim.Adam` comme optimiseur
- `DataLoader` pour le chargement par batch

## Méthodologie pour trouver les hyperparamètres

### Stratégie adaptée à la puissance computationnelle
**Choix : 11 expériences au total** (même approche que shallow network)
- **15 époques par test** pour équilibrer temps/performance
- **Tests séquentiels** d'un paramètre à la fois
- **Architectures représentatives** : 5 architectures (2-3 couches), 3 learning rates, 3 batch sizes
- **Durée réelle mesurée** : 3.6 minutes au total, moyenne de 19.7s par expérience

### Justification des tests choisis
- **Architecture** : Test principal car c'est la nouveauté par rapport au shallow network
- **Learning rate** et **batch size** : Reprise des gammes optimales du shallow network
- **Contrainte temporelle** : 3.6 minutes acceptable pour cette étude comparative

## Résultats expérimentaux

### Test initial
```
Test avec hidden_layers=[256, 128], lr=0.001, batch_size=64
Epoch  1/15 | Train Acc:  91.22% | Val Acc:  94.64% | Test Acc:  95.29% | Time: 1.1s
Epoch  4/15 | Train Acc:  98.18% | Val Acc:  97.22% | Test Acc:  97.86% | Time: 1.1s
Epoch  7/15 | Train Acc:  99.19% | Val Acc:  97.76% | Test Acc:  97.80% | Time: 1.1s
Epoch 10/15 | Train Acc:  99.48% | Val Acc:  97.75% | Test Acc:  97.97% | Time: 1.1s
Epoch 13/15 | Train Acc:  99.57% | Val Acc:  98.06% | Test Acc:  98.07% | Time: 1.1s
Epoch 15/15 | Train Acc:  99.63% | Val Acc:  97.39% | Test Acc:  97.87% | Time: 1.1s

Résultats de base: Val=98.06%, Test=97.87% | Durée: 17.3s
```

### Recherche d'hyperparamètres

#### 1. Influence de l'architecture (Section 1: 106.4s)
| Architecture | Structure | Validation Acc | Test Acc | Temps |
|--------------|-----------|----------------|----------|-------|
| 1 | 784 → 128→64 → 10 | 97.71% | 97.79% | 14.8s |
| 2 | 784 → 256→128 → 10 | 98.05% | 98.06% | 17.3s |
| 3 | 784 → 512→256 → 10 | 98.03% | 98.31% | 25.5s |
| **4** | **784 → 256→128→64 → 10** | **98.13%** | **98.03%** | **19.3s** |
| 5 | 784 → 512→256→128 → 10 | 97.99% | 98.21% | 29.3s |

#### 2. Influence du learning rate (Section 2: 52.4s)
| Learning Rate | Validation Acc | Test Acc | Temps |
|---------------|----------------|----------|-------|
| 0.0001 | 97.10% | 97.23% | 17.1s |
| **0.001** | **97.88%** | **97.73%** | **17.4s** |
| 0.01 | 97.22% | 97.43% | 17.9s |

#### 3. Influence du batch size (Section 3: 57.6s)
| Batch Size | Validation Acc | Test Acc | Temps |
|------------|----------------|----------|-------|
| **32** | **97.89%** | **98.21%** | 28.1s |
| 64 | 97.79% | 98.01% | 17.0s |
| 128 | 97.84% | 97.69% | 12.4s |

### Résumé complet des expériences
| Paramètre | Valeur | Validation | Test | Temps |
|-----------|--------|------------|------|-------|
| architecture | 128→64 | 97.71% | 97.79% | 14.8s |
| architecture | 256→128 | 98.05% | 98.06% | 17.3s |
| architecture | 512→256 | 98.03% | 98.31% | 25.5s |
| architecture | **256→128→64** | **98.13%** | **98.03%** | **19.3s** |
| architecture | 512→256→128 | 97.99% | 98.21% | 29.3s |
| lr | 0.0001 | 97.10% | 97.23% | 17.1s |
| lr | **0.001** | **97.88%** | **97.73%** | **17.4s** |
| lr | 0.01 | 97.22% | 97.43% | 17.9s |
| batch_size | **32** | **97.89%** | **98.21%** | 28.1s |
| batch_size | 64 | 97.79% | 98.01% | 17.0s |
| batch_size | 128 | 97.84% | 97.69% | 12.4s |

## Meilleur résultat
- **Architecture optimale** : 256→128→64 (3 couches cachées)
- **Validation accuracy** : 98.13%
- **Test accuracy** : 98.03%
- **Temps d'entraînement** : 19.3s

## Bilan temporel
- **Moyenne par expérience** : 19.7s
- **Temps total** : 216.4s (3.6 minutes)

## Analyse de l'influence de chaque hyperparamètre

### 1. Architecture (nombre et taille des couches cachées) - Impact : +0.42%
**Observations clés :**
- **2 couches vs 3 couches** : Pas de différence significative (98.05% vs 98.13%)
- **Taille optimale** : 256→128→64 légèrement supérieur à 256→128
- **Réseaux très profonds** : 512→256→128 ne surpasse pas les architectures plus simples (97.99%)
- **Coût computationnel** : Corrélation directe avec la complexité (14.8s à 29.3s)

**Explication** : Pour MNIST, l'ajout de couches n'apporte qu'un gain marginal. La profondeur supplémentaire n'est pas cruciale pour ce dataset relativement simple.

### 2. η (Learning Rate) - Impact : ±0.78%
- **lr=0.0001** : 97.10% → **Apprentissage trop lent** même avec plus de couches
- **lr=0.001** : 97.88% → **Optimal**, cohérent avec shallow network
- **lr=0.01** : 97.22% → **Instabilité** malgré la profondeur du réseau

**Explication** : Le learning rate optimal reste similaire au shallow network. La profondeur n'affecte pas fondamentalement ce hyperparamètre.

### 3. Batch Size - Impact : +0.20%
- **batch=32** : 97.89% → **Optimal** mais plus coûteux (28.1s)
- **batch=64** : 97.79% → **Bon compromis** temps/performance (17.0s)
- **batch=128** : 97.84% → **Plus rapide** (12.4s) avec performance correcte

**Explication** : Tendance similaire au shallow network, avec un léger avantage pour les petits batchs.

## Comparaison avec le shallow network

### Performances
- **Shallow network (Part 2)** : 98.14% test (meilleur cas)
- **Deep network (Part 3)** : 98.03% test (meilleur cas)
- **Différence** : -0.11% (non significative)

### Temps de calcul
- **Shallow network** : 14.8s par expérience en moyenne
- **Deep network** : 19.7s par expérience en moyenne
- **Surcoût** : +33% de temps pour un gain négligeable

## Conclusion

Le deep network n'apporte **pas d'amélioration significative** par rapport au shallow network sur MNIST. La meilleure architecture (256→128→64) atteint 98.13% en validation et 98.03% sur le test, soit des performances comparables au shallow network (98.14% test) mais avec un coût computationnel supérieur (+33% de temps).


















# Part 4: CNN
