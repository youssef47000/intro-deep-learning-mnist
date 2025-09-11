# projet_M2_deep_learning# INTRODUCTION TO DEEP LEARNING
    - Amine ITJI (p2018984)
    - Youssef BOUAMAMA (p2306151)


# Part 1: Perceptrons

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

## Implémentation du CNN

J'ai implémenté **deux architectures CNN** adaptées aux images MNIST en utilisant les outils PyTorch :

### Architectures implémentées

#### 1. **LeNet-5 inspiré** (Architecture classique)
```
Structure : 1 -> Conv2d(6) -> MaxPool -> Conv2d(16) -> MaxPool -> FC(120) -> FC(84) -> FC(10)
- Conv1: 1→6 channels, kernel 5x5, padding 2 (28x28 → 28x28)
- MaxPool1: 28x28 → 14x14
- Conv2: 6→16 channels, kernel 5x5 (14x14 → 10x10)  
- MaxPool2: 10x10 → 5x5
- FC1: 16×5×5=400 → 120
- FC2: 120 → 84
- FC3: 84 → 10 classes
```

#### 2. **CNN Simple** (Architecture moderne)
```
Structure : 1 -> Conv2d(32) -> MaxPool -> Conv2d(64) -> MaxPool -> FC(128) -> FC(10)
- Conv1: 1→32 channels, kernel 3x3, padding 1 (28x28 → 28x28)
- MaxPool1: 28x28 → 14x14
- Conv2: 32→64 channels, kernel 3x3, padding 1 (14x14 → 14x14)
- MaxPool2: 14x14 → 7x7
- FC1: 64×7×7=3136 → 128
- FC2: 128 → 10 classes
```

### Transformation cruciale des données
**Point clé** : Conversion du format vectoriel en format image
```python
# Transformation: [N, 784] -> [N, 1, 28, 28]
data_train = data_train.view(-1, 1, 28, 28)
data_test = data_test.view(-1, 1, 28, 28)
```

### Outils PyTorch utilisés
- `nn.Conv2d` pour les couches convolutionnelles
- `nn.MaxPool2d` pour le pooling
- `F.relu` comme fonction d'activation
- `nn.Dropout(0.5)` pour la régularisation
- `nn.CrossEntropyLoss` comme fonction de perte
- `optim.Adam` comme optimiseur

## Méthodologie et justification de l'approche

### Contraintes computationnelles
- **5 époques** pour les tests d'architecture (vs 10-15 pour les autres tests)
- **Approche séquentielle** : test d'un paramètre à la fois
- **8 expériences au total** adaptées à la puissance de calcul disponible
- **Durée réelle** : 4.7 minutes au total, moyenne de 35.0s par expérience

### Choix des architectures testées
- **LeNet-5** : Architecture historique de référence pour MNIST
- **CNN Simple** : Version modernisée avec plus de filtres (32→64 vs 6→16)
- **Justification** : Test de l'impact de la complexité et du nombre de paramètres

## Résultats expérimentaux

### Test initial (LeNet-5, 15 époques)
```
Transformation des données: [N, 784] -> [N, 1, 28, 28]
Train shape: torch.Size([63000, 1, 28, 28]), Test shape: torch.Size([7000, 1, 28, 28])

Epoch  1/15 | Train Acc:  85.52% | Val Acc:  96.24% | Test Acc:  96.91% | Time: 2.9s
Epoch  4/15 | Train Acc:  97.38% | Val Acc:  98.09% | Test Acc:  98.51% | Time: 3.2s
Epoch  7/15 | Train Acc:  98.25% | Val Acc:  98.65% | Test Acc:  98.94% | Time: 3.2s
Epoch 10/15 | Train Acc:  98.57% | Val Acc:  98.71% | Test Acc:  99.07% | Time: 3.2s
Epoch 15/15 | Train Acc:  98.87% | Val Acc:  99.03% | Test Acc:  99.10% | Time: 3.2s

Résultats CNN de base: Val=99.03%, Test=99.10% | Durée: 48.9s
```

### Recherche d'hyperparamètres

#### 1. Influence de l'architecture CNN (Section 1: 77.5s)
| Modèle | Validation Acc | Test Acc | Temps | Commentaire |
|--------|----------------|----------|-------|-------------|
| **CNN Simple** | **99.02%** | **98.94%** | 60.3s | Plus de filtres (32→64) |
| LeNet-5 | 98.36% | 98.77% | 17.1s | Architecture classique (6→16) |

#### 2. Influence du learning rate (Section 2: 97.5s)
| Learning Rate | Validation Acc | Test Acc | Temps |
|---------------|----------------|----------|-------|
| 0.0001 | 97.22% | 97.64% | 33.1s |
| **0.001** | **98.97%** | **98.80%** | 32.7s |
| 0.01 | 98.31% | 98.43% | 31.6s |

#### 3. Influence du batch size (Section 3: 104.7s)
| Batch Size | Validation Acc | Test Acc | Temps |
|------------|----------------|----------|-------|
| 32 | 98.79% | 98.94% | 43.0s |
| **64** | **98.86%** | **98.87%** | 32.3s |
| 128 | 98.69% | 98.83% | 29.3s |

### Résumé complet des expériences CNN
| Paramètre | Valeur | Validation | Test | Temps |
|-----------|--------|------------|------|-------|
| model_type | **simple** | **99.02%** | **98.94%** | 60.3s |
| model_type | lenet5 | 98.36% | 98.77% | 17.1s |
| lr | 0.0001 | 97.22% | 97.64% | 33.1s |
| lr | **0.001** | **98.97%** | **98.80%** | 32.7s |
| lr | 0.01 | 98.31% | 98.43% | 31.6s |
| batch_size | 32 | 98.79% | 98.94% | 43.0s |
| batch_size | **64** | **98.86%** | **98.87%** | 32.3s |
| batch_size | 128 | 98.69% | 98.83% | 29.3s |

## Meilleur résultat CNN
- **Architecture optimale** : CNN Simple
- **Validation accuracy** : 99.02%
- **Test accuracy** : 98.94%
- **Temps d'entraînement** : 60.3s

## Bilan temporel CNN
- **Moyenne par expérience** : 35.0s
- **Temps total** : 279.6s (4.7 minutes)

## Analyse de l'influence de chaque hyperparamètre

### 1. Architecture (Simple vs LeNet-5) - Impact : +0.66%
**Observations clés :**
- **CNN Simple** : 99.02% validation → **Supérieur** grâce à plus de filtres (32→64)
- **LeNet-5** : 98.36% validation → **3.5× plus rapide** (17.1s vs 60.3s)
- **Trade-off** : Performance vs efficacité computationnelle

**Explication** : Le CNN Simple avec plus de filtres capture mieux les features complexes, mais au coût d'un temps de calcul significativement plus élevé.

### 2. η (Learning Rate) - Impact : ±1.75%
- **lr=0.0001** : 97.22% → **Convergence lente**, même avec les convolutions
- **lr=0.001** : 98.97% → **Optimal**, cohérent avec MLP mais impact plus marqué
- **lr=0.01** : 98.31% → **Instabilité**, dégradation plus prononcée qu'avec MLP

**Explication** : Les CNN sont plus sensibles au learning rate que les MLP. L'optimisation est plus complexe avec les convolutions.

### 3. Batch Size - Impact : +0.17%
- **batch=32** : 98.79% → Performance correcte mais plus lent (43.0s)
- **batch=64** : 98.86% → **Optimal**, bon équilibre temps/performance (32.3s)
- **batch=128** : 98.69% → Plus rapide (29.3s) avec légère baisse

**Explication** : Impact similaire aux MLP mais moins prononcé. Les CNN sont plus robustes aux variations de batch size.

## Comparaison avec les réseaux précédents

### Performances finales
| Architecture | Meilleure Validation Acc | Meilleure Test Acc | Temps moyen |
|--------------|-------------------------|-------------------|-------------|
| **Perceptron** | ~92% | ~92% | ~5s |
| **Shallow Network** | 97.94% | 98.14% | 14.8s |
| **Deep Network** | 98.13% | 98.03% | 19.7s |
| **CNN** | **99.02%** | **98.94%** | 35.0s |

### Analyse comparative
- **Gain CNN vs MLP** : +0.89% à +1.08% de précision
- **Surcoût temporel** : +78% par rapport au Deep Network
- **Convergence** : Plus rapide (99% dès epoch 15 vs plateau MLP à 98%)

## Justification de l'approche choisie

### Pourquoi LeNet-5 comme base ?
- **Architecture éprouvée** : Conçue spécifiquement pour MNIST
- **Complexité maîtrisée** : Nombre de paramètres raisonnable
- **Référence historique** : Permet comparaison avec la littérature

### Adaptations apportées
- **CNN Simple** : Modernisation avec plus de filtres
- **Dropout 0.5** : Régularisation pour éviter l'overfitting
- **Adam optimizer** : Plus efficace que SGD pour ce type d'architecture

### Limitations computationnelles prises en compte
- **5 époques** pour tests architecture (compromis temps/qualité)
- **15 époques** maximum pour tests détaillés
- **Pas de data augmentation** : Trade-off simplicité/performance

## Conclusion CNN

Le CNN apporte une **amélioration significative** (+0.89% à +1.08%) par rapport aux MLP sur MNIST, confirmant l'avantage des convolutions pour les données visuelles. L'architecture **CNN Simple** (99.02% validation, 98.94% test) surpasse légèrement **LeNet-5** mais nécessite 3.5× plus de temps de calcul.

**Configuration optimale identifiée** : CNN Simple, lr=0.001, batch_size=64
**Trade-off principal** : Performance vs temps de calcul (CNN Simple 3.5× plus lent que LeNet-5 pour +0.66% de précision)

Les CNN démontrent leur supériorité pour la classification d'images, même sur un dataset relativement simple comme MNIST, justifiant leur adoption malgré le surcoût computationnel.