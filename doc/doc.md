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

## Résultats expérimentaux

### Test initial
```
Test avec hidden_size=128, lr=0.001, batch_size=64
Epoch  1/20 | Train Acc:  90.18% | Val Acc:  93.78% | Test Acc:  94.31%
Epoch  6/20 | Train Acc:  98.31% | Val Acc:  97.02% | Test Acc:  97.43%
Epoch 11/20 | Train Acc:  99.44% | Val Acc:  97.49% | Test Acc:  97.97%
Epoch 16/20 | Train Acc:  99.83% | Val Acc:  97.67% | Test Acc:  97.94%
Epoch 20/20 | Train Acc:  99.84% | Val Acc:  97.52% | Test Acc:  97.81%

Résultats de base: Val=97.67%, Test=97.81%
```

### Recherche d'hyperparamètres

#### 1. Influence du nombre de neurones cachés
| Hidden Size | Validation Acc | Test Acc |
|-------------|----------------|----------|
| 64          | 97.28%         | 97.71%   |
| 128         | 97.73%         | 97.77%   |
| 256         | 97.88%         | 98.07%   |
| **512**     | **98.14%**     | **98.01%** |

#### 2. Influence du learning rate
| Learning Rate | Validation Acc | Test Acc |
|---------------|----------------|----------|
| 0.0001        | 95.15%         | 95.57%   |
| **0.001**     | **97.80%**     | **98.11%** |
| 0.01          | 96.86%         | 96.93%   |

#### 3. Influence du batch size
| Batch Size | Validation Acc | Test Acc |
|------------|----------------|----------|
| 32         | 97.66%         | 97.77%   |
| **64**     | **97.90%**     | **97.80%** |
| 128        | 97.53%         | 97.57%   |

### Résumé complet des expériences
| Paramètre   | Valeur | Validation | Test    |
|-------------|--------|------------|---------|
| hidden_size | 64     | 97.28%     | 97.71%  |
| hidden_size | 128    | 97.73%     | 97.77%  |
| hidden_size | 256    | 97.88%     | 98.07%  |
| hidden_size | **512** | **98.14%** | **98.01%** |
| lr          | 0.0001 | 95.15%     | 95.57%  |
| lr          | **0.001** | **97.80%** | **98.11%** |
| lr          | 0.01   | 96.86%     | 96.93%  |
| batch_size  | 32     | 97.66%     | 97.77%  |
| batch_size  | **64** | **97.90%** | **97.80%** |
| batch_size  | 128    | 97.53%     | 97.57%  |

## Meilleur résultat
- **Paramètre optimal** : hidden_size = 512
- **Validation accuracy** : 98.14%
- **Test accuracy** : 98.01%

## Analyse et conclusions

J'ai implémenté un MLP avec une couche cachée (784 → hidden_size → 10) utilisant `nn.Linear`, `ReLU`, `CrossEntropyLoss` et `Adam`. La méthodologie consistait à créer un split train/validation (80/20) pour éviter l'overfitting et tester systématiquement trois hyperparamètres : le nombre de neurones cachés [64,128,256,512], le learning rate [0.0001,0.001,0.01], et la batch size [32,64,128]. 

Les résultats montrent que plus de neurones cachés améliorent les performances (64: 97.28% → 512: 98.14%), le learning rate optimal est 0.001 (vs 95.15% pour 0.0001 et 96.86% pour 0.01), et la batch size optimale est 64 (97.90% vs 97.53% pour 128). La configuration optimale (hidden_size=512, lr=0.001, batch_size=64) atteint **98.14%** de précision en validation et **98.01%** sur le test, démontrant l'importance de l'équilibre entre capacité du modèle, vitesse d'apprentissage et stabilité des gradients.

# Part 3: Deep network
# Part 4: CNN
