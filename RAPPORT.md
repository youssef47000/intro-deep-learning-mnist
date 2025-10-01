# MNIST - Amine ITJI (p2018984) & Youssef BOUAMAMA (p2306151)

---

## Partie 1 : Analyse du Perceptron

Cette section analyse les dimensions des tenseurs dans le fichier `perceptron_pytorch.py`. Le tenseur des poids **w** a pour dimensions **[784, 10]** où 784 correspond aux pixels de l'image (28×28) et 10 aux classes de sortie. L'élément w[i,j] représente le poids entre le pixel i et la classe j. Le tenseur des biais **b** a pour dimensions **[1, 10]**, soit un vecteur de biais partagé avec un biais par classe.

Le tenseur d'entrée **x** lors de l'entraînement a pour dimensions **[5, 784]** pour un batch_size de 5, représentant 5 images avec leurs 784 pixels. Le tenseur de sortie **y**, calculé par y = torch.mm(x, w) + b, a pour dimensions **[5, 10]**, donnant les scores pour chaque classe pour chaque image. Le tenseur des labels **t** a également pour dimensions **[5, 10]** avec l'encodage one-hot des classes.

Le tenseur du gradient **grad = (t - y)** a pour dimensions **[5, 10]** et représente l'erreur entre prédictions et cibles. Il est utilisé pour mettre à jour les poids selon w += eta * torch.mm(x.T, grad) et b += eta * grad.sum(axis=0).

| Tenseur | Dimensions | Signification |
|---------|------------|---------------|
| w | [784, 10] | Poids entre pixels et classes |
| b | [1, 10] | Biais par classe |
| x | [5, 784] | Batch de 5 images |
| y | [5, 10] | Scores prédits |
| t | [5, 10] | Labels one-hot |
| grad | [5, 10] | Erreur pour mise à jour |

---

## Partie 2 : Réseau Shallow

L'architecture implémentée est un MLP à **une couche cachée** avec la structure Input (784) → Hidden layer (n neurones, ReLU) → Output (10, linéaire). Les outils PyTorch utilisés sont nn.Linear pour les layers, nn.ReLU pour l'activation, nn.CrossEntropyLoss pour la loss, optim.Adam comme optimizer et DataLoader pour la gestion des données.

La méthodologie de validation repose sur un **split 80/20** des données d'entraînement, soit 48 000 exemples pour le training et 12 000 pour la validation. Le shuffle est effectué avec torch.randperm() pour éliminer les biais. Le **critère de sélection est l'accuracy sur validation uniquement**. Le test set de 10 000 images est préservé et utilisé uniquement pour l'évaluation finale.

La stratégie de recherche adopte une **approche séquentielle** testant un hyperparamètre à la fois. Le nombre de neurones est testé avec [64, 128, 256, 512] en fixant lr=0.001 et batch=64. Le learning rate est testé avec [0.0001, 0.001, 0.01] en fixant hidden=256 et batch=64. Le batch size est testé avec [32, 64, 128] en fixant hidden=256 et lr=0.001. Cette approche donne **11 expériences au total** au lieu de 36 pour un grid search complet, soit une **réduction du coût computationnel de 70%**. La durée totale mesurée est de 148.4 secondes avec 15 epochs par test.

Le test initial avec la configuration hidden_size=128, lr=0.001, batch_size=64 montre une convergence rapide dès l'epoch 1 (89.94% en train, 93.52% en validation). À l'epoch 20, le modèle atteint 99.93% en train mais seulement 97.38% en validation, indiquant un **overfitting modéré avec un écart train-validation de 2.55%**. Le résultat final est Val=97.45%, Test=97.61% en 17.8s.

| Epoch | Train Acc | Val Acc | Test Acc | Temps |
|-------|-----------|---------|----------|-------|
| 1 | 89.94% | 93.52% | 94.10% | 0.9s |
| 6 | 98.29% | 97.21% | 97.46% | 0.8s |
| 16 | 99.82% | 97.45% | 98.09% | 0.8s |
| 20 | 99.93% | 97.38% | 97.61% | 0.8s |

![Shallow Network - Configuration par défaut](doc/shallow_network/Figure_1.png)
*Figure 1 : Courbes d'entraînement du shallow network. À gauche, l'évolution des accuracies montre une convergence rapide avec un écart train-validation croissant (overfitting). À droite, la loss d'entraînement décroît de manière monotone.*

L'influence du nombre de neurones montre un **impact de +0.55%**. La configuration avec 64 neurones atteint 97.39% en validation, 128 neurones donne 97.44%, **256 neurones est optimal avec 97.94%**, et 512 neurones redescend à 97.86%. La dégradation au-delà de 256 neurones indique un début d'overfitting. Le temps de calcul augmente de 81% entre 64 et 512 neurones.

| Hidden Size | Val Acc | Test Acc | Temps (s) |
|-------------|---------|----------|-----------|
| 64 | 97.39% | 97.50% | 12.5 |
| 128 | 97.44% | 97.74% | 13.6 |
| **256** | **97.94%** | **98.14%** | **17.5** |
| 512 | 97.86% | 98.06% | 22.7 |

L'influence du learning rate montre un **impact de ±2.32%**, ce qui en fait **l'hyperparamètre le plus critique**. Avec lr=0.0001, la convergence est trop lente et atteint seulement 95.38% en validation après 15 epochs. Avec **lr=0.001, le modèle atteint l'optimal à 97.70%**. Avec lr=0.01, l'instabilité cause des oscillations et la performance descend à 96.81%. L'écart de 2.32 points entre le pire et le meilleur résultat confirme l'importance de ce paramètre.

| Learning Rate | Val Acc | Test Acc | Temps (s) |
|---------------|---------|----------|-----------|
| 0.0001 | 95.38% | 95.79% | 13.5 |
| **0.001** | **97.70%** | **97.84%** | **13.0** |
| 0.01 | 96.81% | 96.69% | 13.3 |

L'influence du batch size montre un **impact de +0.33%**. Le batch de 32 donne les meilleurs résultats (97.77%) mais nécessite 20.0 secondes. Le batch de 64 offre un compromis avec 97.63% en 12.9s. Le batch de 128 est le plus rapide (9.2s) mais la performance descend à 97.44%. Le batch de 32 est **2.2× plus lent** que le batch de 128.

| Batch Size | Val Acc | Test Acc | Temps (s) |
|------------|---------|----------|-----------|
| **32** | **97.77%** | **98.04%** | **20.0** |
| 64 | 97.63% | 97.96% | 12.9 |
| 128 | 97.44% | 97.53% | 9.2 |

![Analyse des Hyperparamètres - Shallow Network](doc/shallow_network/Figure_2.png)
*Figure 2 : Analyse comparative des hyperparamètres. En haut à gauche, l'impact du nombre de neurones montre un optimum à 256. En haut à droite, le learning rate présente un pic clair à 0.001. En bas à gauche, le batch size a un impact modéré. En bas à droite, les temps d'entraînement varient selon les configurations.*

La **configuration optimale** identifiée est hidden_size=256, learning_rate=0.001, batch_size=32, donnant Val=97.94%, Test=98.14%, Temps=17.5s. La hiérarchie d'importance des hyperparamètres est : **(1) Learning rate ±2.32%**, **(2) Nombre de neurones +0.55%**, **(3) Batch size +0.33%**.

---

## Partie 3 : Réseau Profond

L'architecture implémentée utilise nn.Sequential pour construire des MLP avec **au moins deux hidden layers**. Cinq configurations ont été testées : [128→64], [256→128], [512→256] avec 2 layers, et [256→128→64], [512→256→128] avec 3 layers. Toutes les hidden layers utilisent l'activation ReLU. La méthodologie suit la même approche avec 11 expériences et 15 epochs par test, pour une durée totale de 216 secondes (3.6 minutes).

Le test initial avec hidden_layers=[256, 128], lr=0.001, batch_size=64 montre une convergence initiale plus rapide que le shallow (91.22% vs 89.94% à l'epoch 1). Cependant, l'**écart train-validation est plus prononcé** (99.63% vs 97.39% à l'epoch 15), indiquant un **overfitting plus marqué avec un écart de 2.24%** contre 2.55% pour le shallow. Le **meilleur résultat en validation est 98.06%** atteint à l'epoch 8.

| Epoch | Train Acc | Val Acc | Test Acc | Temps |
|-------|-----------|---------|----------|-------|
| 1 | 91.22% | 94.64% | 95.29% | 1.2s |
| 6 | 99.07% | 97.58% | 97.93% | 1.1s |
| 15 | 99.63% | 97.39% | 97.87% | 1.1s |

![Deep Network - Configuration par défaut](doc/deep_network/Figure_1.png)
*Figure 3 : Courbes d'entraînement du deep network. Le modèle converge rapidement mais présente un overfitting similaire au shallow network. La loss décroît de manière stable.*

L'influence de l'architecture montre un **impact de +0.42%**. L'architecture la plus compacte [128→64] atteint 97.71% en validation. L'architecture [256→128] avec 2 layers donne 98.05%. L'architecture **[256→128→64] avec 3 layers est optimale à 98.13%**. L'architecture [512→256] avec plus de paramètres donne 98.03%, et [512→256→128] la plus large donne 97.99%. L'ajout d'une troisième layer n'apporte qu'un **gain marginal de +0.08%** (de 98.05% à 98.13%). L'augmentation de largeur vers 512 neurones augmente le temps de 69% sans amélioration en validation.

| Structure | Val Acc | Test Acc | Temps (s) |
|-----------|---------|----------|-----------|
| 128→64 | 97.71% | 97.79% | 14.8 |
| 256→128 | 98.05% | 98.06% | 17.3 |
| **256→128→64** | **98.13%** | **98.03%** | **19.3** |
| 512→256 | 98.03% | 98.31% | 25.5 |
| 512→256→128 | 97.99% | 98.21% | 29.3 |

![Analyse des Architectures Deep Network](doc/deep_network/Figure_2.png)
*Figure 4 : Performance et temps d'entraînement par architecture. En haut, les barres montrent les accuracies pour chaque configuration. En bas, les scatter plots révèlent qu'il n'y a pas de corrélation claire entre complexité et performance.*

L'influence du learning rate montre un **impact de ±0.78%**, réduit par rapport au shallow network (±2.32%). Le lr=0.001 reste optimal avec 97.88% en validation. L'influence du batch size montre un **impact de +0.20%**, avec batch=32 donnant les meilleurs résultats (97.89%) en 25.0s.

| Learning Rate | Val Acc | Test Acc | Batch Size | Val Acc | Test Acc |
|---------------|---------|----------|------------|---------|----------|
| 0.0001 | 97.10% | 97.23% | **32** | **97.89%** | **98.21%** |
| **0.001** | **97.88%** | **97.73%** | 64 | 97.79% | 98.01% |
| 0.01 | 97.22% | 97.43% | 128 | 97.84% | 97.69% |

![Analyse des Hyperparamètres - Deep Network](doc/deep_network/Figure_3.png)
*Figure 5 : Impact du learning rate et du batch size. Le learning rate optimal reste à 0.001, confirmant la constance de ce paramètre. Le batch size montre un impact faible mais batch=32 reste préférable.*

La **configuration optimale** identifiée est architecture=[256→128→64], learning_rate=0.001, batch_size=32, donnant Val=98.13%, Test=98.03%, Temps=19.3s.

La **comparaison avec le shallow network** montre que le deep network n'améliore pas les performances sur MNIST. Le test accuracy est inférieur de **-0.11%** (98.14% vs 98.03%) avec un **surcoût temporel de +44%** (0.9s vs 1.3s par epoch) et une complexité accrue de 200% en nombre de layers. Ces résultats indiquent que MNIST ne nécessite pas de feature hierarchies profondes.

| Métrique | Shallow | Deep | Différence |
|----------|---------|------|------------|
| Test Acc | 98.14% | 98.03% | -0.11% |
| Temps/epoch | 0.9s | 1.3s | +44% |
| Layers | 1 | 3 | +200% |
| Écart train-val (final) | 2.55% | 2.24% | -0.31% |

---

## Partie 4 : Réseau Convolutif

Les données MNIST sont fournies en vecteurs de dimension 784. Pour utiliser les convolutions, une transformation est nécessaire : `data_train.view(-1, 1, 28, 28)` qui passe du format [N, 784] au format [N, 1, 28, 28] (N échantillons, 1 channel, 28×28 pixels).

Deux architectures CNN ont été implémentées. **LeNet-5 adaptée** utilise une progression graduelle avec Conv1 (1→6 filtres 5×5) + MaxPool, Conv2 (6→16 filtres 5×5) + MaxPool, suivies de 3 fully connected layers (400→120→84→10). **CNN Simple** utilise des filtres 3×3 avec Conv1 (1→32) + MaxPool, Conv2 (32→64) + MaxPool, suivies de 2 fully connected layers (3136→128→10) avec Dropout(0.5). Les composants PyTorch utilisés sont nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Dropout, nn.CrossEntropyLoss et optim.Adam.

Le test initial avec LeNet-5, lr=0.001, batch_size=64 montre des performances remarquables dès l'epoch 1 avec 97.44% en test, bien supérieur aux MLP (~94%). À l'epoch 15, le modèle atteint 99.04% en test avec un **écart train-validation très faible de 0.02%** (98.96% vs 98.94%), indiquant une **excellente généralisation et absence d'overfitting**. Les durées sont : chargement 1.68s, entraînement 55.27s, total 59s.

| Epoch | Train Acc | Val Acc | Test Acc | Temps |
|-------|-----------|---------|----------|-------|
| 1 | 87.09% | 97.16% | 97.44% | 6.2s |
| 4 | 97.51% | 98.45% | 98.53% | 3.6s |
| 10 | 98.54% | 98.82% | 98.73% | 3.4s |
| 15 | 98.96% | 98.94% | 99.04% | 3.4s |

![CNN LeNet-5 - Configuration par défaut](doc/cnn_network/Figure_1.png)
*Figure 6 : Courbes d'entraînement du CNN LeNet-5. L'écart train-validation reste minimal tout au long de l'entraînement (quasi-superposition des courbes rouge et verte), démontrant l'absence d'overfitting. La loss converge rapidement et se stabilise.*

La comparaison des architectures montre que **LeNet-5 est 3.5× plus rapide** que le CNN Simple (19.4s vs 68.7s pour 15 epochs) avec seulement **-0.47% de précision** (98.40% vs 98.87%). LeNet-5 offre donc un meilleur rapport performance/efficacité pour MNIST grâce à sa progression graduelle des filtres (6→16) plus efficace que le doublement (32→64).

| Modèle | Val Acc | Test Acc | Temps (15 epochs) |
|--------|---------|----------|-------------------|
| CNN Simple | 98.71% | 98.87% | 68.7s |
| **LeNet-5** | **98.27%** | **98.40%** | **19.4s** |

L'influence du learning rate montre un **impact de ±1.08%**. Le lr=0.001 est optimal avec 98.87% en validation. L'influence du batch size montre un **impact de +0.23%**. Le batch=32 donne les meilleurs résultats avec 98.98% en validation.

| Learning Rate | Val Acc | Test Acc | Batch Size | Val Acc | Test Acc |
|---------------|---------|----------|------------|---------|----------|
| 0.0001 | 97.29% | 97.79% | **32** | **98.98%** | **98.99%** |
| **0.001** | **98.87%** | **98.91%** | 64 | 98.93% | 98.90% |
| 0.01 | 97.60% | 97.64% | 128 | 98.75% | 98.99% |

![Analyse des performances CNN](doc/cnn_network/Figure_2.png)
*Figure 7 : Analyse complète des CNN. En haut à gauche, comparaison des deux architectures. En haut à droite, impact du learning rate avec pic à 0.001. En bas à gauche, impact faible du batch size. En bas à droite, temps d'entraînement par configuration.*

La **configuration optimale** identifiée est architecture=LeNet-5, learning_rate=0.001, batch_size=32, donnant Val=98.98%, Test=**98.99%**, Temps ~50s.

---

## Analyse comparative

La synthèse des performances montre une progression nette avec l'architecture. Le perceptron atteint ~87% en test. Le shallow network atteint 98.14% avec ~200K paramètres et 0.9s par epoch. Le deep network atteint 98.03% avec ~300K paramètres et 1.3s par epoch. Le **CNN LeNet-5 atteint 99.04%** (meilleur résultat de toutes les expériences) avec seulement **~60K paramètres et 3.7s par epoch**, démontrant une efficacité paramétrique supérieure.

| Architecture | Test Acc | Temps/epoch | Paramètres | Écart train-val |
|--------------|----------|-------------|------------|-----------------|
| Perceptron | ~87% | ~0.3s | ~8K | N/A |
| Shallow Network | 98.14% | ~0.9s | ~200K | 2.55% |
| Deep Network | 98.03% | ~1.3s | ~300K | 2.24% |
| **CNN (LeNet-5)** | **99.04%** | **~3.7s** | **~60K** | **0.02%** |

La comparaison shallow vs deep confirme que **la profondeur n'apporte pas de gain sur MNIST**. Le deep network est moins performant (-0.11%) avec un surcoût temporel de +44% et +50% de paramètres. L'overfitting est légèrement réduit (2.24% vs 2.55% d'écart train-val) mais reste présent. Cette observation indique que MNIST ne bénéficie pas de feature hierarchies profondes. L'ajout de deux layers supplémentaires augmente la complexité du modèle sans améliorer sa capacité de généralisation.

La comparaison MLP vs CNN montre que **le CNN améliore l'accuracy de +0.90%** (98.14% → 99.04%) avec **3× moins de paramètres** (~200K → ~60K). Cette efficacité s'explique par les biais inductifs des convolutions : les poids partagés exploitent l'invariance par translation des images, et la connectivité locale capture les patterns spatiales. Le **CNN présente un overfitting quasi nul** (écart de 0.02%) comparé aux MLP (2.55% et 2.24%), démontrant une bien meilleure capacité de généralisation.

La hiérarchie des hyperparamètres varie selon l'architecture. Pour le shallow network, le learning rate a l'impact le plus important (±2.32%), suivi du nombre de neurones (+0.55%) et du batch size (+0.33%). Pour le deep network, l'impact du learning rate est réduit à ±0.78%, suggérant une meilleure robustesse des architectures plus profondes aux variations de ce paramètre. Pour le CNN, le learning rate reste dominant (±1.08%) avec un impact faible du batch size (+0.23%). Le **learning rate est l'hyperparamètre le plus critique** pour toutes les architectures testées.

Le récapitulatif des configurations optimales montre une constance dans les choix : lr=0.001 et batch=32 sont optimaux pour les trois architectures. Pour le shallow network, 256 neurones suffisent pour atteindre Val=97.94%, Test=98.14%. Pour le deep network, l'architecture [256→128→64] avec 3 layers n'apporte qu'un gain marginal atteignant Val=98.13%, Test=98.03%. Pour le CNN, LeNet-5 atteint les meilleures performances globales avec Val=98.98%, Test=98.99%.

| Partie | Architecture | LR | Batch | Val Acc | Test Acc | Overfitting |
|--------|--------------|-----|-------|---------|----------|-------------|
| 2 - Shallow | 256 neurones | 0.001 | 32 | 97.94% | 98.14% | Modéré (2.55%) |
| 3 - Deep | 256→128→64 | 0.001 | 32 | 98.13% | 98.03% | Modéré (2.24%) |
| 4 - CNN | LeNet-5 | 0.001 | 32 | 98.98% | **98.99%** | Quasi nul (0.02%) |

Les résultats obtenus démontrent que pour MNIST, **l'architecture CNN est supérieure** aux MLP en termes d'accuracy (+0.90%), d'efficacité paramétrique (3× moins de paramètres) et de généralisation (overfitting quasi nul). La profondeur des réseaux MLP n'apporte pas d'avantage sur ce dataset. Le learning rate reste le paramètre le plus critique à optimiser quelle que soit l'architecture. L'approche séquentielle de recherche d'hyperparamètres s'est révélée efficace, permettant d'identifier de bonnes configurations avec une réduction de 70% du coût computationnel par rapport à un grid search exhaustif.