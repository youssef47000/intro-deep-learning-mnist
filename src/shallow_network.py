# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Shallow Network (MLP avec 1 couche cachée) pour MNIST
# ------------------------------------------------------------------------

import gzip, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ShallowNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(ShallowNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def train_and_evaluate(hidden_size=128, lr=0.001, batch_size=64, nb_epochs=20, verbose=True):
    if verbose:
        print(f"Test avec hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}")
    
    # Chargement des données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('dataset/mnist.pkl.gz'))
    
    # Conversion des labels one-hot en indices de classe pour CrossEntropyLoss
    train_labels_idx = torch.argmax(label_train, dim=1)
    test_labels_idx = torch.argmax(label_test, dim=1)
    
    # Création train/validation split manuel (80/20)
    n_train = len(data_train)
    n_val = int(0.2 * n_train)
    
    # Mélange des indices
    indices = torch.randperm(n_train)
    
    # Split des données
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train = data_train[train_indices]
    y_train = train_labels_idx[train_indices]
    X_val = data_train[val_indices]
    y_val = train_labels_idx[val_indices]
    
    # Création des DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(data_test, test_labels_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialisation du modèle
    model = ShallowNetwork(hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Variables pour tracking
    best_val_acc = 0.0
    
    # Boucle d'entraînement
    for epoch in range(nb_epochs):
        # PHASE D'ENTRAÎNEMENT
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # PHASE DE VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # PHASE DE TEST
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        # Calcul des accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        test_acc = 100 * test_correct / test_total
        
        # Mise à jour du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Affichage des résultats
        if verbose and (epoch % 5 == 0 or epoch == nb_epochs - 1):
            print(f'Epoch {epoch+1:2d}/{nb_epochs} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | Test Acc: {test_acc:6.2f}%')
    
    return best_val_acc, test_acc

def hyperparameter_search():
    print("=== RECHERCHE D'HYPERPARAMÈTRES ===\n")
    
    # Paramètres à tester
    hidden_sizes = [64, 128, 256, 512]
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    
    results = []
    
    # Test 1: Influence du nombre de neurones cachés
    print("1. Test de l'influence du NOMBRE DE NEURONES CACHÉS:")
    print("-" * 60)
    for hidden_size in hidden_sizes:
        best_val, final_test = train_and_evaluate(
            hidden_size=hidden_size, 
            lr=0.001, 
            batch_size=64, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('hidden_size', hidden_size, best_val, final_test))
        print(f"Hidden size {hidden_size:3d}: Val={best_val:5.2f}%, Test={final_test:5.2f}%")
    print()
    
    # Test 2: Influence du learning rate
    print("2. Test de l'influence du LEARNING RATE:")
    print("-" * 60)
    for lr in learning_rates:
        best_val, final_test = train_and_evaluate(
            hidden_size=128, 
            lr=lr, 
            batch_size=64, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('lr', lr, best_val, final_test))
        print(f"Learning rate {lr:6.4f}: Val={best_val:5.2f}%, Test={final_test:5.2f}%")
    print()
    
    # Test 3: Influence du batch size
    print("3. Test de l'influence du BATCH SIZE:")
    print("-" * 60)
    for batch_size in batch_sizes:
        best_val, final_test = train_and_evaluate(
            hidden_size=128, 
            lr=0.001, 
            batch_size=batch_size, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('batch_size', batch_size, best_val, final_test))
        print(f"Batch size {batch_size:3d}: Val={best_val:5.2f}%, Test={final_test:5.2f}%")
    print()
    
    # Trouver les meilleurs paramètres
    best_result = max(results, key=lambda x: x[2])  # Max validation accuracy
    print("=" * 60)
    print("MEILLEUR RÉSULTAT:")
    print(f"Paramètre: {best_result[0]} = {best_result[1]}")
    print(f"Validation accuracy: {best_result[2]:.2f}%")
    print(f"Test accuracy: {best_result[3]:.2f}%")
    print("=" * 60)
    
    return results

if __name__ == '__main__':
    # Test initial
    print("=== TEST INITIAL ===")
    best_val, final_test = train_and_evaluate()
    print(f"Résultats de base: Val={best_val:.2f}%, Test={final_test:.2f}%\n")
    
    # Recherche d'hyperparamètres  
    results = hyperparameter_search()
    
    print("\n=== RÉSUMÉ DES EXPÉRIENCES ===")
    print("Format: Paramètre | Valeur | Validation | Test")
    print("-" * 50)
    for param_type, value, val_acc, test_acc in results:
        print(f"{param_type:12} | {str(value):6} | {val_acc:8.2f}% | {test_acc:7.2f}%")