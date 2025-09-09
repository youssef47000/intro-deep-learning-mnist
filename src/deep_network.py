# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Deep Network (MLP avec plusieurs couches cachées) pour MNIST avec visualisations
# ------------------------------------------------------------------------

import gzip, torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class DeepNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_layers=[256, 128], output_size=10):
        super(DeepNetwork, self).__init__()
        
        # Construction dynamique du réseau
        layers = []
        prev_size = input_size
        
        # Ajout des couches cachées
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, output_size))
        
        # Création du modèle séquentiel
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate(hidden_layers=[256, 128], lr=0.001, batch_size=64, nb_epochs=15, verbose=True, track_history=False):
    start_time = time.time()
    
    if verbose:
        print(f"Test avec hidden_layers={hidden_layers}, lr={lr}, batch_size={batch_size}")
    
    # Chargement des données
    data_start = time.time()
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('dataset/mnist.pkl.gz'))
    data_time = time.time() - data_start
    
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
    model = DeepNetwork(hidden_layers=hidden_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Variables pour tracking
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'test_acc': [], 'train_loss': []} if track_history else None
    
    # Timer pour l'entraînement
    training_start = time.time()
    
    # Boucle d'entraînement
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        
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
        train_loss_avg = train_loss / len(train_loader)
        
        # Sauvegarde historique
        if track_history:
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
            history['train_loss'].append(train_loss_avg)
        
        # Mise à jour du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        epoch_time = time.time() - epoch_start
        
        # Affichage des résultats
        if verbose and (epoch % 3 == 0 or epoch == nb_epochs - 1):
            print(f'Epoch {epoch+1:2d}/{nb_epochs} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Time: {epoch_time:.1f}s')
    
    training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    if verbose:
        print(f"Temps chargement données: {data_time:.2f}s")
        print(f"Temps entraînement: {training_time:.2f}s")
        print(f"Temps total: {total_time:.2f}s")
    
    return best_val_acc, test_acc, total_time, history

def plot_training_curves(history, title="Courbes d'entraînement Deep Network"):
    """Affiche les courbes d'entraînement pour le deep network"""
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.plot(epochs, history['test_acc'], 'g-', label='Test Accuracy', linewidth=2)
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy au cours de l\'entraînement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([85, 100])
    
    # Loss
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss d\'entraînement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_architecture_comparison(results):
    """Affiche la comparaison des différentes architectures"""
    arch_results = [(val, acc, test, time) for param, val, acc, test, time in results if param == 'architecture']
    
    if not arch_results:
        return
    
    architectures, val_accs, test_accs, times = zip(*arch_results)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy par architecture
    x_pos = range(len(architectures))
    width = 0.35
    
    bars1 = ax1.bar([x - width/2 for x in x_pos], val_accs, width, label='Validation', color='skyblue', alpha=0.8)
    bars2 = ax1.bar([x + width/2 for x in x_pos], test_accs, width, label='Test', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Performance par Architecture')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'784→{arch}→10' for arch in architectures], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([97, 98.5])
    
    # Ajouter les valeurs sur les barres
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Temps d'entraînement par architecture
    bars = ax2.bar(x_pos, times, color='lightgreen', alpha=0.8)
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Temps (s)')
    ax2.set_title('Temps d\'entraînement par Architecture')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{arch}' for arch in architectures], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 3. Relation Complexité vs Performance
    complexities = [sum(map(int, arch.split('->'))) for arch in architectures]  # Somme des neurones
    ax3.scatter(complexities, val_accs, s=100, c='blue', alpha=0.7, label='Validation')
    ax3.scatter(complexities, test_accs, s=100, c='red', alpha=0.7, label='Test')
    
    # Ajouter les labels des architectures
    for i, arch in enumerate(architectures):
        ax3.annotate(arch, (complexities[i], val_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Complexité (somme des neurones)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Complexité vs Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relation Temps vs Performance
    ax4.scatter(times, val_accs, s=100, c='blue', alpha=0.7, label='Validation')
    ax4.scatter(times, test_accs, s=100, c='red', alpha=0.7, label='Test')
    
    for i, arch in enumerate(architectures):
        ax4.annotate(arch, (times[i], val_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Temps d\'entraînement (s)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Temps vs Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Analyse des Architectures Deep Network', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_comparison(results):
    """Affiche les graphiques de comparaison des hyperparamètres pour deep network"""
    # Séparer les résultats par type
    lr_results = [(val, acc, test, time) for param, val, acc, test, time in results if param == 'lr']
    batch_size_results = [(val, acc, test, time) for param, val, acc, test, time in results if param == 'batch_size']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Learning Rate vs Accuracy
    if lr_results:
        lrs, val_accs, test_accs, times = zip(*lr_results)
        x_pos = range(len(lrs))
        ax1.plot(x_pos, val_accs, 'o-b', label='Validation', linewidth=2, markersize=8)
        ax1.plot(x_pos, test_accs, 's-r', label='Test', linewidth=2, markersize=8)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Impact du Learning Rate')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{lr:.4f}' for lr in lrs])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([96.5, 98.5])
    
    # 2. Batch Size vs Accuracy
    if batch_size_results:
        batches, val_accs, test_accs, times = zip(*batch_size_results)
        ax2.plot(batches, val_accs, 'o-b', label='Validation', linewidth=2, markersize=8)
        ax2.plot(batches, test_accs, 's-r', label='Test', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Impact de la Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([97, 98.5])
    
    # 3. Learning Rate vs Temps
    if lr_results:
        lrs, _, _, times = zip(*lr_results)
        bars = ax3.bar(range(len(lrs)), times, color=['lightblue', 'blue', 'darkblue'])
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('Temps (s)')
        ax3.set_title('Temps d\'entraînement par Learning Rate')
        ax3.set_xticks(range(len(lrs)))
        ax3.set_xticklabels([f'{lr:.4f}' for lr in lrs])
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}s', ha='center', va='bottom')
    
    # 4. Batch Size vs Temps
    if batch_size_results:
        batches, _, _, times = zip(*batch_size_results)
        bars = ax4.bar(range(len(batches)), times, color=['lightgreen', 'green', 'darkgreen'])
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Temps (s)')
        ax4.set_title('Temps d\'entraînement par Batch Size')
        ax4.set_xticks(range(len(batches)))
        ax4.set_xticklabels([str(batch) for batch in batches])
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}s', ha='center', va='bottom')
    
    plt.suptitle('Analyse des Hyperparamètres - Deep Network', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def hyperparameter_search():
    print("=== RECHERCHE D'HYPERPARAMÈTRES DEEP NETWORK ===\n")
    
    # Paramètres à tester (réduits pour la faisabilité computationnelle)
    architectures = [
        [128, 64],        # 2 couches
        [256, 128],       # 2 couches
        [512, 256],       # 2 couches
        [256, 128, 64],   # 3 couches
        [512, 256, 128]   # 3 couches
    ]
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    
    results = []
    total_experiment_time = time.time()
    
    # Test 1: Influence de l'architecture (nombre et taille des couches)
    print("1. Test de l'influence de l'ARCHITECTURE:")
    print("-" * 80)
    section_start = time.time()
    
    for i, arch in enumerate(architectures):
        arch_str = "->".join(map(str, arch))
        print(f"Architecture {i+1}: 784 -> {arch_str} -> 10")
        best_val, final_test, duration, _ = train_and_evaluate(
            hidden_layers=arch, 
            lr=0.001, 
            batch_size=64, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('architecture', arch_str, best_val, final_test, duration))
        print(f"  Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s\n")
    
    section_time = time.time() - section_start
    print(f"Temps total section 1: {section_time:.1f}s\n")
    
    # Test 2: Influence du learning rate (avec meilleure architecture)
    print("2. Test de l'influence du LEARNING RATE:")
    print("-" * 80)
    section_start = time.time()
    
    for lr in learning_rates:
        best_val, final_test, duration, _ = train_and_evaluate(
            hidden_layers=[256, 128], 
            lr=lr, 
            batch_size=64, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('lr', lr, best_val, final_test, duration))
        print(f"Learning rate {lr:6.4f}: Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s")
    
    section_time = time.time() - section_start
    print(f"Temps total section 2: {section_time:.1f}s\n")
    
    # Test 3: Influence du batch size
    print("3. Test de l'influence du BATCH SIZE:")
    print("-" * 80)
    section_start = time.time()
    
    for batch_size in batch_sizes:
        best_val, final_test, duration, _ = train_and_evaluate(
            hidden_layers=[256, 128], 
            lr=0.001, 
            batch_size=batch_size, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('batch_size', batch_size, best_val, final_test, duration))
        print(f"Batch size {batch_size:3d}: Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s")
    
    section_time = time.time() - section_start
    print(f"Temps total section 3: {section_time:.1f}s\n")
    
    # Trouver les meilleurs paramètres
    best_result = max(results, key=lambda x: x[2])  # Max validation accuracy
    total_experiment_duration = time.time() - total_experiment_time
    
    print("=" * 80)
    print("MEILLEUR RÉSULTAT:")
    print(f"Paramètre: {best_result[0]} = {best_result[1]}")
    print(f"Validation accuracy: {best_result[2]:.2f}%")
    print(f"Test accuracy: {best_result[3]:.2f}%")
    print(f"Temps: {best_result[4]:.1f}s")
    print("=" * 80)
    print(f"TEMPS TOTAL DE TOUTES LES EXPÉRIENCES: {total_experiment_duration:.1f}s ({total_experiment_duration/60:.1f} min)")
    print("=" * 80)
    
    return results, total_experiment_duration

if __name__ == '__main__':
    # Test initial avec historique pour visualisation
    print("=== TEST INITIAL DEEP NETWORK AVEC VISUALISATION ===")
    best_val, final_test, duration, history = train_and_evaluate(
        hidden_layers=[256, 128], 
        track_history=True
    )
    print(f"Résultats de base: Val={best_val:.2f}%, Test={final_test:.2f}% | Durée: {duration:.1f}s\n")
    
    # Affichage des courbes d'entraînement
    if history:
        plot_training_curves(history, "Deep Network - Configuration par défaut")
    
    # Recherche d'hyperparamètres  
    results, total_time = hyperparameter_search()
    
    # Affichage des visualisations spécialisées
    plot_architecture_comparison(results)
    plot_hyperparameter_comparison(results)
    
    print("\n=== RÉSUMÉ DES EXPÉRIENCES DEEP NETWORK ===")
    print("Format: Paramètre | Valeur | Validation | Test | Temps")
    print("-" * 70)
    for param_type, value, val_acc, test_acc, time_taken in results:
        print(f"{param_type:12} | {str(value):15} | {val_acc:8.2f}% | {test_acc:7.2f}% | {time_taken:6.1f}s")
    
    print(f"\nBILAN TEMPOREL DEEP NETWORK:")
    print(f"   • Moyenne par expérience: {total_time/len(results):.1f}s")
    print(f"   • Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")