import gzip, torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

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

def train_and_evaluate(hidden_size=128, lr=0.001, batch_size=64, nb_epochs=20, verbose=True, track_history=False):
    start_time = time.time()
    
    if verbose:
        print(f"Test avec hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}")
    
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
    model = ShallowNetwork(hidden_size=hidden_size)
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
        if verbose and (epoch % 5 == 0 or epoch == nb_epochs - 1):
            print(f'Epoch {epoch+1:2d}/{nb_epochs} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Time: {epoch_time:.1f}s')
    
    training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    if verbose:
        print(f"Temps chargement données: {data_time:.2f}s")
        print(f"Temps entraînement: {training_time:.2f}s")
        print(f"Temps total: {total_time:.2f}s")
    
    return best_val_acc, test_acc, total_time, history

def plot_training_curves(history, title="Courbes d'entraînement"):
    """Affiche les courbes d'entraînement"""
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

def plot_hyperparameter_comparison(results):
    """Affiche les graphiques de comparaison des hyperparamètres"""
    # Séparer les résultats par type
    hidden_size_results = [(val, acc, test, time) for param, val, acc, test, time in results if param == 'hidden_size']
    lr_results = [(val, acc, test, time) for param, val, acc, test, time in results if param == 'lr']
    batch_size_results = [(val, acc, test, time) for param, val, acc, test, time in results if param == 'batch_size']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Hidden Size vs Accuracy
    if hidden_size_results:
        sizes, val_accs, test_accs, times = zip(*hidden_size_results)
        ax1.plot(sizes, val_accs, 'o-b', label='Validation', linewidth=2, markersize=6)
        ax1.plot(sizes, test_accs, 's-r', label='Test', linewidth=2, markersize=6)
        ax1.set_xlabel('Nombre de neurones cachés')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Impact du nombre de neurones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([96.5, 98.5])
    
    # 2. Learning Rate vs Accuracy
    if lr_results:
        lrs, val_accs, test_accs, times = zip(*lr_results)
        x_pos = range(len(lrs))
        ax2.plot(x_pos, val_accs, 'o-b', label='Validation', linewidth=2, markersize=6)
        ax2.plot(x_pos, test_accs, 's-r', label='Test', linewidth=2, markersize=6)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Impact du Learning Rate')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{lr:.4f}' for lr in lrs])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([94, 99])
    
    # 3. Batch Size vs Accuracy
    if batch_size_results:
        batches, val_accs, test_accs, times = zip(*batch_size_results)
        ax3.plot(batches, val_accs, 'o-b', label='Validation', linewidth=2, markersize=6)
        ax3.plot(batches, test_accs, 's-r', label='Test', linewidth=2, markersize=6)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Impact de la Batch Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([97, 98.5])
    
    # 4. Temps d'entraînement par paramètre
    all_params = []
    all_times = []
    all_labels = []
    
    for param_type, results_list in [('Hidden Size', hidden_size_results), 
                                     ('Learning Rate', lr_results), 
                                     ('Batch Size', batch_size_results)]:
        if results_list:
            _, _, _, times = zip(*results_list)
            all_times.extend(times)
            all_labels.extend([f'{param_type}\n{val}' for val, _, _, _ in results_list])
    
    if all_times:
        bars = ax4.bar(range(len(all_times)), all_times, 
                      color=['skyblue']*len(hidden_size_results) + 
                            ['lightcoral']*len(lr_results) + 
                            ['lightgreen']*len(batch_size_results))
        ax4.set_xlabel('Paramètres')
        ax4.set_ylabel('Temps (s)')
        ax4.set_title('Temps d\'entraînement par configuration')
        ax4.set_xticks(range(len(all_labels)))
        ax4.set_xticklabels(all_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Analyse des Hyperparamètres - Shallow Network', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def hyperparameter_search():
    print("=== RECHERCHE D'HYPERPARAMÈTRES ===\n")
    
    # Paramètres à tester
    hidden_sizes = [64, 128, 256, 512]
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    
    results = []
    total_experiment_time = time.time()
    
    # Test 1: Influence du nombre de neurones cachés
    print("1. Test de l'influence du NOMBRE DE NEURONES CACHÉS:")
    print("-" * 70)
    section_start = time.time()
    
    for hidden_size in hidden_sizes:
        best_val, final_test, duration, _ = train_and_evaluate(
            hidden_size=hidden_size, 
            lr=0.001, 
            batch_size=64, 
            nb_epochs=15,
            verbose=False
        )
        results.append(('hidden_size', hidden_size, best_val, final_test, duration))
        print(f"Hidden size {hidden_size:3d}: Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s")
    
    section_time = time.time() - section_start
    print(f"Temps total section 1: {section_time:.1f}s\n")
    
    # Test 2: Influence du learning rate
    print("2. Test de l'influence du LEARNING RATE:")
    print("-" * 70)
    section_start = time.time()
    
    for lr in learning_rates:
        best_val, final_test, duration, _ = train_and_evaluate(
            hidden_size=128, 
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
    print("-" * 70)
    section_start = time.time()
    
    for batch_size in batch_sizes:
        best_val, final_test, duration, _ = train_and_evaluate(
            hidden_size=128, 
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
    
    print("=" * 70)
    print("MEILLEUR RÉSULTAT:")
    print(f"Paramètre: {best_result[0]} = {best_result[1]}")
    print(f"Validation accuracy: {best_result[2]:.2f}%")
    print(f"Test accuracy: {best_result[3]:.2f}%")
    print(f"Temps: {best_result[4]:.1f}s")
    print("=" * 70)
    print(f"TEMPS TOTAL DE TOUTES LES EXPÉRIENCES: {total_experiment_duration:.1f}s ({total_experiment_duration/60:.1f} min)")
    print("=" * 70)
    
    return results, total_experiment_duration

if __name__ == '__main__':
    # Test initial avec historique pour visualisation
    print("=== TEST INITIAL AVEC VISUALISATION ===")
    best_val, final_test, duration, history = train_and_evaluate(track_history=True)
    print(f"Résultats de base: Val={best_val:.2f}%, Test={final_test:.2f}% | Durée: {duration:.1f}s\n")
    
    # Affichage des courbes d'entraînement
    if history:
        plot_training_curves(history, "Shallow Network - Configuration par défaut")
    
    # Recherche d'hyperparamètres  
    results, total_time = hyperparameter_search()
    
    # Affichage des comparaisons d'hyperparamètres
    plot_hyperparameter_comparison(results)
    
    print("\n=== RÉSUMÉ DES EXPÉRIENCES ===")
    print("Format: Paramètre | Valeur | Validation | Test | Temps")
    print("-" * 60)
    for param_type, value, val_acc, test_acc, time_taken in results:
        print(f"{param_type:12} | {str(value):6} | {val_acc:8.2f}% | {test_acc:7.2f}% | {time_taken:6.1f}s")
    
    print(f"\nBILAN TEMPOREL:")
    print(f"   • Moyenne par expérience: {total_time/len(results):.1f}s")
    print(f"   • Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")