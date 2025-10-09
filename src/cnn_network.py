import gzip, torch, time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class LeNet5_CNN(nn.Module):
    """Architecture inspirée de LeNet-5 adaptée pour MNIST"""
    def __init__(self, num_classes=10):
        super(LeNet5_CNN, self).__init__()
        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 14x14 -> 10x10
        
        # Couches fully connected
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Pooling et dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolution + ReLU + MaxPool
        x = self.pool(F.relu(self.conv1(x)))    # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))    # 14x14 -> 5x5
        
        # Aplatissement
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class SimpleCNN(nn.Module):
    """CNN simple pour comparaison"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))    # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_and_evaluate_cnn(model_type='lenet5', lr=0.001, batch_size=64, nb_epochs=15, verbose=True, track_history=False):
    start_time = time.time()
    
    if verbose:
        print(f"Test avec model_type={model_type}, lr={lr}, batch_size={batch_size}")
    
    # Chargement des données
    data_start = time.time()
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('../dataset/mnist.pkl.gz'))
    data_time = time.time() - data_start
    
    # TRANSFORMATION CRUCIALE : 784D -> 28x28 images
    # Reshape des données de [N, 784] vers [N, 1, 28, 28]
    data_train = data_train.view(-1, 1, 28, 28)
    data_test = data_test.view(-1, 1, 28, 28)
    
    if verbose:
        print(f"Transformation des données: [N, 784] -> [N, 1, 28, 28]")
        print(f"Train shape: {data_train.shape}, Test shape: {data_test.shape}")
    
    # Conversion des labels one-hot en indices de classe
    train_labels_idx = torch.argmax(label_train, dim=1)
    test_labels_idx = torch.argmax(label_test, dim=1)
    
    # Création train/validation split manuel (80/20)
    n_train = len(data_train)
    n_val = int(0.2 * n_train)
    indices = torch.randperm(n_train)
    
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
    if model_type == 'lenet5':
        model = LeNet5_CNN()
    elif model_type == 'simple':
        model = SimpleCNN()
    else:
        raise ValueError("model_type doit être 'lenet5' ou 'simple'")
        
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

def plot_training_curves(history, title="Courbes d'entraînement CNN"):
    """Affiche les courbes d'entraînement pour le CNN"""
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
    ax1.set_ylim([90, 100])
    
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

def plot_cnn_comparison(results_cnn, results_comparison=None):
    """Compare CNN avec autres modèles si disponible"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Performance des différents modèles CNN
    model_results = [(model, acc, test, time) for param, model, acc, test, time in results_cnn if param == 'model_type']
    
    if model_results:
        models, val_accs, test_accs, times = zip(*model_results)
        x_pos = range(len(models))
        width = 0.35
        
        bars1 = ax1.bar([x - width/2 for x in x_pos], val_accs, width, label='Validation', color='lightblue', alpha=0.8)
        bars2 = ax1.bar([x + width/2 for x in x_pos], test_accs, width, label='Test', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Modèle CNN')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Performance des modèles CNN')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([98, 100])
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Learning Rate Impact
    lr_results = [(lr, acc, test, time) for param, lr, acc, test, time in results_cnn if param == 'lr']
    if lr_results:
        lrs, val_accs, test_accs, times = zip(*lr_results)
        x_pos = range(len(lrs))
        ax2.plot(x_pos, val_accs, 'o-b', label='Validation', linewidth=2, markersize=8)
        ax2.plot(x_pos, test_accs, 's-r', label='Test', linewidth=2, markersize=8)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Impact du Learning Rate (CNN)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{lr:.4f}' for lr in lrs])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([98, 100])
    
    # 3. Batch Size Impact
    batch_results = [(batch, acc, test, time) for param, batch, acc, test, time in results_cnn if param == 'batch_size']
    if batch_results:
        batches, val_accs, test_accs, times = zip(*batch_results)
        ax3.plot(batches, val_accs, 'o-b', label='Validation', linewidth=2, markersize=8)
        ax3.plot(batches, test_accs, 's-r', label='Test', linewidth=2, markersize=8)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Impact de la Batch Size (CNN)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([98.5, 99.5])
    
    # 4. Comparaison globale des architectures (si données disponibles)
    if results_comparison:
        # Placeholder pour comparaison MLP vs CNN
        ax4.text(0.5, 0.5, 'Comparaison MLP vs CNN\n(nécessite données MLP)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Comparaison Architectures')
    else:
        # Temps d'entraînement par configuration CNN
        all_times = [time for _, _, _, _, time in results_cnn]
        all_labels = [f'{param}\n{val}' for param, val, _, _, _ in results_cnn]
        
        bars = ax4.bar(range(len(all_times)), all_times, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Temps (s)')
        ax4.set_title('Temps d\'entraînement CNN')
        ax4.set_xticks(range(len(all_labels)))
        ax4.set_xticklabels(all_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Analyse des performances CNN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_sample_images():
    """Affiche quelques exemples d'images MNIST"""
    print("=== VISUALISATION DES DONNÉES MNIST ===")
    
    # Chargement des données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('../dataset/mnist.pkl.gz'))
    
    # Conversion labels one-hot vers indices
    train_labels_idx = torch.argmax(label_train, dim=1)
    
    # Reshape pour visualisation
    images = data_train[:16].view(-1, 28, 28).numpy()
    labels = train_labels_idx[:16].numpy()
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    fig.suptitle('Échantillons du dataset MNIST (transformation 784D -> 28x28)', fontsize=14)
    
    for i in range(16):
        row, col = i // 8, i % 8
        axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].set_title(f'Label: {labels[i]}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def cnn_hyperparameter_search():
    print("=== RECHERCHE D'HYPERPARAMÈTRES CNN ===\n")
    
    # Paramètres adaptés pour CNN
    model_types = ['simple', 'lenet5']
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    
    results = []
    total_experiment_time = time.time()
    
    # Test 1: Comparaison des architectures CNN
    print("1. Test de différentes ARCHITECTURES CNN:")
    print("-" * 70)
    section_start = time.time()
    
    for model_type in model_types:
        print(f"Début test modèle {model_type}...")
        try:
            best_val, final_test, duration, _ = train_and_evaluate_cnn(
                model_type=model_type,
                lr=0.001,
                batch_size=64,
                nb_epochs=5,  # Réduit encore plus pour debug
                verbose=True  # Activé pour voir où ça bloque
            )
            results.append(('model_type', model_type, best_val, final_test, duration))
            print(f"Modèle {model_type:>7}: Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s")
        except Exception as e:
            print(f"ERREUR avec modèle {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    section_time = time.time() - section_start
    print(f"Temps total section 1: {section_time:.1f}s\n")
    
    # Test 2: Learning rate (avec meilleure architecture)
    print("2. Test de l'influence du LEARNING RATE:")
    print("-" * 70)
    section_start = time.time()
    
    for lr in learning_rates:
        best_val, final_test, duration, _ = train_and_evaluate_cnn(
            model_type='lenet5',
            lr=lr,
            batch_size=64,
            nb_epochs=10,
            verbose=False
        )
        results.append(('lr', lr, best_val, final_test, duration))
        print(f"Learning rate {lr:6.4f}: Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s")
    
    section_time = time.time() - section_start
    print(f"Temps total section 2: {section_time:.1f}s\n")
    
    # Test 3: Batch size
    print("3. Test de l'influence du BATCH SIZE:")
    print("-" * 70)
    section_start = time.time()
    
    for batch_size in batch_sizes:
        best_val, final_test, duration, _ = train_and_evaluate_cnn(
            model_type='lenet5',
            lr=0.001,
            batch_size=batch_size,
            nb_epochs=10,
            verbose=False
        )
        results.append(('batch_size', batch_size, best_val, final_test, duration))
        print(f"Batch size {batch_size:3d}: Val={best_val:5.2f}%, Test={final_test:5.2f}% | Temps: {duration:5.1f}s")
    
    section_time = time.time() - section_start
    print(f"Temps total section 3: {section_time:.1f}s\n")
    
    # Résultats finaux
    best_result = max(results, key=lambda x: x[2])
    total_experiment_duration = time.time() - total_experiment_time
    
    print("=" * 70)
    print("MEILLEUR RÉSULTAT CNN:")
    print(f"Paramètre: {best_result[0]} = {best_result[1]}")
    print(f"Validation accuracy: {best_result[2]:.2f}%")
    print(f"Test accuracy: {best_result[3]:.2f}%")
    print(f"Temps: {best_result[4]:.1f}s")
    print("=" * 70)
    print(f"TEMPS TOTAL CNN: {total_experiment_duration:.1f}s ({total_experiment_duration/60:.1f} min)")
    print("=" * 70)
    
    return results, total_experiment_duration

if __name__ == '__main__':
    # Visualisation des données d'abord
    visualize_sample_images()
    
    # Test initial CNN avec historique
    print("\n=== TEST INITIAL CNN AVEC VISUALISATION ===")
    best_val, final_test, duration, history = train_and_evaluate_cnn(
        model_type='lenet5',
        track_history=True,
        nb_epochs=15
    )
    print(f"Résultats CNN de base: Val={best_val:.2f}%, Test={final_test:.2f}% | Durée: {duration:.1f}s\n")
    
    # Affichage des courbes d'entraînement
    if history:
        plot_training_curves(history, "CNN LeNet-5 - Configuration par défaut")
    
    # Recherche d'hyperparamètres CNN
    results, total_time = cnn_hyperparameter_search()
    
    # Visualisations spécialisées CNN
    plot_cnn_comparison(results)
    
    print("\n=== RÉSUMÉ DES EXPÉRIENCES CNN ===")
    print("Format: Paramètre | Valeur | Validation | Test | Temps")
    print("-" * 60)
    for param_type, value, val_acc, test_acc, time_taken in results:
        print(f"{param_type:12} | {str(value):10} | {val_acc:8.2f}% | {test_acc:7.2f}% | {time_taken:6.1f}s")
    
    print(f"\nBILAN TEMPOREL CNN:")
    print(f"   • Moyenne par expérience: {total_time/len(results):.1f}s")
    print(f"   • Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")