import gzip
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, learning_rate, batch_size, nb_epochs):

        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

        # les hyperparamètres
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.nb_epochs = int(nb_epochs)
        self.hidden_size = hidden_size
        self.model_name = f"MLP_L{learning_rate}_H{hidden_size}_B{batch_size}_E{nb_epochs}"

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):

        # couche cachée
        hidden = self.hidden_layer(x)
        # applique-la non linéarité
        hidden = self.relu(hidden)
        # couche de sortie
        output = self.output_layer(hidden)

        return output


def load_data():
    """Charges les données de mnist.pkl.gz"""

    print("Chargement des données MNIST")
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('../dataset/mnist.pkl.gz'))

    # decoupage 80 % pour l'entrainement et 20% pour validation
    n_train = len(data_train)
    n_val = int(0.2 * n_train)
    indices = torch.randperm(n_train)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_labels_idx = torch.argmax(label_train, dim=1)
    test_labels_idx = torch.argmax(label_test, dim=1)

    X_train = data_train[train_indices]
    y_train = train_labels_idx[train_indices]
    X_val = data_train[val_indices]
    y_val = train_labels_idx[val_indices]
    test_dataset = TensorDataset(data_test, test_labels_idx)

    print(f"Données chargées: Train={X_train.shape}, Val={X_val.shape}, Test={data_test.shape} \n")
    return X_train, y_train, X_val, y_val, test_dataset


def train_model(model, X_train, y_train, X_val, y_val):
    """Entraîne le modèle"""

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model.batch_size, shuffle=False)

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    epoch_times = []

    total_start_time = time.time()

    for epoch in range(model.nb_epochs):
        epoch_start_time = time.time()

        # partie entraînement
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            model.optimizer.zero_grad()
            outputs = model(batch_x)
            loss = model.criterion(outputs, batch_y)
            loss.backward()
            model.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # partie validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = model.criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # ----------------------------------------
        # Affichage des métriques
        # --------------------------------------------
        print(f"Epoch {epoch + 1}/{model.nb_epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}% ")
    print("\n" + "-" * 80)

    # Supprimer la première époque pour éviter que le temps soit faussé à cause du chargement lors du premier entraînement.
    if len(epoch_times) > 1:
        total_training_time = sum(epoch_times[1:]) # total sans epoque 1
    else:
        total_training_time = time.time() - total_start_time

    best_val_acc = max(val_accs)
    best_epoch = val_accs.index(best_val_acc)

    return best_val_acc, best_epoch, train_accs, val_accs, train_losses, val_losses, total_training_time, epoch_times


def test_model(model,test_dataset):
    """ test le model"""

    test_loader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False)

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = model.criterion(outputs, batch_y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()



    test_loss_avg = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total



    return test_acc, test_loss_avg


def train_and_test_final(config):
    """test sur les hyperparamètres choisis """

    print("-" * 80)
    print("ENTRAÎNEMENT FINAL")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # charge données
    X_train, y_train, X_val, y_val,test_dataset = load_data()

    model = MLP(
        input_size=784,
        hidden_size=config['hidden_size'],
        output_size=10,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        nb_epochs=config['nb_epochs']
    )

    # lance l'entrainement
    best_val_acc, best_epoch,train_accs, val_accs, train_losses, val_losses, _, _ = train_model(
        model, X_train, y_train, X_val, y_val
    )

    print(f"\nMeilleure Val Acc: {best_val_acc:.2f}% (époque {best_epoch + 1})")


    # Calculer les moyennes
    avg_train_acc = np.mean(train_accs)
    avg_val_acc = np.mean(val_accs)
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)


    # lance le test
    test_acc, test_loss = test_model(model, test_dataset)

    print("\n" + "-" * 80)
    print("Résultats train et val")
    print("-" * 80)
    print(f"Train Acc (best): {max(train_accs):.2f}% | Val Acc (best): {best_val_acc:.2f}% ")
    print(f"Train Acc (avg): {avg_train_acc:.2f}% | Val Acc (avg): {avg_val_acc:.2f}% ")
    print(f"Train Loss (avg): {avg_train_loss:.4f} | Val Loss (avg): {avg_val_loss:.4f}")
    print("-" * 80)
    print("\n"+ "Résultat du test")
    print(f"Test Loss: {test_loss:.2f} | Test : {test_acc:.2f}%")
    print("-" * 80)


# -------------------------------------------------------------------------------------------------------
# les tests effectués pour trouver les hyperparametres
# -------------------------------------------------------------------------------------------------------

#

# --- TEST 1: LEARNING RATE ---
PARAM_TO_TEST = "learning_rate"
PARAM_NAME = "Learning Rate"
VALUES_TO_TEST = [0.0001, 0.0005, 0.001, 0.005, 0.01]
DEFAULT_CONFIG = {
    'hidden_size': 256,
    'batch_size': 50,
    'nb_epochs': 15
}


# --- TEST 2: HIDDEN SIZE ---
# PARAM_TO_TEST = "hidden_size"
# PARAM_NAME = "Hidden Size"
# VALUES_TO_TEST = [64,128,256,512,1024]
# DEFAULT_CONFIG = {
#     'learning_rate': 0.001,
#     'batch_size': 50,
#     'nb_epochs': 15
# }


# --- TEST 3: BATCH SIZE ---
# PARAM_TO_TEST = "batch_size"
# PARAM_NAME = "Batch Size"
# VALUES_TO_TEST = [16, 32, 64, 128, 256]
# DEFAULT_CONFIG = {
#     'learning_rate': 0.001,
#     'hidden_size': 256,
#     'nb_epochs': 15
# }

# --- TEST 4: NOMBRE D'ÉPOQUES ---
# PARAM_TO_TEST = "nb_epochs"
# PARAM_NAME = "Nombre d'Époques"
# VALUES_TO_TEST = [5, 10, 20, 40, 80]
# DEFAULT_CONFIG = {
#     'learning_rate': 0.001,
#     'hidden_size': 256,
#     'batch_size': 50
# }

# --- TEST UNIQUEMENT ---
config_choisi = {
        'learning_rate': 0.001,
        'hidden_size': 256,
        'batch_size': 64,
        'nb_epochs': 15
    }
# ---------------------------------------------------------------------------------------------------------------------


def run_comparison():
    """lance le test avec les valeurs à tester"""

    print("=" * 70)
    print(f"COMPARAISON: {PARAM_NAME}")
    print(f"   Valeurs testées: {VALUES_TO_TEST}")
    print("=" * 70 + "\n")

    # chargement des données
    X_train, y_train, X_val, y_val,_ = load_data()


    results = []
    all_histories = []
    training_times = []
    all_epoch_times = []

    # test des différentes valeurs pour chaque hyperparamètre
    for i, value in enumerate(VALUES_TO_TEST, 1):
        print(f"[{i}/{len(VALUES_TO_TEST)}] Test: {PARAM_NAME} = {value}")
        print("-" * 70)

        # Créer la configuration
        config = DEFAULT_CONFIG.copy()
        config[PARAM_TO_TEST] = value


        model = MLP(
            input_size=784,
            hidden_size=config.get('hidden_size', 256),
            output_size=10,
            learning_rate=config.get('learning_rate', 0.001),
            batch_size=config.get('batch_size', 64),
            nb_epochs=config.get('nb_epochs', 10)
        )


        best_val_acc, best_epoch, train_hist, val_hist, train_loss_hist, val_loss_hist, total_time, epoch_times = train_model(
            model, X_train, y_train, X_val, y_val
        )


        results.append(best_val_acc)
        all_histories.append(
            (train_hist, val_hist, train_loss_hist, val_loss_hist, f"{PARAM_NAME}={value}", best_epoch))
        training_times.append(total_time)
        all_epoch_times.append(epoch_times)

        print(f"   Meilleur résultat: {best_val_acc:.2f}% (époque {best_epoch + 1})")
        print(f"   Temps total (hors époque 1): {total_time:.2f} secondes")
        avg_time_per_epoch = np.mean(epoch_times[1:]) if len(epoch_times) > 1 else np.mean(epoch_times)
        print(f"   Temps moyen/époque (hors époque 1): {avg_time_per_epoch:.3f} secondes")


        print(f"   Val Acc début: {val_hist[0]:.2f}% | Val Acc finale: {val_hist[-1]:.2f}%")
        print(f"   Val train début: {train_hist[0]:.2f}% | Val train finale: {train_hist[-1]:.2f}%")
        print(f"   Val Loss début: {val_loss_hist[0]:.4f} | Val Loss finale: {val_loss_hist[-1]:.4f}")
        print()


    plot_comparison_with_time(VALUES_TO_TEST, results, all_histories, training_times, all_epoch_times)


def plot_comparison_with_time(values, results, histories, training_times, all_epoch_times):
    """Affiche les graphiques de comparaison avec analyse du temps"""

    fig = plt.figure(figsize=(28, 16))

    # Graphique 1 : analyse de l'accuracy de validation

    ax1 = plt.subplot(3, 3, 1)


    if PARAM_TO_TEST == 'learning_rate':
        x_positions = range(len(values))
        labels = [f'{v:.4f}' if v < 0.01 else f'{v:.2f}' for v in values]
        ax1.plot(x_positions, results, 'bo-', linewidth=2, markersize=10)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax1.plot(values, results, 'bo-', linewidth=2, markersize=10)

    ax1.set_xlabel(PARAM_NAME, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy MAX (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Impact de {PARAM_NAME} (Meilleure Val Acc)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    if PARAM_TO_TEST == 'learning_rate':
        for i, y in enumerate(results):
            ax1.annotate(f'{y:.3f}%', (i, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
    else:
        for x, y in zip(values, results):
            ax1.annotate(f'{y:.3f}%', (x, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')


    best_idx = np.argmax(results)
    if PARAM_TO_TEST == 'learning_rate':
        ax1.scatter(best_idx, results[best_idx],
                    color='red', s=400, zorder=5, marker='*', edgecolors='darkred', linewidth=2)
    else:
        ax1.scatter(values[best_idx], results[best_idx],
                    color='red', s=400, zorder=5, marker='*', edgecolors='darkred', linewidth=2)

    # ----------------------------------------Graphique 2: train vs val-----------------------------------------------

    ax2 = plt.subplot(3, 3, 2)

    distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    colors = distinct_colors[:len(histories)]

    for (train_hist, val_hist, train_loss_hist, val_loss_hist, label, best_epoch), color in zip(histories, colors):
        epochs = range(1, len(train_hist) + 1)

        ax2.plot(epochs, train_hist, linewidth=2, color=color, linestyle='-', alpha=0.7)

        ax2.plot(epochs, val_hist, label=label, linewidth=2.5, color=color, linestyle='--', marker='o', markersize=3,
                 markevery=2)

        ax2.scatter(best_epoch + 1, val_hist[best_epoch], s=150, color=color, marker='*', zorder=5,
                    edgecolors='black', linewidth=1)

    ax2.set_xlabel('Époque', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Train vs Val Accuracy (Train=ligne, Val=pointillés)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right', ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(91, 100)


    # ----------------------------------- Graphique 3 : Convergence Loss----------------------------------------------

    ax3 = plt.subplot(3, 3, 3)

    for (train_hist, val_hist, train_loss_hist, val_loss_hist, label, best_epoch), color in zip(histories, colors):
        epochs = range(1, len(val_loss_hist) + 1)
        ax3.plot(epochs, train_loss_hist, linewidth=2, color=color, linestyle='-', alpha=0.7)
        ax3.plot(epochs, val_loss_hist, label=label, linewidth=2.5, color=color, linestyle='--', marker='s',
                 markersize=3, markevery=2)
        ax3.scatter(best_epoch + 1, val_loss_hist[best_epoch], s=150, color=color, marker='*', zorder=5,
                    edgecolors='black', linewidth=1)

    ax3.set_xlabel('Époque', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Loss (Train=ligne, Val=pointillés, best)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.4)

    # Graphique 4 : validation

    ax4 = plt.subplot(3, 3, 4)

    for (train_hist, val_hist, train_loss_hist, val_loss_hist, label, best_epoch), color in zip(histories, colors):
        epochs = range(1, len(val_hist) + 1)
        ax4.plot(epochs, val_hist, label=label, linewidth=2.5, color=color, marker='o', markersize=4, markevery=2)
        ax4.scatter(best_epoch + 1, val_hist[best_epoch], s=150, color=color, marker='*', zorder=5, edgecolors='black',
                    linewidth=1)

    ax4.set_xlabel('Époque', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Courbes de Validation (meilleure époque)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)

    # -----------------------------------Graphique 5 : Détection Sous-apprentissage------------------------------------

    ax5 = plt.subplot(3, 3, 5)
    train_accs_best = [train_hist[best_epoch] for
                       train_hist, val_hist, train_loss_hist, val_loss_hist, label, best_epoch in histories]
    val_accs_best = [val_hist[best_epoch] for train_hist, val_hist, train_loss_hist, val_loss_hist, label, best_epoch in
                     histories]

    x_pos = np.arange(len(values))
    width = 0.35

    train_colors = ['#2ca02c' if t > 85 else '#ff7f0e' if t > 70 else '#d62728' for t in train_accs_best]
    val_colors = ['#1f77b4' if v > 85 else '#ff7f0e' if v > 70 else '#d62728' for v in val_accs_best]

    bars1 = ax5.bar(x_pos - width / 2, train_accs_best, width, label='Train Acc', color=train_colors, edgecolor='black',
                    linewidth=1)
    bars2 = ax5.bar(x_pos + width / 2, val_accs_best, width, label='Val Acc', color=val_colors, edgecolor='black',
                    linewidth=1)

    ax5.set_xlabel(PARAM_NAME, fontsize=12, fontweight='bold')
    ax5.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Détection Sous-apprentissage (à la meilleure époque)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)

    if PARAM_TO_TEST == 'learning_rate':
        labels = [f'{v:.4f}' if v < 0.01 else f'{v:.2f}' for v in values]
        ax5.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax5.set_xticklabels([str(v) for v in values], rotation=45, ha='right')

    ax5.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Seuil bon (85%)')
    ax5.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Seuil faible (70%)')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)


    # ------------------------------Graphique 7 : Temps Total d'Entraînement------------------------------------------

    ax7 = plt.subplot(3, 3, 7)

    time_colors = []
    for t in training_times:
        if t < np.percentile(training_times, 33):
            time_colors.append('#4CAF50')
        elif t < np.percentile(training_times, 66):
            time_colors.append('#FFC107')
        else:
            time_colors.append('#f44336')

    bars_time = ax7.bar(range(len(values)), training_times, color=time_colors, edgecolor='black', linewidth=1)
    ax7.set_xlabel(PARAM_NAME, fontsize=12, fontweight='bold')
    ax7.set_ylabel('Temps Total (secondes)', fontsize=12, fontweight='bold')
    ax7.set_title('Temps d\'Entraînement Total (hors époque 1)', fontsize=14, fontweight='bold')
    ax7.set_xticks(range(len(values)))

    if PARAM_TO_TEST == 'learning_rate':
        labels = [f'{v:.4f}' if v < 0.01 else f'{v:.2f}' for v in values]
        ax7.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax7.set_xticklabels([str(v) for v in values], rotation=45, ha='right')

    ax7.grid(True, alpha=0.3, axis='y')

    for bar, time_val in zip(bars_time, training_times):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2., height + max(training_times) * 0.01,
                 f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # -----------------------------------------Graphique 8 : Efficacité --------------------------------------*

    ax8 = plt.subplot(3, 3, 8)

    efficiency = [acc / time for acc, time in zip(results, training_times)]

    best_eff_idx = np.argmax(efficiency)
    eff_colors = ['#4CAF50' if i == best_eff_idx else '#2196F3' for i in range(len(efficiency))]

    if PARAM_TO_TEST == 'learning_rate':
        x_positions = range(len(values))
        ax8.bar(x_positions, efficiency, color=eff_colors, edgecolor='black', linewidth=1)
        ax8.set_xticks(x_positions)
        labels = [f'{v:.4f}' if v < 0.01 else f'{v:.2f}' for v in values]
        ax8.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax8.bar(range(len(values)), efficiency, color=eff_colors, edgecolor='black', linewidth=1)
        ax8.set_xticks(range(len(values)))
        ax8.set_xticklabels([str(v) for v in values], rotation=45, ha='right')

    ax8.set_xlabel(PARAM_NAME, fontsize=12, fontweight='bold')
    ax8.set_ylabel('Efficacité (Acc% / seconde)', fontsize=12, fontweight='bold')
    ax8.set_title('Efficacité: Performance / Temps (hors époque 1)', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    if PARAM_TO_TEST == 'learning_rate':
        ax8.scatter(best_eff_idx, efficiency[best_eff_idx] * 1.05,
                    s=300, marker='*', color='gold', edgecolors='darkgoldenrod', linewidth=2, zorder=5)
    else:
        ax8.scatter(best_eff_idx, efficiency[best_eff_idx] * 1.05,
                    s=300, marker='*', color='gold', edgecolors='darkgoldenrod', linewidth=2, zorder=5)

    for i, (bar_x, eff) in enumerate(zip(range(len(values)), efficiency)):
        ax8.text(bar_x, eff + max(efficiency) * 0.01,
                 f'{eff:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --------------------------Graphique 9 : Temps par époque --------------------------------------------------------

    ax9 = plt.subplot(3, 3, 9)

    for i, (epoch_times, color) in enumerate(zip(all_epoch_times, colors[:len(all_epoch_times)])):
        epochs = range(2, len(epoch_times) + 1)
        label = f"{PARAM_NAME}={values[i]}"
        ax9.plot(epochs, epoch_times[1:], label=label, linewidth=2, color=color, alpha=0.7, marker='o', markersize=3,
                 markevery=2)

    ax9.set_xlabel('Époque', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Temps par Époque (secondes)', fontsize=12, fontweight='bold')
    ax9.set_title('Évolution du Temps par Époque (à partir époque 2)', fontsize=14, fontweight='bold')
    ax9.legend(fontsize=8, loc='best', ncol=2)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'comparison_{PARAM_TO_TEST}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    plt.show()


if __name__ == "__main__":

    # Lance l’entraînement avec les hyperparamètres choisis
    train_and_test_final(config_choisi)

    # Fonction pour lancer les tests à la recherche des meilleurs hyperparamètres
    # les tests à choisir ligne 247
    # run_comparison()
