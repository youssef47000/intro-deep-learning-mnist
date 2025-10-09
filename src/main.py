"""
main.py - Menu principal pour lancer les différents réseaux
"""

import sys
import os

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "=" * 60)
    print("MNIST - Réseaux de Neurones")
    print("=" * 60)
    print("\n[1] Shallow Network")
    print("[2] Deep Network")
    print("[3] CNN Network")
    print("[0] Quitter\n")
    
    choice = input("Choix: ")
    
    if choice == '1':
        print("\nLancement Shallow Network...\n")
        import shallow_network
        config = {
            'learning_rate': 0.001,
            'hidden_size': 256,
            'batch_size': 64,
            'nb_epochs': 15
        }
        shallow_network.train_and_test_final(config)
    
    elif choice == '2':
        print("\nLancement Deep Network...\n")
        import deep_network
        best_val, final_test, duration, history = deep_network.train_and_evaluate(
            hidden_layers=[256, 128], 
            track_history=True
        )
        if history:
            deep_network.plot_training_curves(history)
        results, total_time = deep_network.hyperparameter_search()
        deep_network.plot_architecture_comparison(results)
        deep_network.plot_hyperparameter_comparison(results)
    
    elif choice == '3':
        print("\nLancement CNN Network...\n")
        import cnn_network
        cnn_network.visualize_sample_images()
        best_val, final_test, duration, history = cnn_network.train_and_evaluate_cnn(
            model_type='lenet5',
            track_history=True,
            nb_epochs=15
        )
        if history:
            cnn_network.plot_training_curves(history)
        results, total_time = cnn_network.cnn_hyperparameter_search()
        cnn_network.plot_cnn_comparison(results)
    
    elif choice == '0':
        print("\nAu revoir!\n")
        return
    
    else:
        print("\nChoix invalide!\n")

if __name__ == "__main__":
    main()