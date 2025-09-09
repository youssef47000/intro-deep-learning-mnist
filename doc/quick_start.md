# Quick Start

## Project Structure

```
projet_M2_deep_learning/
├── dataset/
│   └── mnist.pkl.gz
├── doc/
│   └── doc.md
├── src/
│   ├── perceptron_pytorch.py
│   ├── perceptron_pytorch_data_auto_layer_optim.py
│   ├── shallow_network.py
│   ├── deep_network.py
│   └── cnn_network.py
├── .gitignore
├── projet.pdf
├── README.md
└── requirements.txt
```

## Installation

### 1. Setup Environment
```bash
python3 -m venv env
source env/bin/activate  # Linux/Mac
# or env\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print('PyTorch OK')"
```

## Run

```bash
# Part 1 - Perceptron Analysis
python src/perceptron_pytorch.py

# Part 2 - Shallow Network
python src/shallow_network.py

# Part 3 - Deep Network  
python src/deep_network.py

# Part 4 - CNN
python src/cnn_network.py
```