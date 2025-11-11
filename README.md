# Molecular Property Prediction with GNN

A Graph Neural Network (GNN) implementation for predicting molecular properties using PyTorch Geometric. This project uses Graph Isomorphism Network with Edge attributes (GINE) to predict molecular energies from atomic structures.

## Features

- **Graph Neural Network Architecture**: Multi-layer GINEConv with batch normalization
- **Automatic Bond Inference**: Distance-based molecular bond detection
- **Energy Prediction**: Trained to predict molecular energy values
- **Visualization**: Built-in molecular graph visualization
- **Extensible Design**: Easy to adapt for other molecular properties

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- NumPy
- Matplotlib
- NetworkX
- KaggleHub

## Installation
```bash
# Clone the repository
git clone https://github.com/mishraansh07/MoleculeGNN.git

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the "Predict Molecular Properties" dataset from Kaggle:
- Dataset: `burakhmmtgl/predict-molecular-properties`
- Automatically downloaded via KaggleHub

## Usage

### Training the Model
```python
python molecule_gnn.py
```

### Using the Trained Model
```python
import torch
from molecule_gnn import MoleculeGNN, predict_energy

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoleculeGNN(input_dim=5, hidden_dim=128, num_layers=3, dropout=0.2)
model.load_state_dict(torch.load('molecule_gnn.pt'))
model.to(device)

# Predict energy for a molecule
predicted_energy = predict_energy(model, your_molecule_data, device)
print(f"Predicted Energy: {predicted_energy:.4f}")
```

## Model Architecture

- **Input**: One-hot encoded atom types (H, C, N, O, F)
- **Layers**: 3 GINEConv layers with 128 hidden dimensions
- **Pooling**: Global mean pooling for graph-level representation
- **Output**: Single value (molecular energy)
- **Regularization**: Dropout (0.2) and Batch Normalization

## Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: MSE Loss
- **Batch Size**: 32
- **Epochs**: 800
- **Train/Test Split**: 15000/remainder

## Results

The model achieves competitive performance on molecular energy prediction tasks. Training and evaluation metrics are logged during execution.