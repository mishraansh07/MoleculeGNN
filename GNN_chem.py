import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import kagglehub


class MoleculeGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=1, num_layers=3, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        nn_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINEConv(nn_input))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn_hidden))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        num_edges = edge_index.size(1)

        for i, conv in enumerate(self.convs):
            edge_attr = torch.zeros((num_edges, x.size(-1)), device=x.device)
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)


def atom_to_feature(atom_type):

    mapping = {
        'H': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'N': [0, 0, 1, 0, 0],
        'O': [0, 0, 0, 1, 0],
        'F': [0, 0, 0, 0, 1]
    }
    return mapping.get(atom_type, [0, 0, 0, 0, 0])


def compute_bonds(atoms, cutoff=1.6):

    coords = np.array([a["xyz"] for a in atoms])
    edges = []
    for i, j in combinations(range(len(coords)), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < cutoff:
            edges.append([i, j])
            edges.append([j, i])
    return edges


def create_molecular_graph(mol_data):

    atoms = mol_data['atoms']
    edges = compute_bonds(atoms, cutoff=1.6)
    
    x = torch.tensor([atom_to_feature(a['type']) for a in atoms], dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([mol_data['En']], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)


def visualize_molecule(graph_data):
    """Visualize molecular graph structure."""
    G = to_networkx(graph_data)
    plt.figure(figsize=(5, 5))
    nx.draw(G, node_size=400, with_labels=True, node_color="skyblue")
    plt.show()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.view(-1), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Evaluate model on given data loader."""
    model.eval()
    total_mae = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            mae = (pred.view(-1) - batch.y).abs().sum().item()
            total_mae += mae
    
    return total_mae / len(loader.dataset)


def predict_energy(model, mol_data, device):

    atoms = mol_data['atoms']
    edges = compute_bonds(atoms, cutoff=1.6)
    
    x = torch.tensor([atom_to_feature(a['type']) for a in atoms], dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    graph = Data(x=x, edge_index=edge_index).to(device)
    
    model.eval()
    with torch.no_grad():
        batch_index = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        prediction = model(graph.x, graph.edge_index, batch=batch_index)
    
    return prediction.item()


def plot_training_curve(train_losses, save_path=None):
    """Plot and optionally save training loss curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """Main training and evaluation pipeline."""
    
    # Download dataset
    path = kagglehub.dataset_download("burakhmmtgl/predict-molecular-properties")
    print(f"Dataset path: {path}")
    
    # Load data
    data_path = os.path.join(path, 'pubChem_p_00025001_00050000.json')
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Number of molecules: {len(raw_data)}")
    
    # Create graph dataset
    data_list = [create_molecular_graph(mol) for mol in raw_data]
    
    # Split into train and test
    train_loader = DataLoader(data_list[:15000], batch_size=32, shuffle=True)
    test_loader = DataLoader(data_list[15000:], batch_size=32, shuffle=False)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoleculeGNN(input_dim=5, hidden_dim=128, num_layers=3, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    num_epochs = 800
    
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # Evaluate
    mae = evaluate(model, test_loader, device)
    print(f"Test MAE: {mae:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "molecule_gnn.pt")
    print("Model saved successfully as 'molecule_gnn.pt'")
    
    # Plot training curve
    plot_training_curve(train_losses, save_path="training_curve.png")
    
    # Example prediction
    predicted_energy = predict_energy(model, raw_data[0], device)
    print(f"Predicted Energy for first molecule: {predicted_energy:.4f}")


if __name__ == "__main__":
    main()