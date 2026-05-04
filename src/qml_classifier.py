import pennylane as qml
from pennylane import numpy as np
import torch

# We use a simple 2-qubit system for feature classification
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    inputs: Data features (normalized angles)
    weights: Trainable parameters for the quantum circuit
    """
    # Angle encoding
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        
    # Variational layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measurement (expectation value of PauliZ on the first qubit)
    return qml.expval(qml.PauliZ(0))

class QMLClassifier:
    def __init__(self, layers=2):
        self.layers = layers
        # Weights shape: (layers, n_qubits, 3) for StronglyEntanglingLayers
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.layers, n_wires=n_qubits)
        self.weights = torch.nn.Parameter(torch.rand(shape, requires_grad=True))
        
        self.optimizer = torch.optim.Adam([self.weights], lr=0.1)
        self.loss_fn = torch.nn.MSELoss()
        
    def predict(self, x):
        """
        x: tensor of shape (batch_size, 2)
        Returns: expectation values mapped between 0 and 1
        """
        preds = []
        for xi in x:
            pred = quantum_circuit(xi, self.weights)
            preds.append(pred)
        preds = torch.stack(preds)
        return (preds + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
        
    def update(self, x, y):
        self.optimizer.zero_grad()
        predictions = self.predict(x)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
