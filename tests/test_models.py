import pytest
import torch

from src.dqn_agent import DQN, DQNAgent
from src.qml_classifier import QMLClassifier


def test_dqn_forward():
    """Test DQN forward pass output shape"""
    num_songs = 200
    state_length = 3

    model = DQN(num_songs, state_length)

    # Batch size = 2
    dummy_input = torch.tensor([[10, 20, 30], [5, 15, 25]], dtype=torch.long)

    output = model(dummy_input)

    assert output.shape == (2, num_songs)


def test_qml_classifier_predict():
    """Test QML classifier output"""
    classifier = QMLClassifier(layers=1)

    dummy_input = torch.tensor([[0.5, 0.2], [0.1, 0.9]], dtype=torch.float32)

    output = classifier.predict(dummy_input)

    # Convert to tensor if needed
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)

    assert output.shape[0] == 2

    # Values between 0 and 1
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)