import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, num_songs, state_length, embedding_dim=16):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(num_songs, embedding_dim)
        # The state is a sequence of `state_length` songs.
        self.fc1 = nn.Linear(state_length * embedding_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_songs) # Output Q-values for all songs
        
    def forward(self, x):
        # x shape: (batch_size, state_length)
        x = self.embedding(x)
        # flatten embeddings
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, num_songs, state_length=3, gamma=0.9, epsilon=0.2, lr=0.001):
        self.num_songs = num_songs
        self.state_length = state_length
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.model = DQN(num_songs, state_length)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_songs)
            
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
            
    def update(self, state, action, reward, next_state):
        state_tensor = torch.tensor([state], dtype=torch.long)
        next_state_tensor = torch.tensor([next_state], dtype=torch.long)
        action_tensor = torch.tensor([[action]], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
        q_values = self.model(state_tensor)
        q_value = q_values.gather(1, action_tensor).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward_tensor + self.gamma * max_next_q_value
            
        loss = self.criterion(q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def recommend(self, state, k=5):
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long)
            q_values = self.model(state_tensor).squeeze(0)
            # Get top K actions
            top_k = torch.topk(q_values, k).indices.tolist()
            return top_k
