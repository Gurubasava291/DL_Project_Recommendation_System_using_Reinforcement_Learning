import numpy as np
import random

class QLearningRecommender:
    def __init__(self, num_songs, state_length=3, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.num_songs = num_songs
        self.state_length = state_length
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: dictionary mapping state (tuple) -> numpy array of size num_songs
        self.q_table = {}
        
    def _get_q_values(self, state):
        if state not in self.q_table:
            # Initialize with small random values to break ties
            self.q_table[state] = np.random.uniform(low=-0.01, high=0.01, size=self.num_songs)
        return self.q_table[state]
        
    def choose_action(self, state):
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.num_songs - 1)
        else:
            q_values = self._get_q_values(state)
            return np.argmax(q_values)
            
    def update(self, state, action, reward, next_state):
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        
        # Bellman equation update
        best_next_q = np.max(next_q_values)
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - q_values[action]
        
        self.q_table[state][action] += self.alpha * td_error
        
    def recommend(self, state, k=5):
        q_values = self._get_q_values(state)
        # Get top k actions with highest Q-values
        top_k_actions = np.argsort(q_values)[::-1][:k]
        return top_k_actions.tolist()
