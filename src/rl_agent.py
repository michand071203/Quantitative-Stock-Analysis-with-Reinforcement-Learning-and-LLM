import numpy as np
import pandas as pd

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995, min_exploration_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size, action_size))
        self.actions = ['buy', 'sell', 'hold']

    def discretize_state(self, state: np.ndarray) -> int:
        """Discretize continuous state into a single index."""
        bins = np.linspace(-1, 1, self.state_size)
        state_index = np.digitize(state[0], bins) - 1
        return np.clip(state_index, 0, self.state_size - 1)

    def choose_action(self, state: np.ndarray) -> int:
        """Choose an action using epsilon-greedy policy."""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)
        state_index = self.discretize_state(state)
        return np.argmax(self.q_table[state_index])

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update Q-table using Q-learning formula."""
        state_index = self.discretize_state(state)
        next_state_index = self.discretize_state(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index, best_next_action]
        self.q_table[state_index, action] += self.learning_rate * (
            td_target - self.q_table[state_index, action]
        )
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
