import gymnasium as gym
import numpy as np
from collections import defaultdict

class QAgent:
    def __init__(
        self, 
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.01,
        final_epsilon: float = 0.05,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: np.zeros((env.action_space.n)))
        self.training_error = []
    
    def get_action(self, obs):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # explore: random action
            return self.env.action_space.sample()
        else:
            # exploit: select greedy action (best q-value)
            return np.argmax(self.q_values[self.hashed_obs(obs)])
        
    
    def hashed_obs(self, obs):
        """Convert observation to a hashable type for use in Q-table."""
        hashed = [obs["direction"]]
        hashed.extend(tuple(obs["view"].flatten()))
        return tuple(hashed)
    
    def action_index(self, action):
        return action
        # act_type, act_angle = int(action["type"]), int(action["angle"])
        # return (act_type, act_angle)

    def reset_error(self):
        """Reset the training error list."""
        self.training_error = []
    
    def update(
        self, 
        obs: dict,
        action: tuple,
        reward: float,
        terminated: bool,
        next_obs: dict
    ):
        """Update Q-values based on observation."""
        future_q_value = (not terminated) * np.max(self.q_values[self.hashed_obs(next_obs)])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[self.hashed_obs(obs)][self.action_index(action)]
        self.q_values[self.hashed_obs(obs)][self.action_index(action)] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        """Reduce epsilon after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)