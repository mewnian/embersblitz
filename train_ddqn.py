import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from UnifiedEnvWrapper import UnifiedEnvWrapper

# Q-network: simple MLP
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), actions, rewards, np.stack(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# DDQN Agent
class DDQNAgent:
    def __init__(
        self, state_dim, action_dim, buffer_capacity=100000,
        batch_size=64, gamma=0.99, lr=1e-3, tau=0.005, device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        q_vals = self.policy_net(states).gather(1, actions)

        # Double DQN target Q calculation
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q_vals = self.target_net(next_states).gather(1, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * next_q_vals

        loss = nn.MSELoss()(q_vals, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

def train_ddqn(
    env_name, env_type="continuous", grid_size=10, partial_view_size=5,
    num_episodes=500, max_steps=200, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.995
):
    env = gym.make(env_name)
    wrapped_env = UnifiedEnvWrapper(env, env_type, grid_size, partial_view_size)
    state_dim = wrapped_env.observation_space.shape[0]
    action_dim = wrapped_env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)
    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):
        state, _ = wrapped_env.reset()
        episode_reward = 0.0
        for step in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = wrapped_env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
            if done:
                break

        epsilon = max(epsilon_final, epsilon * epsilon_decay)
        if ep % 10 == 0:
            print(f"Episode {ep}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train_ddqn()
