import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# --- Neural Network for Q-values ---
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# --- DDQN Agent ---
class DDQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=64,
                 gamma=0.99, lr=1e-3, sync_freq=1000, epsilon_start=1.0,
                 epsilon_final=0.01, epsilon_decay=50000):
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.online_net = QNetwork(obs_size, n_actions)
        self.target_net = QNetwork(obs_size, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_freq = sync_freq

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.frame_idx = 0
        self.n_actions = n_actions

    def select_action(self, state):
        self.frame_idx += 1
        eps = self.epsilon_final + (self.epsilon - self.epsilon_final) * \
            np.exp(-1. * self.frame_idx / self.epsilon_decay)
        if random.random() < eps:
            return random.randrange(self.n_actions)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.online_net(state_v)
            return int(q_vals.argmax(dim=1).item())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Current Q values
        q_values = self.online_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN target calculation
        next_actions = self.online_net(next_states).argmax(dim=1)
        next_q_values = self.target_net(next_states)
        next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q = rewards + (1 - dones) * self.gamma * next_q_value

        loss = nn.MSELoss()(q_value, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network
        if self.frame_idx % self.sync_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# --- Training Loop ---
def train(env_name: str, num_frames=200000):
    env = gym.make(env_name)
    agent = DDQNAgent(env)

    state, _ = env.reset()
    episode_reward = 0

    for frame in range(1, num_frames + 1):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        episode_reward += reward

        if done:
            state, _ = env.reset()
            print(f"Frame: {frame}, Episode Reward: {episode_reward:.2f}")
            episode_reward = 0

    env.close()

if __name__ == '__main__':
    train('CustomFloorPlanEnv-v0')
