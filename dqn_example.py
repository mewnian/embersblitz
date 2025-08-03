import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import discrete_env
import embers_env.envs

from embers_env.envs.discrete_world import DiscreteWorldEnv

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train_dqn():
    # Environment setup

    # replace gym.make(...) with direct instantiation:
   

    env = DiscreteWorldEnv(width=10, height=10, render_mode="rgb_array")

    
    

    
    # Parameters
    input_size = 4  # state dimensions (agent_x, agent_y, target_x, target_y)
    output_size = 8  # number of actions (UP, RIGHT, DOWN, LEFT)
    learning_rate = 1e-4
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.997
    batch_size = 64
    buffer_size = 50000
    
    # Initialize DQN and target network
    policy_net = DQN(input_size, output_size)
    target_net = DQN(input_size, output_size)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    epsilon = epsilon_start
    episode_rewards = []

    step_penalty = 0.0  # Small penalty for each step

    
    for episode in range(2000): #number of episodes
        state_dict, _ = env.reset()
        state = np.concatenate([
            state_dict['agent'].astype(np.float32), 
            state_dict['exit'].astype(np.float32)
        ])
        episode_reward = 0
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape (1,4)
                    q_values = policy_net(state_tensor)  # (1,8)
                    action = q_values.argmax(dim=1).item()
                
            
            # Take action
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            reward = reward - step_penalty


            env.render()  # This will show the current state
            next_state = np.concatenate([
                next_state_dict['agent'].astype(np.float32),
                next_state_dict['exit'].astype(np.float32)
            ])
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            min_replay_size = 2000

            #if len(replay_buffer) >= max(batch_size, min_replay_size):
            if len(replay_buffer) >= min_replay_size:
                batch = replay_buffer.sample(batch_size)
                # Convert batch of tuples to separate lists and then to numpy arrays
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
                
                # Convert numpy arrays to tensors efficiently
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Compute Q values
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q = target_net(next_states).max(1)[0].detach()
                target_q = rewards + gamma * next_q * (1 - dones)
                
                # Compute loss and update
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tau = 0.005
                for p, t in zip(policy_net.parameters(), target_net.parameters()):
                    t.data.copy_(tau * p.data + (1 - tau) * t.data)
            
            if done:
                break
                
            state = next_state
        
        # Update target network periodically
        #if episode % 10 == 0:
            #target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        if episode % 50 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
            print(f"Epsilon: {epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train_dqn()