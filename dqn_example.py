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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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




def evaluate_policy(policy_net, env, num_episodes=20, max_steps_per_episode=200):
    successes = 0
    total_steps = 0
    for _ in range(num_episodes):
        state_dict, _ = env.reset()
        state = np.concatenate([state_dict['agent'], state_dict['exit']]).astype(np.float32)
        done = False
        steps = 0
        terminated = False  # ensure defined
        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = policy_net(st).argmax(dim=1).item()
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            next_state = np.concatenate([next_state_dict['agent'], next_state_dict['exit']]).astype(np.float32)
            done = terminated or truncated
            state = next_state
            steps += 1
        if terminated:
            successes += 1
        total_steps += steps
    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    return success_rate, avg_steps



def train_dqn():
    # Environment setup

    # replace gym.make(...) with direct instantiation:
   

    env = DiscreteWorldEnv(width=10, height=10, render_mode="rgb_array")

    
    # implementing success rate tracking
    from collections import deque

    success_history = deque(maxlen=100)  # last 100 episodes


    
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

    step_penalty = 0.001  # Small penalty for each step

    
    for episode in range(2000): #number of episodes
        state_dict, _ = env.reset()
        state = np.concatenate([
            state_dict['agent'].astype(np.float32), 
            state_dict['exit'].astype(np.float32)
        ])
        episode_reward = 0
        
        min_replay_size = 2000
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


            #env.render()  # This will show the current state
            next_state = np.concatenate([
                next_state_dict['agent'].astype(np.float32),
                next_state_dict['exit'].astype(np.float32)
            ])
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            #print("1")

            #if len(replay_buffer) >= max(batch_size, min_replay_size):
            if len(replay_buffer) >= min_replay_size:
                batch = replay_buffer.sample(batch_size)
                # Convert batch of tuples to separate lists and then to numpy arrays
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
                #print("1.5")
                # Convert numpy arrays to tensors efficiently
                states = torch.from_numpy(states).to(device).float()
                actions = torch.from_numpy(actions).to(device).long()
                rewards = torch.from_numpy(rewards).to(device).float()
                next_states = torch.from_numpy(next_states).to(device).float()
                dones = torch.from_numpy(dones).to(device).float()

                
                # Compute Q values
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q = target_net(next_states).max(1)[0].detach()
                target_q = rewards + gamma * next_q * (1 - dones)
                #print("2")
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
            #print("1.2")
            state = next_state



        # Determine if this episode succeeded (terminated, not just truncated)
        success = terminated  # since done = terminated or truncated, use terminated specifically
        success_history.append(1 if success else 0)

        #print("1.3")
        episode_rewards.append(episode_reward)
        # Logging (example every 50 episodes)
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = sum(success_history) / len(success_history) if success_history else 0.0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Success Rate (last 100): {success_rate:.2%}")
            print(f"Epsilon: {epsilon:.3f}")



        
        # Update target network periodically
        #if episode % 10 == 0:
            #target_net.load_state_dict(policy_net.state_dict())
        
        #print("1.4")
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        '''
        if episode % 50 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
            print(f"Epsilon: {epsilon:.3f}")
        '''
        
        #print("1.7")
        if episode % 50 == 0:
            #print("1.75")
            sr, avg_len = evaluate_policy(policy_net, env, num_episodes=20)
            #print("1.755")
            print(f"[Eval] Greedy success rate: {sr:.2%}, avg steps: {avg_len:.1f}")
        #print("1.8")
    

    env.close()



if __name__ == "__main__":
    train_dqn()