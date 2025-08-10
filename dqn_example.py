import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import discrete_env
import embers_env.envs
import math

from UnifiedEnvWrapper import UnifiedEnvWrapper  # assume you saved the earlier code as unified_wrapper.py
from embers_env.envs.discrete_world import DiscreteWorldEnv
from embers_env.envs.continuous_world import ContinuousWorldEnv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        use_layernorm: bool = False
    ):
        super().__init__()
        layers = []
        in_features = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation(inplace=True) if 'inplace' in activation.__init__.__code__.co_varnames else activation())
            in_features = h

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        # Orthogonal init tends to work nicely for MLPs with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make sure we see [batch, feat]; cast to float for safety
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x.float().contiguous())

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)




def evaluate_policy(policy_net, env, num_episodes=20, max_steps_per_episode=400):
    successes = 0
    total_steps = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        terminated = False
        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(st).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
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
   
    #Discrete
    #base_env = DiscreteWorldEnv(width=10, height=10, render_mode="rgb_array")
    #env = UnifiedEnvWrapper(base_env, env_type="discrete", grid_size=10)


    # define geometry
    targets = [
        (0, 920, 1000, 1000),
    ]

    obstacles = [
        (45, 880, 95, 940), # pillar one
        (60, 250, 80, 880),
        (45, 190, 95, 250),
        (765, 880, 815, 940), # pillar two
        (780, 250, 800, 880),
        (765, 190, 815, 250),
        (95, 190, 765, 215), # wall
        (0, 190, 45, 215),
        (765, 190, 1000, 215),
        (420, 215, 440, 915),
        (195, 915, 665, 935),
        (0, 915, 45, 935),
        (815, 915, 1000, 935),
        (215, 730, 420, 740),
        (215, 835, 225, 915),
        (440, 730, 645, 740),
        (635, 835, 645, 915),
        (215, 300, 420, 395), # furniture
        (440, 300, 645, 395),
        (380, 395, 420, 455),
        (440, 395, 480, 455),
        (215, 455, 420, 550),
        (440, 455, 645, 550),
        (80, 265, 120, 525),
        (740, 265, 780, 525),
    ]

    # Continuous
    base_env = ContinuousWorldEnv(width=1000, height=1000, render_mode="rgb_array", episode_step_limit=400)
    #xmin, ymin = 300 - 25, 200 - 25
    #xmax, ymax = 300 + 25, 200 + 25
    #base_env.add_target((xmin, ymin, xmax, ymax))
    # add target(s)
    for t in targets:
        base_env.add_target(t)

    # add obstacles
    for o in obstacles:
        base_env.add_obstacle(o)
    env = UnifiedEnvWrapper(base_env, env_type="continuous", grid_size=1000)  # grid_size is used for normalization scale

    # Separate evaluation env (no rendering to keep it fast)
    eval_base = ContinuousWorldEnv(width=1000, height=1000, render_mode="rgb_array", episode_step_limit=400)
    # add target(s)
    for t in targets:
        eval_base.add_target(t)

    # add obstacles
    for o in obstacles:
        eval_base.add_obstacle(o)
    eval_env = UnifiedEnvWrapper(eval_base, env_type="continuous", grid_size=1000)

    
    # implementing success rate tracking
    from collections import deque

    success_history = deque(maxlen=100)  # last 100 episodes

    episode_rewards = []
    avg_q_values_per_episode = []

    
    # Parameters
    #input_size = 4  # state dimensions (agent_x, agent_y, target_x, target_y)
    #output_size = 8  # number of actions (UP, RIGHT, DOWN, LEFT)
    input_size = env.observation_space.shape[0]  # should match wrapper (4)
    output_size = env.action_space.n             # 8
    learning_rate = 1e-4
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.997
    batch_size = 64
    buffer_size = 50000
    
    # Initialize DQN and target network
    
    policy_net = DQN(input_size, output_size).to(device)
    target_net = DQN(input_size, output_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    epsilon = epsilon_start
    #episode_rewards = []

    step_penalty = 0.001  # Small penalty for each step

    
    for episode in range(2000): #number of episodes
        state, _ = env.reset()
        #state_dict, _ = env.reset()
        #state = np.concatenate([ state_dict['agent'].astype(np.float32), state_dict['exit'].astype(np.float32) ])
        episode_reward = 0
        
        min_replay_size = 2000
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
                
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward


            #env.render()  # This will show the current state
            

            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
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

                
                # --- compute current and target Q ---
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]  # [B]
                    target_q = rewards + gamma * next_q * (1 - dones)  # Bellman target

                # --- loss and optimization ---
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                # --- soft update target network ---
                tau = 0.005  # keep this consistent with your existing code
                for p, t in zip(policy_net.parameters(), target_net.parameters()):
                    t.data.copy_(tau * p.data + (1 - tau) * t.data)


                
                optimizer.step()

                tau = 0.005
                for p, t in zip(policy_net.parameters(), target_net.parameters()):
                    t.data.copy_(tau * p.data + (1 - tau) * t.data)
            
            if done:
                break
            #print("1.2")
            state = next_state


        # after episode is done
        episode_rewards.append(episode_reward)

        # compute average Q-value if enough in buffer
        avg_q = None
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            states_tensor = torch.from_numpy(states).to(device).float()
            with torch.no_grad():
                # max over actions per state
                q_vals = policy_net(states_tensor)  # shape [B, A]
                max_q, _ = q_vals.max(1)           # shape [B]
                avg_q = max_q.mean().item()
        else:
            # fallback: current state value estimate
            with torch.no_grad():
                st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = policy_net(st)
                max_q = q_vals.max(1)[0]
                avg_q = max_q.item()
        avg_q_values_per_episode.append(avg_q)




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
        
        #viz
        if episode > 1000 and success and episode % 50 == 0: 
            #visualize_episode(policy_net, eval_env)
            #plot_metrics(episode_rewards, avg_q_values_per_episode, smooth_window=20)
            print("")


        if episode % 50 == 0:
            # sr, avg_len = evaluate_policy(policy_net, env, num_episodes=20) # discrete

            policy_net.eval()
            sr, avg_len = evaluate_policy(policy_net, eval_env, num_episodes=20)
            policy_net.train()
            print(f"[Eval] Greedy success rate: {sr:.2%}, avg steps: {avg_len:.1f}")

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = sum(success_history) / len(success_history) if success_history else 0.0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Success Rate (last 100): {success_rate:.2%}, Avg Q: {avg_q_values_per_episode[-1]:.3f}")
            if episode % 250 == 0 or episode == 1950: 
                plot_metrics(episode_rewards, avg_q_values_per_episode, smooth_window=20)


    

    env.close()

import matplotlib.pyplot as plt

def unnormalize_agent_pos(state_vec, env):
    # For discrete/continuous wrapper: first two entries are agent_norm in [0,1] scaled by (grid_size-1)
    agent_norm = state_vec[:2]  # e.g., agent_x, agent_y normalized
    agent_center = agent_norm * (env.grid_size - 1)  # in env coordinate units
    return agent_center  # e.g., [x, y] in world coords

def world_to_pixel(coord, env):
    # coord: [x, y] in world units (same scale as ContinuousWorldEnv width/height)
    # need base env window size and world size to map to pixels
    base = env.env  # underlying env
    # For continuous_world, width,height are base.width, base.height; window_size is pixel dimensions
    scale = np.array(base.window_size) / np.array([base.width, base.height])
    # Note: depending on coordinate conventions (y up/down), you may need to flip Y
    pixel = coord * scale
    return pixel.astype(int)

def visualize_episode(policy_net, env, max_steps=400):
    policy_net.eval()
    state, _ = env.reset()
    path_world = []
    frames = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        agent_world = unnormalize_agent_pos(state, env)
        path_world.append(agent_world.copy())

        with torch.no_grad():
            st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy_net(st).argmax(dim=1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        steps += 1

        # Get current frame from underlying environment (rgb_array)
        frame = env.env.render()  # wrapper.step returns state_vec; underlying render is from env.env
        if frame is not None:
            frames.append(frame.copy())

    # Use final frame as background
    if len(frames) == 0:
        print("No frames captured.")
        return

    final_frame = frames[-1]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(final_frame)

    # Convert world path to pixel and plot
    pixel_path = np.array([world_to_pixel(p, env) for p in path_world])  # shape (T,2)
    # Depending on orientation, you might need to swap axes: pixel_path[:,1] is y
    ax.plot(pixel_path[:,0], pixel_path[:,1], marker='o', markersize=4, linewidth=2, color='red')
    ax.set_title("Agent trajectory (last frame background)")
    plt.show()
    policy_net.train()

import matplotlib.pyplot as plt
import numpy as np

def smooth(x, window=10):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window)/window, mode="valid")

def plot_metrics(reward_history, q_history, smooth_window=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(reward_history, label="raw")
    if len(reward_history) >= smooth_window:
        axes[0].plot(range(smooth_window-1, len(reward_history)), smooth(reward_history, smooth_window), label=f"smoothed({smooth_window})")
    axes[0].set_title("Average reward by episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(q_history, label="raw")
    if len(q_history) >= smooth_window:
        axes[1].plot(range(smooth_window-1, len(q_history)), smooth(q_history, smooth_window), label=f"smoothed({smooth_window})")
    axes[1].set_title("Average Q-value by episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Q-value")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_dqn()