import numpy as np
import gymnasium as gym
import embers_env
from agents.q_agent import QAgent
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

env = gym.make(
    'embers_env/ContinuousWorld-v0', 
    width=1000, height=1000,
    window_size=(1000, 1000),
    episode_step_limit=5000,
    render_mode='human',
)

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

targets = [
    (0, 920, 1000, 1000),
    
]

env.unwrapped.add_obstacles(obstacles)
env.unwrapped.add_targets(targets)

n_episodes = 500

episode_over = False
total_reward = 0

agent = QAgent(env, epsilon_decay=0.005, final_epsilon=0.01, discount_factor=0.95)

env.unwrapped.set_agent(250, 250)
obs, info = env.reset()

# wait for 5 seconds before closing
errors = []

for episode in tqdm(range(n_episodes), desc="Training episodes"):
    cnt = 0
    env.unwrapped.set_agent(250, 250)
    obs, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = agent.get_action(obs)
        # if action["type"] == 0:
        #     action["angle"] = obs.get("agent_angle")  # Use current agent angle if available
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            next_obs=next_obs
        )

        total_reward += reward
        episode_over = terminated or truncated
        obs = next_obs
        cnt += 1
        # print(f"Step {cnt+1}: Info {info}")
        
    errors.append(np.mean(np.abs(agent.training_error)))
    agent.reset_error()
    agent.decay_epsilon()
    print(f"Episode {episode + 1} finished! Total reward: {total_reward}, Epsilon: {agent.epsilon}")

env.close()
# Plotting the training error
plt.plot(errors)
plt.xlabel('Episode')
plt.ylabel('Mean training error')
plt.title('Training error over episodes')
plt.show()