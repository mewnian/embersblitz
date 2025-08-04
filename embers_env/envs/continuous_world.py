from typing import Optional, Dict
from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class Rectangle:
    def __init__(self, center: np.ndarray = np.array([-1, -1]), shape: np.ndarray = np.array([0, 0])):
        self.center = center
        self.shape = shape

    def __array__(self):
        return np.concatenate([self.center - self.shape / 2, self.center + self.shape / 2])
    
    def is_overlapping(self, other) -> bool:
        cur_xmin, cur_ymin, cur_xmax, cur_ymax = self.__array__()
        other_xmin, other_ymin, other_xmax, other_ymax = other.__array__()
        return not (min(cur_xmax, other_xmax) < max(cur_xmin, other_xmin) or min(cur_ymax, other_ymax) < max(cur_ymin, other_ymin))
    
    def set_center(self, center: np.ndarray):
        """Set the center of the rectangle."""
        self.center = center
    
    def get_area(self):
        return np.prod(self.shape)

# Discrete room plan environment
# Each block is either an empty space, a wall or an exit
class ContinuousWorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 240,
    }

    def __init__(self, 
        width: float = 600, height: float = 400, 
        targets: Optional[list] = None,
        obstacles: Optional[list] = None,
        episode_step_limit: Optional[int] = None,
        window_size: tuple = (512, 512), render_mode = None
    ):
        super(ContinuousWorldEnv, self).__init__()
        # size of the room
        self.width = width
        self.height = height

        # size of the pygame window
        self.window_size = window_size

        # episode step limit
        self.episode_step_limit = episode_step_limit
        self._episode_steps = 0

        # initialize position
        self._agent_pos = None
        self._agent_size = min(width, height) / 50.0
        self._agent_angle = 0
        self._agent_direction = np.array([1.0, 0.0])
        self._targets = targets or []
        self._obstacles = obstacles or []

        # observation space (2D coordinates)
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0,0]), high=np.array([self.width, self.height]), dtype=np.float32
            ),
            "agent_angle": gym.spaces.Discrete(360),  # angle in degrees (0-359)
            "targets": gym.spaces.Sequence(
                gym.spaces.Box(low=np.array([0,0]), high=np.array([self.width, self.height]), dtype=np.float32)
            ),
            "obstacles": gym.spaces.Sequence(
                gym.spaces.Box(low=np.array([0,0]), high=np.array([self.width, self.height]), dtype=np.float32)
            )
        })

        # define action space
        # two choices: move forward or change direction
        self.action_space = gym.spaces.Discrete(3)
        # specify action space by direction

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        
        return {
            "agent": np.array(self._agent_pos),
            "agent_angle": self._agent_angle,
            "targets": np.array([np.array(target) for target in self._targets]),
            "obstacles": np.array([np.array(obstacle) for obstacle in self._obstacles])
        }
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            # "distance": np.linalg.norm(self._agent_pos.center - self._target_pos.center)
        }
    
    def set_agent(self, center_x, center_y):
        """Set the agent position in the environment.

        Args:
            xmin (float): Minimum x-coordinate of the agent rectangle.
            ymin (float): Minimum y-coordinate of the agent rectangle.
            xmax (float): Maximum x-coordinate of the agent rectangle.
            ymax (float): Maximum y-coordinate of the agent rectangle.
        """
        rect = Rectangle(np.array([center_x, center_y]), np.array([self._agent_size, self._agent_size]))
        if self.is_overlapping_with_target(rect) or self.is_colliding_with_obstacles(rect):
            return False
        self._agent_pos = rect
        return True
    
    def add_target(self, target: tuple):
        """Add a target to the environment.

        Args:
            target (tuple): The rectangle representing the target, in order of (xmin, ymin, xmax, ymax).
        """
        center = np.array([(target[0] + target[2]) / 2, (target[1] + target[3]) / 2])
        shape = np.array([target[2] - target[0], target[3] - target[1]])
        self._targets.append(Rectangle(center, shape))
        return True
    
    def add_targets(self, targets: list):
        """Add multiple targets to the environment.

        Args:
            targets (list): List of Rectangle instances representing targets.
        """
        for target in targets:
            self.add_target(target)
        return True
    
    def add_obstacle(self, obstacle: tuple):
        """Add an obstacle to the environment.

        Args:
            obstacle (tuple): The rectangle representing the obstacle, in order of (xmin, ymin, xmax, ymax).
        """
        center = np.array([(obstacle[0] + obstacle[2]) / 2, (obstacle[1] + obstacle[3]) / 2])
        shape = np.array([obstacle[2] - obstacle[0], obstacle[3] - obstacle[1]])
        self._obstacles.append(Rectangle(center, shape))
        return True
    
    def add_obstacles(self, obstacles: list):
        """Add multiple obstacles to the environment.

        Args:
            obstacles (list): List of Rectangle instances representing obstacles.
        """
        for obstacle in obstacles:
            self.add_obstacle(obstacle)
        return True
    
    def is_colliding_with_obstacles(self, position: Rectangle) -> bool:
        """Check if the given position collides with any obstacles.

        Args:
            position (Rectangle): The rectangle representing the position to check.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        for obstacle in self._obstacles:
            if position.is_overlapping(obstacle):
                return True
        return False
    
    def is_overlapping_with_target(self, position: Rectangle) -> bool:
        """Check if the given position overlaps with any target.

        Args:
            position (Rectangle): The rectangle representing the position to check.

        Returns:
            bool: True if there is an overlap with a target, False otherwise.
        """
        for target in self._targets:
            if position.is_overlapping(target):
                return True
        return False
    
    def distance_to_nearest_target(self, position: Rectangle) -> float:
        """Calculate the distance to the nearest target from the given position.

        Args:
            position (Rectangle): The rectangle representing the position to check.

        Returns:
            float: The distance to the nearest target, or infinity if no targets exist.
        """
        if not self._targets:
            return float('inf')
        distances = [np.linalg.norm(position.center - target.center, ord=1) for target in self._targets]
        return min(distances)
    
    # def _resolve_collision(self, current_pos, movement):
    #     """Resolve collision with sliding along walls.
        
    #     Args:
    #         current_pos (np.ndarray): Current agent position
    #         movement (np.ndarray): Desired movement vector
            
    #     Returns:
    #         np.ndarray: Final position after collision resolution
    #     """
    #     # Try full movement first
    #     new_pos = np.clip(
    #         current_pos + movement,
    #         np.array([self._agent_size / 2, self._agent_size / 2]),
    #         np.array([self.width - self._agent_size / 2, self.height - self._agent_size / 2])
    #     )
        
    #     test_rect = Rectangle(new_pos, self._agent_pos.shape)
    #     if not self.is_colliding_with_obstacles(test_rect):
    #         return new_pos
        
    #     # Try sliding along X and Y axes separately
    #     for axis in [0, 1]:  # Try X then Y
    #         slide_movement = movement.copy()
    #         slide_movement[1-axis] = 0  # Zero out the other axis
            
    #         slide_pos = np.clip(
    #             current_pos + slide_movement,
    #             np.array([self._agent_size / 2, self._agent_size / 2]),
    #             np.array([self.width - self._agent_size / 2, self.height - self._agent_size / 2])
    #         )
            
    #         if not self.is_colliding_with_obstacles(Rectangle(slide_pos, self._agent_pos.shape)):
    #             return slide_pos
        
    #     return current_pos  # No movement possible
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to start a new episode.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.

        Returns:
            tuple: (observation, info) Initial observation and info
        """
        super().reset(seed=seed)

        # Randomly place agent
        self._agent_pos = None
        if self._agent_pos is None:
            while True:
                # Ensure agent is not overlapping with target or obstacles
                center = self.np_random.uniform(
                    low=[self._agent_size / 2, self._agent_size / 2], 
                    high=[self.width - self._agent_size / 2, self.height - self._agent_size / 2]
                )
                if self.set_agent(center[0], center[1]): break
        
        observation = self._get_obs()
        info = self._get_info()
        self._episode_steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action: Dict):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-7 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._episode_steps += 1
        reward = 0
        terminated = False
        if action == 0: # Move forward
            new_center = np.clip(
                self._agent_pos.center + self._agent_direction,
                np.array([self._agent_size / 2, self._agent_size / 2]),
                np.array([self.width - self._agent_size / 2, self.height - self._agent_size / 2])
            )
            # if agent is not colliding with obstacles, move forward
            if not self.is_colliding_with_obstacles(Rectangle(new_center, self._agent_pos.shape)):
                old_distance = self.distance_to_nearest_target(self._agent_pos)
                self._agent_pos.set_center(new_center)
                new_distance = self.distance_to_nearest_target(self._agent_pos)
                reward = 10 * (old_distance - new_distance) / (self.width + self.height)  # Reward for moving closer to target
            else:
                # resolve collision by sliding along walls and punish
                # new_center = self._resolve_collision(self._agent_pos.center, self._agent_direction)
                # self._agent_pos.set_center(new_center)
                reward = -self.width * self.height # Penalty for collision
                terminated = True
        elif action == 1: # Turns left by 15 degrees
            self._agent_angle = (self._agent_angle + 15) % 360
            self._agent_direction = np.array([np.cos(np.pi * self._agent_angle / 180.0), np.sin(np.pi * self._agent_angle / 180.0)])
            reward = -0.00001
        elif action == 2: # Turns right by 15 degrees
            self._agent_angle = (self._agent_angle + 360 - 15) % 360
            self._agent_direction = np.array([np.cos(np.pi * self._agent_angle / 180.0), np.sin(np.pi * self._agent_angle / 180.0)])
            reward = -0.00001
        # clip the agent's position to stay within bounds
        # check if agent reaches the target
        terminated = terminated or self.is_overlapping_with_target(self._agent_pos)
        if terminated: reward = max(self.width, self.height)  # reward for reaching the target
        truncated = self.episode_step_limit is not None and self._episode_steps >= self.episode_step_limit 
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def render(self):    
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        unit_square = np.array([self.window_size[0] / self.width, self.window_size[1] / self.height])  
        # The size of a single grid square in pixels

        # First we draw the target
        for target in self._targets:
            corner = np.array(target)[:2]
            pygame.draw.rect(
                canvas,
                (255, 255, 0),
                pygame.Rect(*(corner * unit_square), *(target.shape * unit_square))
            )
        # Then the obstacles
        for obstacle in self._obstacles:
            corner = np.array(obstacle)[:2]
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(*(corner * unit_square), *(obstacle.shape * unit_square))
            )
        # Now we draw the agent
        corner = np.array(self._agent_pos)[:2]
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(*(corner * unit_square), *(self._agent_pos.shape * unit_square))
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        


    