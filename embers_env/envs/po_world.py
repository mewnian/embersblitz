from typing import Optional, Dict
from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class Cell(Enum):
    EMPTY = 0
    TARGET = 1
    OBSTACLE = -1

class Directions(Enum):
    NORTH = 0
    # NORTHEAST = 1
    EAST = 1
    # SOUTHEAST = 3
    SOUTH = 2
    # SOUTHWEST = 5
    WEST = 3
    # NORTHWEST = 7

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

# Partially observable room plan environment
# Agent will have a limited 10x10 view
# Each block is either an empty space, a wall or an exit
class PartialObservationWorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 240,
    }

    def __init__(self, 
        width: int = 512, height: int = 512, 
        view_size: int = 20,
        agent_size: int = 10,
        targets: Optional[list] = None,
        obstacles: Optional[list] = None,
        episode_step_limit: Optional[int] = None,
        window_size: tuple = (512, 512), render_mode = None
    ):
        super(PartialObservationWorldEnv, self).__init__()
        # size of the room
        self.width = width
        self.height = height
        self.view_size = view_size # (window will have length (2*view + 1))
        self._map = np.zeros((self.width, self.height), dtype=np.int32)

        # size of the pygame window
        self.window_size = window_size

        # episode step limit
        self.episode_step_limit = episode_step_limit
        self._episode_steps = 0

        # initialize position
        self._agent_pos = np.array([0, 0], dtype=np.int32)
        self._agent_size = agent_size
        self._agent_direction = 0
        self._targets = targets or []
        self._obstacles = obstacles or []

        # observation space (2D coordinates)
        self.observation_space = gym.spaces.Dict({
            "view": spaces.Box(
                low=-1, high=1, 
                shape=(),
                dtype=np.int8,
            ),
            "direction": spaces.Discrete(8)
        })

        # define action space
        # two choices: move forward or change direction

        # define action space
        self.action_space = spaces.Discrete(3) # three actions
        # specify action space by direction
        self._named_direction = {
            # rewrite the action and match the comment to the direction of the action
            Directions.NORTH.value: np.array([0, 1], dtype=np.int8),   # up (0)
            # Directions.NORTHEAST.value: np.array([1, 1], dtype=np.int8), # right-up (45)
            Directions.EAST.value: np.array([1, 0], dtype=np.int8),    # right (90)
            # Directions.SOUTHEAST.value: np.array([1, -1], dtype=np.int8), # right-down (135)
            Directions.SOUTH.value: np.array([0, -1], dtype=np.int8),  # down (180)
            # Directions.SOUTHWEST.value: np.array([-1, -1], dtype=np.int8), # left-down (225)
            Directions.WEST.value: np.array([-1, 0], dtype=np.int8),   # left (270)
            # Directions.NORTHWEST.value: np.array([-1, 1], dtype=np.int8), # left-up (315)
        }

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
        x, y = self._agent_pos
        return {
            "view": self._map[
                (x - self.view_size):(x + self.view_size),
                (y - self.view_size):(y + self.view_size)
            ],
            "direction": self._agent_direction
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
        self._agent_pos = np.array([center_x, center_y], dtype=np.int32)
        return True
    
    def add_target(self, xmin, ymin, xmax, ymax):
        """Add a target to the environment.

        Args:
            target (tuple): The rectangle representing the target, in order of (xmin, ymin, xmax, ymax). (inclusive)
        """
        self._map[xmin:xmax, ymin:ymax] = Cell.TARGET.value
        return True
    
    def add_targets(self, targets: list):
        """Add multiple targets to the environment.

        Args:
            targets (list): List of Rectangle instances representing targets.
        """
        added = 0
        for target in targets:
            if self.add_target(*target):
                added += 1
                self._targets.append(np.array(target))
        return added
    
    def add_obstacle(self, xmin, ymin, xmax, ymax):
        """Add an obstacle to the environment.

        Args:
            obstacle (tuple): The rectangle representing the obstacle, in order of (xmin, ymin, xmax, ymax).
        """
        self._map[xmin:xmax, ymin:ymax] = Cell.OBSTACLE.value
        return True
    
    def add_obstacles(self, obstacles: list):
        """Add multiple obstacles to the environment.

        Args:
            obstacles (list): List of Rectangle instances representing obstacles.
        """
        added = 0
        for obstacle in obstacles:
            if self.add_obstacle(*obstacle):
                added += 1
                self._obstacles.append(np.array(obstacle))
        return added
    
    def is_colliding_with_obstacles(self, position: np.ndarray) -> bool:
        """Check if the given position collides with any obstacles.

        Args:
            position (ndarray): The rectangle representing the position to check.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        x, y = position
        return np.any(self._map[
            (x - self._agent_size):(x + self._agent_size),
            (y - self._agent_size):(y + self._agent_size),
        ] == Cell.OBSTACLE.value)
    
    def is_overlapping_with_target(self, position: np.ndarray) -> bool:
        """Check if the given position overlaps with any target.

        Args:
            position (ndarray): The rectangle representing the position to check.

        Returns:
            bool: True if there is an overlap with a target, False otherwise.
        """
        x, y = position
        return np.any(self._map[
            (x - self._agent_size):(x + self._agent_size),
            (y - self._agent_size):(y + self._agent_size),
        ] == Cell.TARGET.value)
    
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to start a new episode.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.

        Returns:
            tuple: (observation, info) Initial observation and info
        """
        super().reset(seed=seed)
        
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
        reward = -1
        terminated = False
        if action == 0: # Move forward
            new_pos = self._agent_pos + self._named_direction[self._agent_direction]
            # if agent is not colliding with obstacles, move forward
            if self.is_colliding_with_obstacles(new_pos):
                reward = -self.episode_step_limit
                terminated = True
            else:
                self._agent_pos = new_pos
                if self.is_overlapping_with_target(self._agent_pos):
                    reward = self.episode_step_limit / 2
                    terminated = True
        elif action == 1: # Turns right by 45 degrees
            self._agent_direction = (self._agent_direction + 1) % len(Directions)
        elif action == 2: # Turns left by 45 degrees
            self._agent_direction = (self._agent_direction + len(Directions) - 1) % len(Directions)
        terminated = terminated or (self._episode_steps >= self.episode_step_limit)
        truncated = False
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
            size = target[2:] - target[:2]
            pygame.draw.rect(
                canvas,
                (255, 255, 0),
                pygame.Rect(*(target[:2] * unit_square), *(size * unit_square))
            )
        # Then the obstacles
        for obstacle in self._obstacles:
            size = obstacle[2:] - obstacle[:2]
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(*(obstacle[:2] * unit_square), *(size * unit_square))
            )
        # Now we draw the agent
        agent_topleft = self._agent_pos - np.array([1, 1]) * self._agent_size
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                *(agent_topleft * unit_square), 
                *((self._agent_size * 2, self._agent_size * 2) * unit_square)
            )
        )

        # Draw a bounding box around the agent's viewpoint
        view_topleft = self._agent_pos - np.array([1, 1]) * self.view_size
        pygame.draw.rect(
            canvas,
            (0, 255, 128),
            pygame.Rect(
                *(view_topleft * unit_square), 
                *((self.view_size * 2, self.view_size * 2) * unit_square)
            ),
            5
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
        


    