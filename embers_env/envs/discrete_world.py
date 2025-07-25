from typing import Optional, Dict
from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

# Discrete room plan environment
# Each block is either an empty space, a wall or an exit
class DiscreteWorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, width: int = 5, height: int = 5, window_size: tuple = (512, 512), render_mode = None):
        super(DiscreteWorldEnv, self).__init__()
        # size of the room
        self.width = width
        self.height = height
        # size of the pygame window
        self.window_size = window_size

        # initialize position
        # assume single-agent environment and single exit for now
        self._agent_pos = np.array([-1, -1], dtype=np.int32)
        self._target_pos = np.array([-1, -1], dtype=np.int32)

        # observation space (2D coordinates)
        self.observation_space = spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0,0]), high=np.array([self.width-1, self.height-1]), dtype=np.int32
            ),
            "exit": gym.spaces.Box(
                low=np.array([0,0]), high=np.array([self.width-1, self.height-1]), dtype=np.int32
            ),
        })

        # define action space
        self.action_space = gym.spaces.Discrete(8) # eight direction
        # specify action space by direction
        self._action_to_direction = {
            # rewrite the action and match the comment to the direction of the action
            Actions.NORTH.value: np.array([0, 1]),   # up (0)
            Actions.NORTHEAST.value: np.array([1, 1]), # right-up (45)
            Actions.EAST.value: np.array([1, 0]),    # right (90)
            Actions.SOUTHEAST.value: np.array([1, -1]), # right-down (135)
            Actions.SOUTH.value: np.array([0, -1]),  # down (180)
            Actions.SOUTHWEST.value: np.array([-1, -1]), # left-down (225)
            Actions.WEST.value: np.array([-1, 0]),   # left (270)
            Actions.NORTHWEST.value: np.array([-1, 1]), # left-up (315)
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
        return {
            "agent": self._agent_pos,
            "exit": self._target_pos,
        }
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(self._agent_pos - self._target_pos, ord=1)
        }
    
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
        self._agent_pos = self.np_random.integers(low=0, high=np.array([self.width, self.height]), size=2, dtype=np.int32)
        self._target_pos = None
        while self._target_pos is None or np.array_equal(self._agent_pos, self._target_pos):
            # Ensure agent and target are not in the same position
            self._target_pos = self.np_random.integers(low=0, high=np.array([self.width, self.height]), size=2, dtype=np.int32)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action: int):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-7 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        direction = self._action_to_direction[action]
        # clip the agent's position to stay within bounds
        self._agent_pos = np.clip(self._agent_pos + direction, 0, np.array([self.width - 1, self.height - 1]))
        # check if agent reaches the target
        terminated = np.array_equal(self._agent_pos, self._target_pos)
        truncated = False 
        distance = np.linalg.norm(self._agent_pos - self._target_pos, ord=1)
        reward = 1 if terminated else -0.1 * distance
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
        pix_square = (self.window_size[0] / self.width, self.window_size[1] / self.height)  
        # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square * self._target_pos, pix_square
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_pos + 0.5) * pix_square,
            min(pix_square) / 3,
        )

        # Finally, add some gridlines
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square[0] * y),
                (self.window_size[0], pix_square[0] * y),
                width=3,
            )

        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square[1] * x, 0),
                (pix_square[1] * x, self.window_size[1]),
                width=3,
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
        


    