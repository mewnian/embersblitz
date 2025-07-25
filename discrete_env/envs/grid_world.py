from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=(10, 10), render_mode=None):
        self.size = size  # (width, height)
        self.width, self.height = size
        self.window_size = 512  # Default window size in pixels

        # Calculate grid cell size based on the larger dimension
        self.cell_size = self.window_size / max(self.width, self.height)


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([self.width - 1, self.height - 1]), 
                shape=(2,), 
                dtype=np.int32
            ),
            "target": spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([self.width - 1, self.height - 1]), 
                shape=(2,), 
                dtype=np.int32
            )
        })

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        # Initialize state
        self.agent_pos = None
        self.target_pos = None
        
        # Set render mode
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(
            low=0,
            high=[self.width, self.height],
            size=2,
            dtype=np.int32
        ).astype(np.int32)  # Ensure int32 type

        # Sample target location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                low=0,
                high=[self.width, self.height],
                size=2,
                dtype=np.int32
            ).astype(np.int32)  # Ensure int32 type

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 
            [0, 0],
            [self.width - 1, self.height - 1]
        ).astype(np.int32)  # Ensure int32 type
        
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = self.cell_size

        # Draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw gridlines
        for x in range(self.width + 1):  # Changed from self.size to self.width
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        
        for y in range(self.height + 1):  # Changed from self.size to self.height
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (self.window_size, pix_square_size * y),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
