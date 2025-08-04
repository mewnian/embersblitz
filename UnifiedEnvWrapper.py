import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Direction vectors in grid-space (x,y): N, NE, E, SE, S, SW, W, NW
_DIRECTION_VECTORS = [
    np.array([0, 1], dtype=np.int32),   # N
    np.array([1, 1], dtype=np.int32),   # NE
    np.array([1, 0], dtype=np.int32),   # E
    np.array([1, -1], dtype=np.int32),  # SE
    np.array([0, -1], dtype=np.int32),  # S
    np.array([-1, -1], dtype=np.int32), # SW
    np.array([-1, 0], dtype=np.int32),  # W
    np.array([-1, 1], dtype=np.int32),  # NW
]


def bbox_to_center(bbox: np.ndarray) -> np.ndarray:
    """Convert [xmin, ymin, xmax, ymax] to center (x,y)."""
    xmin, ymin, xmax, ymax = bbox
    return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=np.float32)


class UnifiedEnvWrapper(gym.Env):
    def __init__(self, env, env_type: str, grid_size: int = 10, partial_view_size: int = 5):
        """
        env: underlying environment instance
        env_type: one of "discrete", "continuous", "partial"
        grid_size: used for normalization of coordinates if relevant (spatial scale)
        partial_view_size: expected window size for partial observation env
        """
        super().__init__()
        self.env = env
        self.env_type = env_type  # "discrete", "continuous", "partial"
        self.grid_size = grid_size
        self.partial_view_size = partial_view_size

        # Unified action space: 8 high-level directions
        self.action_space = spaces.Discrete(8)

        # Build observation space dynamically
        if env_type == "discrete":
            self.observation_dim = 4  # agent_x, agent_y, exit_x, exit_y normalized
        elif env_type == "continuous":
            self.observation_dim = 5  # agent_x, agent_y, target_x, target_y, orientation_norm
        elif env_type == "partial":
            self.observation_dim = partial_view_size * partial_view_size + 8  # view + heading one-hot
        else:
            raise ValueError(f"Unsupported env_type: {env_type}")
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)

        # For partial env
        if env_type == "partial":
            self.heading = 0  # internal heading
        # For continuous env, cache last seen agent angle (degrees)
        if env_type == "continuous":
            self._last_continuous_angle = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.env_type == "continuous":
            # initialize cached angle from raw observation
            self._last_continuous_angle = float(obs.get("agent_angle", 0.0))
        self._sync_heading_from_obs(obs)
        state_vec = self._process_obs(obs)
        return state_vec, info

    def step(self, action):
        if self.env_type == "discrete":
            raw_action = action  # assumes underlying discrete env uses 0..7 mapping
            raw_obs, reward, terminated, truncated, info = self.env.step(raw_action)
        elif self.env_type == "continuous":
            raw_obs, reward, terminated, truncated, info = self._step_continuous(action)
        elif self.env_type == "partial":
            raw_obs, reward, terminated, truncated, info = self._step_partial(action)
        else:
            raise ValueError(f"Unsupported env_type: {self.env_type}")

        # Update cached orientation if continuous
        if self.env_type == "continuous":
            self._last_continuous_angle = float(raw_obs.get("agent_angle", self._last_continuous_angle))

        self._sync_heading_from_obs(raw_obs)
        state_vec = self._process_obs(raw_obs)
        done = terminated or truncated
        return state_vec, reward, terminated, truncated, info

    def _process_obs(self, obs):
        if self.env_type == "discrete":
            agent = obs["agent"].astype(np.float32)
            exit_ = obs["exit"].astype(np.float32)
            # normalize to [0,1]
            vec = np.concatenate([agent, exit_]) / (self.grid_size - 1)
            return vec.astype(np.float32)

        elif self.env_type == "continuous":
            # ContinuousWorldEnv returns:
            # "agent": bbox via Rectangle __array__()
            # "agent_angle": degrees 0-359
            # "targets": sequence of bboxes
            agent_bbox = np.array(obs["agent"], dtype=np.float32)
            agent_center = bbox_to_center(agent_bbox)  # (2,)
            targets = np.array(obs.get("targets", []), dtype=np.float32)
            if targets.size == 0:
                target_center = np.zeros(2, dtype=np.float32)
            else:
                # compute centers and pick nearest by L1
                target_centers = np.array([bbox_to_center(t) for t in targets], dtype=np.float32)
                dists = np.linalg.norm(target_centers - agent_center, ord=1, axis=1)
                target_center = target_centers[np.argmin(dists)]
            # normalize spatially
            agent_norm = agent_center / (self.grid_size - 1)
            target_norm = target_center / (self.grid_size - 1)
            angle_deg = float(obs.get("agent_angle", 0.0))  # 0..359
            angle_rad = math.radians(angle_deg)
            orient_norm = np.array([angle_rad / math.pi], dtype=np.float32)  # in [-1,1]
            return np.concatenate([agent_norm, target_norm, orient_norm]).astype(np.float32)

        elif self.env_type == "partial":
            view = obs.get("view")
            if view is None:
                raise KeyError("Partial env observation missing 'view'")
            view_flat = np.array(view, dtype=np.float32).flatten()
            # normalize view (avoid divide by zero)
            maxv = np.max(view_flat)
            norm_view = view_flat / maxv if maxv > 0 else view_flat
            h = self.heading % 8
            heading_onehot = np.zeros(8, dtype=np.float32)
            heading_onehot[h] = 1.0
            return np.concatenate([norm_view, heading_onehot]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported env_type: {self.env_type}")

    def _sync_heading_from_obs(self, obs):
        if self.env_type == "partial":
            if "heading" in obs:
                self.heading = int(obs["heading"]) % 8

    def _step_continuous(self, direction_idx):
        """
        High-level discrete 8-way direction -> continuous env atomic actions:
        0 = N, 1 = NE, ..., 7 = NW. ContinuousWorldEnv action space is:
          0: forward
          1: turn left (+15°)
          2: turn right (-15°)
        We'll compute desired angle in degrees so that:
          direction 0 (N) maps to 90°, 2 (E) -> 0°, 4 (S) -> 270°, etc.
        """
        # Desired angle in degrees (0 is +x, increasing counterclockwise)
        desired_angle_deg = (90 - direction_idx * 45) % 360
        current_angle_deg = self._last_continuous_angle
        # Shortest delta in [-180,180]
        delta = ((desired_angle_deg - current_angle_deg + 180) % 360) - 180

        if abs(delta) > 10:  # need to rotate
            if delta > 0:
                # turn left (angle increases by +15)
                raw_obs, reward, terminated, truncated, info = self.env.step(1)
                # update cached angle
                self._last_continuous_angle = (current_angle_deg + 15) % 360
            else:
                # turn right (angle decreases by 15)
                raw_obs, reward, terminated, truncated, info = self.env.step(2)
                self._last_continuous_angle = (current_angle_deg - 15) % 360
        else:
            # move forward
            raw_obs, reward, terminated, truncated, info = self.env.step(0)
            # angle unchanged

        return raw_obs, reward, terminated, truncated, info

    def _step_partial(self, direction_idx):
        # Approximate desired heading with forward/left/right
        desired_heading = direction_idx % 8
        diff = (desired_heading - self.heading) % 8
        if diff == 0:
            action = 0  # forward
        elif diff <= 4:
            action = 2  # turn right
            self.heading = (self.heading + 1) % 8
        else:
            action = 1  # turn left
            self.heading = (self.heading - 1) % 8
        return self.env.step(action)
