import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.spaces import Box
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

class CustomPendulumEnv(PendulumEnv):
    def __init__(self):
        super().__init__()
        super().reset()
        high = np.asarray([np.pi, self.max_speed])
        self.observation_space = Box(
            low=np.float32(-high),
            high=np.float32(high),
            dtype=np.float32)

    def step(self, action):
        super().step([action])
        state = self.state
        reward = (np.cos(state[0]) - 1) - 0.02*np.abs(state[1])
        while state[0] < -np.pi:
            state[0] += 2 * np.pi
        while state[0] > np.pi:
            state[0] -= 2 * np.pi
        return np.float32(state), np.squeeze(reward), False, {}, {}

    def reset(self):
        state = np.array([2 * (np.random.rand() - 0.5) * np.pi, 2 * (np.random.rand() - 0.5) * self.observation_space.high[1]])
        self.state = state
        info = {}
        return state, info
