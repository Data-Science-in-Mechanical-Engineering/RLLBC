import numpy as np
import gymnasium as gym
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.spaces import Discrete, Box
from gym.envs.registration import register


class CustomPendulumEnv(PendulumEnv):
    def __init__(self):
        super().__init__()
        super().reset()
        high = np.asarray([np.pi, self.max_speed])
        self.observation_space = Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def step(self, action):
        super().step([action])
        state = self.state
        reward = (np.cos(state[0]) - 1) - 0.02*np.abs(state[1])
        while state[0] < -np.pi:
            state[0] += 2 * np.pi
        while state[0] > np.pi:
            state[0] -= 2 * np.pi
        return state, np.squeeze(reward), False, {}

    def reset(self):
        state = np.array([2 * (np.random.rand() - 0.5) * np.pi, 2 * (np.random.rand() - 0.5) * self.observation_space.high[1]])
        self.state = state
        return state


register(
    id='CustomPendulum-v0',
    entry_point='CustomPendulumEnv:CustomPendulumEnv',
    max_episode_steps=200,
)
