import numpy as np
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


class CustomPendulumEnvDiscrete(CustomPendulumEnv):
    def __init__(self):
        super().__init__()
        super().reset()
        self.action_space = Discrete(5)

    def step(self, action):
        act = (action - 2.0)
        return super().step(act)


register(
    id='CustomPendulum-v0',
    entry_point='custompendulumenv:CustomPendulumEnv',
    max_episode_steps=200,
)

register(
    id='CustomPendulumDiscrete-v0',
    entry_point='custompendulumenv:CustomPendulumEnvDiscrete',
    max_episode_steps=100,
)
