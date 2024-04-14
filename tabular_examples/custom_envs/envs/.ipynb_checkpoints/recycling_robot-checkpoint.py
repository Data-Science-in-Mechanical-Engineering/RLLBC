import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from screeninfo import get_monitors
from typing import Optional
from os import path
import time
import pygame
import gymnasium as gym


class RecyclingRobotEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "text", "None"],
        "render_type": ["robot", "node", "None"],
        "render_fps": 4,
    }
    def __init__(self, alpha=0.8, beta=0.1, duration=20, render_type="None", render_mode="text", r_search=3, r_broken=-3,
                 r_wait=1, r_recharge=0, initial_state=1, render_time=1.5):
        """
        render_mode: human = Display, rgb_array = saving rgb array, text = Displaying text
        render_type: robot = robot animation, node = simple node display
        """
        self.observation_space = spaces.Discrete(2)  # 0 = low, 1 = high
        self.action_space = spaces.Discrete(3)  # 0 = search, 1 = wait, 2 = recharge
        self.alpha = alpha  # probability of keeping high power level when searching
        self.beta = beta  # probability of keeping low power level when searching
        self.r_search = r_search
        self.r_wait = r_wait
        self.r_recharge = r_recharge
        self.r_broken = r_broken
        self.state = initial_state
        self.render_mode = render_mode
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
            )
        self.render_type = render_type
        if self.render_mode != "text" and self.render_type == "node":
            self.window_size = (get_monitors()[0].height - 200, get_monitors()[0].height - 500)
            self.cell_size = (self.window_size[0], self.window_size[1])
            self.render_time = render_time
        elif self.render_mode != "text" and self.render_type == "robot":
            self.window_size = (get_monitors()[0].height - 50, get_monitors()[0].height - 50)
            self.cell_size = (self.window_size[0], self.window_size[1])
            self.render_time = render_time
        self.window_surface = None
        self.rb_broken = None
        self.rb_high = None
        self.rb_high_searching_high = None
        self.rb_high_searching_low = None
        self.rb_high_waiting = None
        self.rb_low = None
        self.rb_low_searching_high = None
        self.rb_low_searching_low = None
        self.rb_low_waiting = None
        self.rb_recharging = None
        # State transition probability:
        # first: state we are in, second: action we can pick, followed by a list containing all the possible states
        # List follows structure: [(probability, state we could enter, reward, termination),(...),...]
        self.P = {0: {0: [(self.beta, 0, self.r_search, False), (1 - self.beta, 1, self.r_broken, True)],
                      1: [(1, 0, self.r_wait, False)],  # Added the possibility of the robot breaking when waiting
                      2: [(1, 1, self.r_recharge, False)]
                      },
                  1: {0: [(self.alpha, 1, self.r_search, False), (1 - self.alpha, 0, self.r_search, False)],
                      1: [(1, 1, self.r_wait, False)]
                      }
                  }
        self.duration = duration  # max length of the episode
        self.time = 0
        self.lastaction = None
        self.laststate = None

    def reset(self, seed=None, options: Optional[dict] = None):
        self.state = 1  # going back to high
        self.time = 0  # set timer back to 0
        self.lastaction = None
        self.laststate = None
        return self.state, {}

    def step(self, a):
        transitions = self.P[self.state][int(a)]
        self.time += 1
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        self.lastaction = a
        self.laststate = self.state
        p, s, r, t = transitions[i]
        self.state = s
        # rendering s --> a --> s'
        if not t:
            self.render()
        else:
            # This was implemented since only the "robot" has an animation for the state of being "broken"
            # In all other cases the environment is simply reset
            if self.render_mode == "human" and self.render_type == "robot":
                # hacky solution for displaying the robot breaking
                self.window_surface.blit(self.rb_low, (0, 0))
                pygame.event.pump()
                pygame.display.update()
                time.sleep(self.render_time)
                self.window_surface.blit(self.rb_low_searching_low, (0, 0))
                pygame.event.pump()
                pygame.display.update()
                time.sleep(self.render_time)
                self.window_surface.blit(self.rb_broken, (0, 0))
                pygame.event.pump()
                pygame.display.update()
                time.sleep(self.render_time)
            elif self.render_mode == "text":
                self.render()
                print('Robot needs a manual recharge')
            else:
                self.render()
        if self.time >= self.duration:
            t = True
            #print('Reached time-out')
            s, p = self.reset()
        return (int(s), r, t, False, {})

    def action_meaning(self, action):
        if action == 0:
            return 'searching'
        if action == 1:
            return 'waiting'
        if action == 2:
            return 'recharging'

    def state_meaning(self, state):
        if state == 0:
            return 'low'
        if state == 1:
            return 'high'

    def render(self):
        if self.render_mode == "None":
            return
        elif self.render_mode == 'text':
            print('Robot battery is ', self.state_meaning(self.laststate), ', and now it does ',
                  self.action_meaning(self.lastaction))
            return
        else:  # self.render_mode is "human" or "rgb_array" and requires pygame
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy_text]`"
            )

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Recycling Robot")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        # Load the images
        if self.render_type == "robot":
            if self.rb_broken is None:
                file_name = path.join(path.dirname(__file__), "img/rb_broken.png")
                self.rb_broken = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high is None:
                file_name = path.join(path.dirname(__file__), "img/rb_high.png")
                self.rb_high = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high_searching_high is None:
                file_name = path.join(path.dirname(__file__), "img/rb_high_searching.png")
                self.rb_high_searching_high = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high_searching_low is None:
                file_name = path.join(path.dirname(__file__), "img/rb_high_searching.png")
                self.rb_high_searching_low = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high_waiting is None:
                file_name = path.join(path.dirname(__file__), "img/rb_high_waiting.png")
                self.rb_high_waiting = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low is None:
                file_name = path.join(path.dirname(__file__), "img/rb_low.png")
                self.rb_low = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low_waiting is None:
                file_name = path.join(path.dirname(__file__), "img/rb_low_waiting.png")
                self.rb_low_waiting = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low_searching_high is None:
                file_name = path.join(path.dirname(__file__), "img/rb_low_searching.png")
                self.rb_low_searching_high = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low_searching_low is None:
                file_name = path.join(path.dirname(__file__), "img/rb_low_searching.png")
                self.rb_low_searching_low = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_recharging is None:
                file_name = path.join(path.dirname(__file__), "img/rb_recharging.png")
                self.rb_recharging = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
        if self.render_type == "node":
            if self.rb_high is None:
                file_name = path.join(path.dirname(__file__), "img/node_high.jpg")
                self.rb_high = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high_searching_high is None:
                file_name = path.join(path.dirname(__file__), "img/node_high_search_high.jpg")
                self.rb_high_searching_high = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high_searching_low is None:
                file_name = path.join(path.dirname(__file__), "img/node_high_search_low.jpg")
                self.rb_high_searching_low = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_high_waiting is None:
                file_name = path.join(path.dirname(__file__), "img/node_high_wait_high.jpg")
                self.rb_high_waiting = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low is None:
                file_name = path.join(path.dirname(__file__), "img/node_low.jpg")
                self.rb_low = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low_waiting is None:
                file_name = path.join(path.dirname(__file__), "img/node_low_wait_low.jpg")
                self.rb_low_waiting = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low_searching_high is None:
                file_name = path.join(path.dirname(__file__), "img/node_low_search_high.jpg")
                self.rb_low_searching_high = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_low_searching_low is None:
                file_name = path.join(path.dirname(__file__), "img/node_low_search_low.jpg")
                self.rb_low_searching_low = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.rb_recharging is None:
                file_name = path.join(path.dirname(__file__), "img/node_low_recharge_high.jpg")
                self.rb_recharging = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
        rgb_arrays = []
        # Displaying the step following s, a, s' (self.laststate, self.lastaction, self.state) in each step
        # Display s and a
        if self.laststate == 0:
            self.window_surface.blit(self.rb_low, (0, 0))
            rgb_array = self._load_render(mode)
            rgb_arrays.append(rgb_array)
            if self.lastaction == 0:
                if self.state == 0: # Goes back to low after searching
                    self.window_surface.blit(self.rb_low_searching_low, (0, 0))
                else:
                    self.window_surface.blit(self.rb_low_searching_high, (0, 0))
            elif self.lastaction == 1:
                self.window_surface.blit(self.rb_low_waiting, (0, 0))
            elif self.lastaction == 2:
                self.window_surface.blit(self.rb_recharging, (0, 0))
            rgb_array = self._load_render(mode)
            rgb_arrays.append(rgb_array)
        elif self.laststate == 1:
            self.window_surface.blit(self.rb_high, (0, 0))
            rgb_array = self._load_render(mode)
            rgb_arrays.append(rgb_array)
            if self.lastaction == 0:
                if self.state == 1:
                    self.window_surface.blit(self.rb_high_searching_high, (0, 0))
                else:
                    self.window_surface.blit(self.rb_high_searching_low, (0, 0))
            elif self.lastaction == 1:
                self.window_surface.blit(self.rb_high_waiting, (0, 0))
            rgb_array = self._load_render(mode)
            rgb_arrays.append(rgb_array)
        # Display s'
        if self.state == 0:
            self.window_surface.blit(self.rb_low, (0, 0))
            rgb_array = self._load_render(mode)
            rgb_arrays.append(rgb_array)
        elif self.state == 1:
            self.window_surface.blit(self.rb_high, (0, 0))
            rgb_array = self._load_render(mode)
            rgb_arrays.append(rgb_array)

        if not all(arr is None for arr in rgb_arrays):
            return rgb_arrays

    def _load_render(self, mode):
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            time.sleep(self.render_time)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
