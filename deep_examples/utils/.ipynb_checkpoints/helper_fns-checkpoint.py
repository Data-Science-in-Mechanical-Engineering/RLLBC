# Adapted from: https://github.com/DLR-RM/stable-baselines3

import os
import warnings
import collections
from typing import Callable, List, Dict, SupportsFloat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ruamel.yaml import YAML
from easydict import EasyDict as edict
from gymnasium import spaces

import gymnasium as gym
import wandb
import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]


'''
Here is a list of important helper functions (in no particular order). You may come back to this later as you start getting more into the different sections of the notebook. For now, here a quick preview:
1. `make_single_env`, `make_env` - helps in creating a single gym environment
2. `make_env` - helps in creating a vectorized gym environment
3. `setup_logging` - setup Weights & Biases and TensorBoard logging
4. `save_train_config_to_yaml`, `save_model` and `save_tracked_values` - at the end training
5. `evaluate_agent` - used to evaluate agent performance during or post-training
6. plotter functions to generate plot in the section **Compare Trained Agents and Display Behaviour**
7. 'make_atari_env' - helps in creating a vectorized gym environment for the Breakeout Atari game
'''


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

def make_single_env(env_id, seed):
    """
    Creates a single instance of a Gym environment with the given ID and seed
    Args:
        env_id: string containing the ID of the Gym environment to create
        seed: integer seed value to use for the environment's random number generator
    Returns:
        A Gym environment object
    """
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def make_env(env_id, seed):
    """
    Returns a thunk that creates and initializes a gym environment with the given ID and seed
    Args:
        env_id: string identifying the gym environment to create
        seed: integer specifying the random seed to use for the environment
    Returns:
        callable thunk that creates and returns a gym environment with a seeded initial state, action space, and observation spaces
    """

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def make_atari_breakout_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def wandb_logging(wandb_prj_name, run_name, config=None, save_code=False):
    """
    Sets up and initializes a Weights & Biases (wandb) logging session for the given notebook and run.
    Args:
        wandb_prj_name: string specifying the name of the wandb project to log to
        run_name: string specifying the name of the wandb run to create
        config: optional dictionary of experiment configuration values to log to wandb
        save_code: boolean indicating whether to save the code associated with the run to wandb
    Returns:
        None
    """
    log_dir = make_log_dir()
    wandb.login()
    wandb.init(
        dir=log_dir,
        project=wandb_prj_name,
        sync_tensorboard=True,
        name=run_name,
        config=config,
        save_code=save_code,
        )


def setup_logging(wandb_prj_name, exp_dict, hypp_dict):
    """
    Sets up and initializes logging for an experiment, including wandb and TensorBoard Summary writer.
    Args:
        wandb_prj_name: string specifying the name of the wandb project to log to.
        exp_dict: dict containing experiment-specific configuration.
        hypp_dict: dict containing hyperparameters for the experiment.
    Returns:
        TensorBoard SummaryWriter instance for logging training data to TensorBoard.
    """
    if wandb.run is not None:
        wandb.finish()
    if exp_dict.enable_wandb_logging:
        wandb_logging(wandb_prj_name, exp_dict.run_name, dict(exp_dict, **hypp_dict))
    else:
        wandb.init(mode="disabled")
    exp_folder = "" if exp_dict.exp_type is None else exp_dict.exp_type
    tb_writer = SummaryWriter(f"logs/{exp_folder}/{exp_dict.run_name}/tb")
    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in dict(exp_dict, **hypp_dict).items()])),
    )
    return tb_writer


def create_folder_relative(folder_name, assert_flag=False):
    """
    Creates a folder with the given name in the current directory and returns the absolute path to the folder. The current directory is wrt the directory of the notebook that calls this function
    Args:
        folder_name: A string specifying the name of the folder to create.
        assert_flag: A boolean indicating whether to raise an AssertionError if the folder already exists.
    Returns:
        A tuple containing:
        - The absolute path to the newly created folder or existing folder.
        - A boolean indicating whether the folder already existed.
    """
    log_dir = make_log_dir()
    abs_folder_path = os.path.abspath(f"{log_dir}/{folder_name}")
    if not os.path.isdir(abs_folder_path):
        if assert_flag:
            assert False, "Following folder does not exist %s" % (abs_folder_path)
        os.makedirs(abs_folder_path)
        folder_already_exist = False
    else:
        folder_already_exist = True
    return abs_folder_path, folder_already_exist


def make_log_dir():
    log_dir = os.path.abspath('logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir


def save_train_config_to_yaml(exp_dict, hypparam_dict):
    """
    Saves the experiment and hyperparameter configurations to a YAML file in the respective log folder
    Args:
        exp_dict: dict containing experiment configuration settings
        hypparam_dict: dict containing hyperparameters
    Returns:
        None
    """
    exp_folder = "" if exp_dict.exp_type is None else exp_dict.exp_type
    folder_path, _ = create_folder_relative(f"{exp_folder}/{exp_dict.run_name}")
    yaml = YAML()
    yaml.default_flow_style = False
    file_full_path = f"{folder_path}/experiment_config.yml"
    with open(file_full_path, 'w') as f:
        yaml.dump(dict(experiment_parameters=dict(exp_dict), hyperparameters=dict(hypparam_dict)), f)


def save_tracked_values(returns_over_runs, episode_len_over_runs, episode_list, eval_count, run_name, exp_type=None):
    """
    Saves tracked performance metrics to a CSV file in a folder inside logs with the given run name.
    Args:
        returns_over_runs: list of floats containing the returns for each evaluation run
        episode_len_over_runs: list of integers containing the episode lengths for each evaluation run
        episode_list: 2D numpy array containing the episode and global step number for each evaluation step
        eval_count: integer specifying the number of evaluation runs performed
        run_name: string containing the name of the current experiment run
        exp_type: subdirectory to where the run is saved
    Returns:
        None
    """

    exp_folder = "" if exp_type is None else exp_type

    eval_id = [f"eval_idx{i:02d}" for i in range(eval_count)]
    sub_run_index = np.repeat([eval_id], len(returns_over_runs), axis=0).reshape(-1)

    episode_list = np.repeat(episode_list, eval_count, axis=0)
    returns_over_runs = np.array(returns_over_runs).reshape(-1)
    episode_len_over_runs = np.array(episode_len_over_runs).reshape(-1)

    df = pd.DataFrame(data=[episode_list[:, 0], episode_list[:, 1], sub_run_index, returns_over_runs, episode_len_over_runs]).T
    df.columns = ['episode', 'global_step', 'sub_run_index', 'returns', 'episode_length']

    folder_path, _ = create_folder_relative(f"{exp_folder}/{run_name}")
    csv_full_path = f"{folder_path}/tracked_performance_training.csv"
    with open(csv_full_path, 'wb') as f:
        df.to_csv(f, index=False)


def save_model(model, run_name, exp_type=None, print_path=True):
    exp_folder = "" if exp_type is None else exp_type
    folder_path, _ = create_folder_relative(f"{exp_folder}/{run_name}")
    model_full_path = f"{folder_path}/agent_model.pt"
    with open(model_full_path, 'wb') as f:
        torch.save(model, f)
    if print_path:
        print(f"Agent model saved to path: \n{model_full_path}")


def load_model(run_name=None, folder_path=None, exp_type=None):
    exp_folder = "" if exp_type is None else exp_type
    if run_name is None:
        raise Exception("input run_name missing")
    if folder_path is None:
        folder_path, path_exi = create_folder_relative(f"{exp_folder}/{run_name}", assert_flag=True)
    model_full_path = f"{folder_path}/agent_model.pt"
    model = torch.load(model_full_path,
                       map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def evaluate_agent(envs, model, run_count, seed, greedy_actor=False):
    """
    Evaluate an agent on vectorized environment and return the returns and episode lengths of each run
    Args:
        envs: vectorized gym environment
        model: agent's policy model
        run_count: integer value specifying the number of runs to evaluate the agent
        seed: integer value representing the initial random seed
        greedy_actor: boolean flag that controls whether to use a greedy policy or not

    Returns:
        returns_over_runs: list of floats, representing the return of each run
        episode_len_over_runs: list of integers, representing the episode length of each run
    """
    next_obs = torch.Tensor(envs.reset()[0]).to("cuda" if torch.cuda.is_available() else "cpu")
    returns_over_runs = []
    episode_len_over_runs = []
    finish = False
    envs.reset(seed=list(range(seed, seed+envs.num_envs)))
    model.eval()
    while not finish:
        with torch.no_grad():
            actions = model.get_action(next_obs, greedy_actor)
        next_obs, rewards, _, _, info = envs.step(actions.cpu().numpy())
        next_obs = torch.Tensor(next_obs).to("cuda" if torch.cuda.is_available() else "cpu")
        if '_episode' in info and 'episode' in info:
            for i, item in enumerate(info['_episode']):
                if item: 
                    returns_over_runs.append(info['episode']['r'][i])
                    episode_len_over_runs.append(info['episode']['l'][i])
                    if run_count == len(returns_over_runs):
                        finish = True
                        break
    model.train()
    return returns_over_runs, episode_len_over_runs


def plotter_agents_training_stats(exp_settings, agent_labels=None, episode_axis_limit=None, plot_returns=True, plot_episode_len=True):
    """
    Plots the average training statistics over episodes for multiple agents
    Args:
        exp_settings: Dictionary containing the settings for the experiment, including the agent names and the corresponding experiment type
        agent_labels: List of labels to use for the agents. If None, the function generates labels based on the agent names
        episode_axis_limit: Upper limit for the episode axis. If None, the function uses the maximum episode number
        plot_returns: If True, the function plots the average return for each episode
        plot_episode_len: If True, the function plots the average episode length for each episode
    Returns:
        None
    """
    plt.style.use('ggplot')
    dfList = []
    run_names = [exp_settings[key] for key in exp_settings.keys() if 'run_name' in key]
    exp_types = [exp_settings[key] for key in exp_settings.keys() if 'exp_type' in key]

    assert len(run_names)==len(exp_types), "count of experiment names is not equal to count of experiment types!"

    for idx, (exp_type, run_name) in enumerate(zip(exp_types, run_names)):
        if exp_type is None:
            exp_type = ""
        folder_path, _ = create_folder_relative(f"{exp_type}/{run_name}")
        full_path = f"{folder_path}/tracked_performance_training.csv"
        if os.path.isfile(full_path):
            dfList.append(pd.read_csv(full_path))
        else:
            raise Exception("Can't find filename%.2d: %s.csv in folder: %s" % (idx, run_name, folder_path))

    if agent_labels is None or not agent_labels:
        agent_labels = generate_agent_labels(exp_settings, agent_abbrevation=False)
    elif agent_labels is not None and len(agent_labels) != len(dfList):
        raise Exception("Expected %d labels but got %d" % (len(dfList), len(agent_labels)))

    cols = int(plot_returns+plot_episode_len)
    fig, axes = plt.subplots(1, cols, figsize=(7*cols, 3))

    for idx, df in enumerate(dfList):
        if plot_returns:
            ax = axes[0] if isinstance(axes, collections.abc.Iterable) else axes
            sns.lineplot(data=df, x='episode', y='returns', ax=ax, label=agent_labels[idx], legend=False)
            ax.set(xlabel='Episode Number',
                   ylabel='Average Return',)
            transform = ax.transAxes
        if plot_episode_len:
            ax = axes[1] if isinstance(axes, collections.abc.Iterable) else axes
            sns.lineplot(data=df, x='episode', y='episode_length', ax=ax, label=agent_labels[idx], legend=False)
            ax.set(xlabel='Episode Number',
                   ylabel='Average Episode Length',)
    fig.suptitle("Average Statistics Over Episodes During Training")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -0.14),  bbox_transform=transform,
               ncol=1, fancybox=True, shadow=True)
    plt.setp(axes, xlim=(None, episode_axis_limit))
    plt.show()


def generate_agent_labels(exp_settings, agent_abbrevation=False):
    labels = []
    for key in exp_settings.keys():
        if 'run_name' in key:
            labels.append(exp_settings[key].replace('__', ', '))
    if agent_abbrevation:
        labels = [f"a{idx:02d}: {label}" for idx, label in enumerate(labels)]
    return labels


def record_video(env_id, agent, file, exp_type=None, greedy=False, env_wrapper: List[Callable]= []):
    """
    Records one episode of the agent acting on the environment env_id and saving the video to file.
    Args:
        env_id: The environment id for the agent to run in e.g. Cartpole-v1
        agent: Either a pytorch model or name of a finished experiment
        file: the file to which the video is written to
        exp_type: the experiment type to which the agent is belonging (if it is provided by name string instead of model)
        greedy: whether the agent performs actions in a greedy way
    Return:
        None
    """
    frames = []
    env = gym.make(env_id, render_mode='rgb_array')
    for wrapper in env_wrapper:
        env = wrapper(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type(agent) == str:
        agent = load_model(run_name=agent, exp_type=exp_type)

    state, done = env.reset(), False
    state = state[0]
    while not done:
        with torch.no_grad():
            action = agent.get_action(torch.Tensor(state).unsqueeze(0).to(device), greedy=greedy)

        state, _, terminated, _, info = env.step(action.squeeze(0).cpu().numpy())

        if terminated:
            done = True

        out = env.render()
        frames.append(out)

    env.close()

    # First set up the figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    img = plt.imshow(frames[0])

    def animate(frame):
        img.set_data(frames[frame])
        return [img]

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
    plt.close()

    anim.save(file, writer="ffmpeg", fps=30)


def save_and_log_agent(exp_dict, agent, episode_step, greedy=False, print_path=True, env_wrapper: List[Callable]=[]):
    """
    Saves the agent model and records a video if video recording is enabled. Logs video to WandB if WandB is enabled.
    Args:
        exp_dict: Dictionary of experiment parameters.
        agent: Agent model to be saved.
        episode_step: Episode step
        greedy: Whether to use a greedy policy for video. Defaults to False
        print_path: Boolean flag to print model path. Defaults to True
        env_wrapper: List of environment wrappers. Defaults to empty list
    Returns:
        None
    """

    save_model(agent, exp_dict.run_name, exp_type=exp_dict.exp_type, print_path=print_path)
    exp_folder = "" if exp_dict.exp_type is None else exp_dict.exp_type
    if exp_dict.capture_video:
        filepath, _ = create_folder_relative(f"{exp_folder}/{exp_dict.run_name}/videos")
        video_file = f"{filepath}/{episode_step}.mp4"
        record_video(exp_dict.env_id, agent, video_file, greedy=greedy, env_wrapper=env_wrapper)
        if wandb.run is not None:
            wandb.log({"video": wandb.Video(video_file, fps=4, format="gif")})


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info

class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))