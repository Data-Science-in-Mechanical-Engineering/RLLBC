# Deep Reinforcement Learning — Algorithm Library

This folder contains notebook implementations of the core deep RL algorithms covered in the RLLBC course. The notebooks progress from value-based methods through policy gradients to continuous-action actor-critic algorithms. All implementations are built in PyTorch and include TensorBoard logging.

Students are encouraged to modify hyperparameters, swap environments, and compare training curves to deepen their understanding.

---

## Table of Contents

1. [Value-Based Methods](#1-value-based-methods)
2. [Policy Gradient Methods](#2-policy-gradient-methods)
3. [On-Policy Actor-Critic](#3-on-policy-actor-critic)
4. [Off-Policy Actor-Critic](#4-off-policy-actor-critic)

---

## 1. Value-Based Methods

### [dqn.ipynb](dqn.ipynb)

Implements **Deep Q-Network (DQN)** on the Atari Breakout environment. Combines a convolutional Q-network with experience replay and a target network to stabilise training on raw pixel observations.

- **Course context:** Deep RL — Value-Based Methods / DQN

---

## 2. Policy Gradient Methods 

Policy gradient methods directly optimise the policy parameters by gradient ascent on expected return. All notebooks in this section train on `CartPole-v1`.

### [reinforce.ipynb](reinforce.ipynb)

Implements vanilla **REINFORCE** — the foundational Monte Carlo policy gradient algorithm. Updates the policy using the full episodic return as a reward signal with no baseline.

- **S&B reference:** Section 13.3 (REINFORCE: Monte Carlo Policy Gradient)

---

## 3. On-Policy Actor-Critic



### [a2c-simple-adv.ipynb](a2c-simple-adv.ipynb)

Introduces **Advantage Actor-Critic (A2C)** with a simple n-step advantage estimate `G − V(s)`. Adds a learned value baseline to the REINFORCE update to reduce variance.

- **S&B reference:** Section 13.5 (Actor-Critic Methods)

---

### [a2c.ipynb](a2c.ipynb)

Full **A2C** implementation using a generalized advantage estimate (GAE) and separate actor and critic networks. Extends the simple variant with more stable multi-step bootstrapping.

- **S&B reference:** Section 13.5 (Actor-Critic Methods)

---

### [trpo-simple-adv.ipynb](trpo-simple-adv.ipynb)

Introduces **Trust Region Policy Optimisation (TRPO)** with a simple advantage estimator. Constrains the policy update step size via a KL-divergence trust region to prevent destructive large updates.

- **Course context:** Deep RL — Trust Region Methods (Schulman et al. 2015)

---

### [trpo.ipynb](trpo.ipynb)

Full **TRPO** implementation with **Generalised Advantage Estimation (GAE)**, combining multi-step returns with an exponentially weighted baseline for lower-variance advantage estimates.

- **Course context:** Deep RL — Trust Region Methods + GAE (Schulman et al. 2015, 2016)

---

## 4. Off-Policy Actor-Critic

Actor-critic algorithms for continuous action spaces, all trained on `Pendulum-v1`.

### [ddpg.ipynb](ddpg.ipynb)

Implements **Deep Deterministic Policy Gradient (DDPG)** — an off-policy actor-critic method that extends DQN to continuous actions using a deterministic policy and experience replay.

- **Course context:** Deep RL — Continuous Control (Lillicrap et al. 2015)

---

### [td3.ipynb](td3.ipynb)

Implements **Twin Delayed Deep Deterministic Policy Gradient (TD3)**, which improves DDPG stability via clipped double Q-learning, delayed policy updates, and target policy smoothing.

- **Course context:** Deep RL — Continuous Control (Fujimoto et al. 2018)

---

### [sac.ipynb](sac.ipynb)

Implements **Soft Actor-Critic (SAC)** — an off-policy algorithm that maximises both return and policy entropy, yielding strong sample efficiency and robust exploration in continuous action spaces.

- **Course context:** Deep RL — Maximum Entropy RL (Haarnoja et al. 2018)

---
## Hierarchical structure of the `logs` folder
The deep reinforcement learning examples have different types of logs saved every time you run an experiment (or, let's say, the training loop). The types of logs saved can be set in the **Training Params & Agent Hyperparams** section of each notebook. We refer to the individual notebooks for further details on this.

The notebooks support the creation of a hierarchical logs folder structure to keep track of experiments and sub-experiments. Here's a brief overview of how this can be done:
- The root log folder is called `logs`. If it does not exist, it is created at the same directory level as the notebook. 
- `exp.exp_name` and `exp.exp_type` allow you to control the hierarchical folder creation. 
    - If `exp.type = None`, logs of that experiment are saved to a folder with the name `exp.exp_name` inside the root folder `logs`, ie, `/logs/<exp.run_name>`
    - If `exp.type = "learning_rate"`, here learning_rate is your sub-experiment, and the logs are saved to the directory `/logs/learning_rate/<exp.run_name>`
    
## Grouping of logged parameters in Notebooks

| Group           | Parameter/Hyperparameter |
|-----------------|--------------------------|
| rollout         | episodic_return          | 
|                 | epsiodic_length          |
| hyperparameters | learning_rate            |
|                 | epsilon (dqn exploration)|
|                 | alpha (entropy temperature)|
| train           | value_loss               |
|                 | policy_loss              |
|                 | entropy                  |
|                 | kl_divergence            |
|                 | kl_divergence02          |
|                 | q_values                 |
|                 | q1_values                |
|                 | q2_values                |
|                 | q1_loss                  |
|                 | q2_loss                  |
|                 | q_loss                   |
|                 | alpha_loss               |
|                 | clipfrac                 |
|                 | explained_variance       |
|                 | is_line_search_success   |
|                 | surrogate_objective      |
| Charts          | episode_step             |
|                 | gradient_step            |
| others          | SPS                      |
