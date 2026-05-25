# Tabular Reinforcement Learning — Algorithm Library

This folder contains notebook implementations of the core tabular RL algorithms covered in the RLLBC course. All notebooks follow **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)** and use the custom Gymnasium environments from [`custom_envs/`](custom_envs/).

Students are encouraged to modify hyperparameters, swap environments, and experiment with the code to deepen their understanding.

> **Sutton & Barto** — *Reinforcement Learning: An Introduction*, 2nd ed. (2018)  
> Available free at [incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)

---

## Table of Contents

1. [Dynamic Programming](#1-dynamic-programming)
2. [Monte Carlo Methods](#2-monte-carlo-methods)
3. [Temporal Difference Learning](#3-temporal-difference-learning)

---

## 1. Dynamic Programming

DP algorithms compute the optimal policy by iterating the Bellman equations. They require **full knowledge of the transition dynamics and reward function**. Both notebooks use the `RecyclingRobot-v1` environment.

### [policy_iteration.ipynb](dynamic_programming/policy_iteration.ipynb)

Alternates between **Policy Evaluation** (iterative Bellman expectation updates until convergence) and **Policy Improvement** (greedy update) until the policy stabilises. Implements both in-place and non-in-place evaluation variants.

- **S&B reference:** Section 4.3 (Policy Iteration)

---

### [value_Iteration.ipynb](dynamic_programming/value_Iteration.ipynb)

Collapses evaluation and improvement into a single **Bellman optimality sweep** per iteration, avoiding the cost of running evaluation to full convergence. The greedy policy is extracted from the converged value function.

- **S&B reference:** Section 4.4 (Value Iteration)

---

## 2. Monte Carlo Methods

MC methods learn from **complete episodes** without requiring a model. Both notebooks implement MC Control with ε-greedy exploration on `CustomFrozenLake-v1` and visualise the learned Q-function as a triangle-grid heatmap.

### [first_visit_mc.ipynb](tabular_rl/MC/first_visit_mc.ipynb)

Updates Q(s, a) using the return from the **first visit** to each (state, action) pair per episode; subsequent visits within the same episode are skipped.

- **S&B reference:** Section 5.4 (MC Control with Exploring Starts)

---

### [every_visit_mc.ipynb](tabular_rl/MC/every_visit_mc.ipynb)

Updates Q(s, a) using the return from **every visit** to each (state, action) pair per episode. All other aspects are identical to the first-visit variant.

- **S&B reference:** Section 5.1 (MC Prediction)

---

## 3. Temporal Difference Learning

TD methods combine the sampling of MC with the bootstrapping of DP, learning online from incomplete episodes without a model.

### [sarsa.ipynb](tabular_rl/TD/sarsa.ipynb)

**On-policy** TD control on `CliffWalking-v1`. The same ε-greedy policy drives both action selection and the TD target `R + γ Q(S', A')`, so SARSA converges to the optimal policy among ε-greedy policies and finds safer paths around the cliff.

- **S&B reference:** Section 6.4 (SARSA), Example 6.6 (Cliff Walking)

---

### [q-learning.ipynb](tabular_rl/TD/q-learning.ipynb)

**Off-policy** TD control on `CustomFrozenLake-v1`. The TD target uses the greedy next action `max Q(S', ·)` regardless of the behaviour policy, directly approximating the optimal Q*.

- **S&B reference:** Section 6.5 (Q-Learning)

---

### [dyna-q.ipynb](tabular_rl/TD/dyna-q.ipynb)

Augments Q-Learning with a **learned environment model** on `CustomFrozenLake-v1`. After each real step, additional planning updates are sampled from the model, dramatically reducing the number of real environment interactions needed to converge.

- **S&B reference:** Section 8.2 (Dyna-Q)

---