# Class Examples

This folder contains all lecture and exercise companion notebooks for the course **Reinforcement Learning and Learning-based Control (RLLBC)** at RWTH Aachen University (DSME).

The notebooks follow the progression of the course from foundational MDP theory through tabular RL to function approximation and learning-based control. Each section references the corresponding chapters and examples in the primary RL textbook:

> **Sutton & Barto** — *Reinforcement Learning: An Introduction*, 2nd ed. (2018)  
> Available free at [incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)

---

## Table of Contents

1. [Foundations: MDPs and Value Functions](#1-foundations-mdps-and-value-functions)
2. [Dynamic Programming](#2-dynamic-programming)
3. [Monte Carlo Methods](#3-monte-carlo-methods)
4. [Temporal-Difference Learning](#4-temporal-difference-learning)
5. [Planning: Dyna-Q](#5-planning-dyna-q)
6. [Function Approximation](#6-function-approximation)
7. [Learning-Based Control](#7-learning-based-control)

---

## 1. MDPs and Value Functions

These notebooks introduce the core mathematical framework: Markov Decision Processes, value functions, and the Bellman equations.

### [tic-tac-toe.ipynb](1.%20MDP%20and%20Value%20Functions/tic-tac-toe.ipynb)
A self-play tic-tac-toe agent learned via temporal-difference updates on win/loss outcomes. Demonstrates how RL can learn a strategy purely through trial and error without a hand-crafted evaluation function.

- **S&B reference:** Chapter 1.5 (introductory example)

---

### [markov_process.ipynb](1.%20MDP%20and%20Value%20Functions//markov_process.ipynb)
Defines an MDP by its transition matrix and simulates state trajectories. Illustrates how the Markov property enables compact specification of an environment.

- **S&B reference:** Chapter 3 — Finite MDPs

---

### [recycling_bot_value_function.ipynb](1.%20MDP%20and%20Value%20Functions/)/recycling_bot_value_function.ipynb)
Computes and visualises state value functions for the Recycling Robot MDP under different fixed policies. Shows how the value of a state depends on the policy and the discount factor γ.

- **S&B reference:** Example 3.3 (Recycling Robot), Section 3.5 (Policies and Value Functions)

---

### [bellman_opt_eq.ipynb](1.%20MDP%20and%20Value%20Functions//bellman_opt_eq.ipynb)
Solves the **Bellman optimality equations** directly (using a nonlinear root-finder) for two environments:
- **Recycling Robot** — derives the optimal value function and reads off the optimal policy.
- **4×4 Gridworld** — solves for V\* and compares against the policy iteration result from `dp_gridworld.ipynb`.

- **S&B reference:** Example 3.3 (Recycling Robot), Example 3.8/3.9 / Section 3.6 (Bellman Optimality), Example 4.1 (Gridworld)

---

## 2. Dynamic Programming

DP algorithms assume a perfect model of the environment and solve the MDP exactly.

### [dp_gridworld.ipynb](2.%20Dynamic%20Programming/dp_gridworld.ipynb)
Step-by-step demonstration of the three core DP algorithms on the classic 4×4 gridworld:
- **Policy Evaluation** — iterative computation of Vπ for the equiprobable random policy.
- **Policy Improvement** — greedy update from a given value function.
- **Policy Iteration** — alternating evaluation and improvement until convergence.

- **S&B reference:** Example 4.1 (Gridworld), Sections 4.1–4.3 (Policy Evaluation, Improvement, Iteration)

---

### [dp_gridworld2.ipynb](2.%20Dynamic%20Programming/dp_gridworld2.ipynb)
Extends `dp_gridworld.ipynb` with two additional algorithms and an important implementation variant:
- **Policy Iteration with in-place updates** — updates the value array in-place during a sweep (often converges faster).
- **Value Iteration** — collapses evaluation and improvement into a single max-backup sweep.
- **Value Iteration with in-place updates**

- **S&B reference:** Example 4.1, Section 4.3 (Policy Iteration), Section 4.4 (Value Iteration)

---

## 3. Monte Carlo Methods

MC methods learn directly from complete episodes without a model.

### [BlackJack.ipynb](3.%20Monte%20Carlo%20Methods/BlackJack.ipynb)
Trains a Blackjack agent with **Monte Carlo Control with Exploring Starts**. The learned policy and action-value function are visualised as surface plots over (player sum, dealer card) state space.

- **S&B reference:** Example 5.1 and Example 5.3 (Blackjack), Section 5.3 (MC Control with Exploring Starts)

---

## 4. Temporal-Difference Learning

TD methods combine the sampling of MC with the bootstrapping of DP to learn online from incomplete episodes.

### [td0_vs_constant_alpha_mc.ipynb](4.%20Temporal%20Difference%20Learning/td0_vs_constant_alpha_mc.ipynb)
Side-by-side comparison of **TD(0)** and **constant-α MC** for policy *evaluation* (prediction) on the FrozenLake environment. Tracks the deviation from the true value function over training episodes to illustrate the bias-variance trade-off between the two methods.

- **S&B reference:** Section 6.1 (TD Prediction), Example 6.2 (Random Walk comparison)

---

### [cliffwalking_on_vs_off-policy.ipynb](4.%20Temporal%20Difference%20Learning/cliffwalking_on_vs_off-policy.ipynb)
Runs **SARSA** (on-policy) and **Q-Learning** (off-policy) on the Cliff Walking grid. Plots the sum of rewards per episode to illustrate how on-policy control learns a safe path while off-policy control learns the optimal but risky cliff-edge path.

- **S&B reference:** Example 6.6 (Cliff Walking), Section 6.4 (SARSA), Section 6.5 (Q-Learning)

---

### [td_vs_mc_control.ipynb](4.%20Temporal%20Difference%20Learning/td_vs_mc_control.ipynb)
Compares **SARSA** (TD control) and **First-Visit MC Control** on FrozenLake. Visualises learning curves and final policies side by side to highlight the practical convergence speed difference.

- **S&B reference:** Chapter 5.3 (MC Control), Chapter 6.4 (SARSA/TD Control)

---

## 5. Planning: Dyna-Q

Planning integrates a learned environment model with direct RL to improve sample efficiency.

### [Dyna-Q_vs_Q-Learning.ipynb](5.%20Dyna-Q/Dyna-Q_vs_Q-Learning.ipynb)
Compares **Dyna-Q** against plain **Q-Learning** on a maze environment. Demonstrates that by re-using simulated experience from an internal model, Dyna-Q can reach the same policy with significantly fewer real environment interactions.

- **S&B reference:** Chapter 8, Section 8.2 (Dyna: Integrated Planning, Acting, and Learning)

---

## 6. Function Approximation

Extends tabular RL to large or continuous state spaces by parameterising the value function.

### [random_walk.ipynb](6.%20Function%20Approximation/random_walk.ipynb)
Applies **linear value function approximation** (state aggregation and polynomial/Fourier bases) to the 1000-state Random Walk benchmark. Compares approximated value functions against the true values to illustrate the effect of feature design.

- **S&B reference:** Example 9.1 (1000-state Random Walk), Chapter 9 (On-policy Prediction with Approximation)

---

### [linear_sarsa.ipynb](6.%20Function%20Approximation/linear_sarsa.ipynb)
Solves the continuous **Mountain Car** problem with **Semi-gradient SARSA** using a linear function approximator built on tile coding (radial basis features). Demonstrates on-policy control with function approximation.

- **S&B reference:** Section 9.5 (Feature Construction — Tile Coding), Section 10.1 (Episodic Semi-gradient Control), Example 10.1 (Mountain Car)

---

### [nonlinear_approximation.ipynb](6.%20Function%20Approximation/nonlinear_approximation.ipynb)
Uses a **feedforward neural network** as a nonlinear function approximator to balance the CartPole. The network is updated via semi-gradient TD, bridging the gap between classical function approximation and deep RL.

- **S&B reference:** Section 9.7 (Nonlinear Function Approximation — ANNs)

---

## 7. Learning-Based Control

These notebooks go beyond standard RL and apply system identification and model-based control to physical systems.

### [lin_sys_id_oscillator.ipynb](7.%20Learning-Based%20Control/lin_sys_id_oscillator.ipynb)
Identifies a **linear state-space model** of a damped spring-mass oscillator from noisy data under varying noise conditions. Covers least-squares system identification and analyses how noise level affects model accuracy.

- **Course context:** Learning-Based Control — System Identification

---

### [Cartpole_lqr.ipynb](7.%20Learning-Based%20Control/Cartpole_lqr.ipynb)
Derives and applies a **Linear Quadratic Regulator (LQR)** to balance the CartPole. Linearises the nonlinear cartpole dynamics around the upright equilibrium, solves the discrete algebraic Riccati equation (DARE) for the optimal gain matrix, and simulates the closed-loop response.

- **Course context:** Learning-Based Control — LQR / Optimal Control

---

### [Cartpole_NARX.ipynb](7.%20Learning-Based%20Control/Cartpole_NARX.ipynb)
Trains a **Nonlinear AutoRegressive eXogenous (NARX)** model of the CartPole environment using a feedforward neural network. The learned model can be used as a differentiable simulator for model-based policy optimisation.

- **Course context:** Learning-Based Control — Nonlinear System Identification

