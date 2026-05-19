# Learning-Based Control — Example Notebooks

This folder contains notebook implementations of learning-based control methods covered in the RLLBC course. All notebooks use the CartPole as a common benchmark system and progress from classical optimal control through data-driven dynamics learning to model-based and data-efficient control.

Students are encouraged to modify cost weights, prediction horizons, and noise levels to deepen their understanding.

---

## Table of Contents

1. [LQR — Optimal Linear Control](#1-lqr--optimal-linear-control)
2. [Dynamics Learning](#2-dynamics-learning)
3. [Model Predictive Control](#3-model-predictive-control)
4. [Bayesian Optimization for Controller Tuning](#4-bayesian-optimization-for-controller-tuning)

---

## 1. LQR — Optimal Linear Control

### [lqr.ipynb](lqr.ipynb)

Designs a **Linear Quadratic Regulator (LQR)** to stabilise the CartPole at its upper equilibrium. Linearises the nonlinear dynamics, solves the discrete algebraic Riccati equation (DARE) for the optimal gain matrix K, and simulates the closed-loop response.

- **Course context:** Learning-Based Control — Optimal Control / LQR

---

## 2. Dynamics Learning

### [dynamics_learning_discrete.ipynb](dynamics_learning_discrete.ipynb)

Learns a **discrete-time dynamics model** of the CartPole from data using a feedforward neural network trained in PyTorch. Covers data generation, model training, and evaluation of prediction accuracy — forming the basis for model-based control approaches.

- **Course context:** Learning-Based Control — System Identification / Dynamics Learning

---

## 3. Model Predictive Control

### [mpc.ipynb](mpc.ipynb)

Implements a **Model Predictive Control (MPC)** scheme for the CartPole swing-up task. At each step, MPC optimises a finite-horizon trajectory using the system model and applies only the first control input, enabling complex non-equilibrium manoeuvres beyond the reach of LQR.

- **Course context:** Learning-Based Control — Model Predictive Control

---

## 4. Bayesian Optimization for Controller Tuning

### [bo.ipynb](bo.ipynb)

Uses **Bayesian Optimization (BO)** to automatically tune the cost matrices of an LQR controller. A Gaussian process surrogate models the mapping from cost weights to closed-loop performance, allowing data-efficient tuning without manual grid search.

- **Course context:** Learning-Based Control — Data-Efficient Optimisation / Bayesian Optimisation

---