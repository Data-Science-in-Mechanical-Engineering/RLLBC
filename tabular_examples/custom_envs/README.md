# Custom Environments

This package provides custom Gymnasium environments used throughout the RLLBC course notebooks. All environments follow the standard [Gymnasium API](https://gymnasium.farama.org/) and are registered on import via `import custom_envs`.

---

## Registered Environments

### `CustomFrozenLake-v1`

**File:** `envs/frozen_lake.py` | **Used in:** `3. Monte Carlo Methods`, `4. Temporal Difference Learning`, `5. Dyna-Q`

A grid-world navigation task adapted from Gymnasium's `FrozenLake-v1`. The agent crosses a frozen lake from start `S` to goal `G` without falling into holes `H`. Accepts a custom map layout and supports deterministic (`is_slippery=False`) movement.

---

### `RecyclingRobot-v1`

**File:** `envs/recycling_robot.py` | **Used in:** `1. MDP and Value Functions`

The recycling robot MDP from Sutton & Barto (Example 3.3). A robot with a two-level battery chooses to search, wait, or recharge, trading off reward against battery depletion risk. Transition probabilities and rewards are configurable.

---

### `BlackJack-v1`

**File:** `envs/black_jack.py` | **Used in:** `3. Monte Carlo Methods`

A Blackjack environment following the simplified rules from Sutton & Barto (Example 5.1). The agent plays against a fixed dealer strategy with observations of player sum, dealer card, and usable ace.

---

### `CliffWalking-v1`

**File:** `envs/cliff_walking.py` | **Used in:** `4. Temporal Difference Learning`

The cliff-walking grid from Sutton & Barto (Example 6.6). The agent navigates a 4×12 grid from start to goal, receiving a large negative reward for stepping off the cliff edge and returning to start.

---

### `CustomCartPole-v1`

**File:** `envs/cart_pole.py` | **Used in:** `7. Learning-Based Control`

A cartpole with a continuous force action and full nonlinear equations of motion following [Florian (2007)](https://coneural.org/florian/papers/05_cart_pole.pdf). Designed for LQR synthesis and NARX model learning. Exposes system matrices for use with the [python-control](https://python-control.readthedocs.io/) library.

---

### `CustomPendulum-v1`

**File:** `envs/pendulum.py` | **Used in:** `6. Function Approximation`

A modified version of Gymnasium's `Pendulum-v1` with a simplified observation space (`[θ, θ̇]` directly instead of trigonometric encoding) and a reward function designed for value function approximation experiments with a PD controller baseline.

---

## Adding a New Environment

1. Create a new file in `envs/` implementing a `gymnasium.Env` subclass.
2. Register it in `custom_envs/__init__.py`:

```python
register(
    id='MyEnv-v1',
    entry_point='custom_envs.envs.my_env:MyEnvClass',
)
```