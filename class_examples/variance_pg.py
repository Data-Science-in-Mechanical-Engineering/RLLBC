import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Softmax policy with single parameter theta
def policy(theta):
    prob_0 = 1 / (1 + np.exp(-theta))
    return np.array([prob_0, 1 - prob_0])

# Sample action based on policy
def sample_action(theta):
    return np.random.choice([0, 1], p=policy(theta))

# Compute gradient of log-policy
def grad_log_policy(theta, action):
    prob_0 = policy(theta)[0]

    if action == 0:
        return 1 - prob_0
    else:
        return -prob_0

# Mean rewards for actions
reward_means = np.array([1.0, 0.1])

# Reward noise: this makes the task stochastic
reward_stds = np.array([0.05, 0.05])

def sample_reward(action):
    return reward_means[action] + np.random.normal(0.0, reward_stds[action])

def expected_reward(theta):
    return np.dot(policy(theta), reward_means)

def true_policy_gradient(theta):
    p = policy(theta)[0]
    q = 1 - p
    return p * q * (reward_means[0] - reward_means[1])

def value_baseline(theta):
    return expected_reward(theta)

def optimal_variance_baseline(theta):
    p = policy(theta)[0]
    q = 1 - p

    # For this scalar two-action policy:
    # b* = E[R * score^2] / E[score^2]
    # Since reward noise has zero mean, only reward_means enter.
    return q * reward_means[0] + p * reward_means[1]


# Parameters
theta = 0.0
learning_rate = 0.05
num_updates = 200
episodes_per_update = 20

# Choose which estimator is used to update theta
# Options: "no_baseline", "value_baseline", "optimal_baseline"
update_with = "optimal_baseline"

# Storage
theta_history = []

grad_no_baseline_history = []
grad_value_baseline_history = []
grad_optimal_baseline_history = []
true_grad_history = []

# Training loop
for update in range(num_updates):

    grad_no_baseline_batch = []
    grad_value_baseline_batch = []
    grad_optimal_baseline_batch = []

    b_value = value_baseline(theta)
    b_optimal = optimal_variance_baseline(theta)

    # Accumulate gradients over several episodes
    for episode in range(episodes_per_update):
        action = sample_action(theta)
        reward = sample_reward(action)

        score = grad_log_policy(theta, action)

        grad_no_baseline_batch.append(score * reward)
        grad_value_baseline_batch.append(score * (reward - b_value))
        grad_optimal_baseline_batch.append(score * (reward - b_optimal))

    # Average gradients over accumulated episodes
    grad_no_baseline = np.mean(grad_no_baseline_batch)
    grad_value_baseline = np.mean(grad_value_baseline_batch)
    grad_optimal_baseline = np.mean(grad_optimal_baseline_batch)

    # Store values before update
    theta_history.append(theta)

    grad_no_baseline_history.append(grad_no_baseline)
    grad_value_baseline_history.append(grad_value_baseline)
    grad_optimal_baseline_history.append(grad_optimal_baseline)
    true_grad_history.append(true_policy_gradient(theta))

    # Pick gradient estimator for the theta update
    if update_with == "no_baseline":
        grad_for_update = grad_no_baseline
    elif update_with == "value_baseline":
        grad_for_update = grad_value_baseline
    elif update_with == "optimal_baseline":
        grad_for_update = grad_optimal_baseline
    else:
        raise ValueError(f"Unknown update_with option: {update_with}")

    # Gradient ascent update
    theta += learning_rate * grad_for_update


# Convert to arrays
theta_history = np.array(theta_history)

grad_no_baseline_history = np.array(grad_no_baseline_history)
grad_value_baseline_history = np.array(grad_value_baseline_history)
grad_optimal_baseline_history = np.array(grad_optimal_baseline_history)
true_grad_history = np.array(true_grad_history)


# Plot gradients over time
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(grad_no_baseline_history, alpha=0.7, label="Batch gradient without baseline")
plt.plot(grad_value_baseline_history, alpha=0.7, label="Batch gradient with value baseline")
plt.plot(grad_optimal_baseline_history, alpha=0.9, label="Batch gradient with optimal baseline")
plt.plot(true_grad_history, linewidth=2, label="True expected policy gradient")
plt.axhline(0, color="black", linewidth=1)
plt.title("Policy Gradient Estimates Over Updates")
plt.xlabel("Update step")
plt.ylabel("Gradient")
plt.legend()
plt.grid(True)
plt.show()


# Plot theta over time
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(theta_history)
plt.title(f"Theta Over Time, updated with: {update_with}")
plt.xlabel("Update step")
plt.ylabel(r"$\theta$")
plt.grid(True)
plt.show()



# ------------------------------------------------------------
# One-sample gradients for theta = 0
# With noise and without noise
# Only: no baseline vs value baseline
# ------------------------------------------------------------

theta_fixed = 0.0
n_single_sample_steps = 10

x = np.arange(n_single_sample_steps)

b_value_fixed = value_baseline(theta_fixed)
true_grad_fixed = true_policy_gradient(theta_fixed)

# Use the same sampled actions for both plots
actions = [sample_action(theta_fixed) for _ in range(n_single_sample_steps)]


# ============================================================
# Plot 1: one-sample gradients WITH reward noise
# ============================================================

single_grad_no_baseline_noise = []
single_grad_value_baseline_noise = []

for action in actions:
    reward = sample_reward(action)   # stochastic reward

    score = grad_log_policy(theta_fixed, action)

    single_grad_no_baseline_noise.append(score * reward)
    single_grad_value_baseline_noise.append(score * (reward - b_value_fixed))

single_grad_no_baseline_noise = np.array(single_grad_no_baseline_noise)
single_grad_value_baseline_noise = np.array(single_grad_value_baseline_noise)

plt.figure(figsize=(10, 5), dpi=300)

plt.stem(
    x - 0.1,
    single_grad_no_baseline_noise,
    linefmt="C0-",
    markerfmt="C0o",
    basefmt="k-",
    label="No baseline",
    )

plt.stem(
    x + 0.1,
    single_grad_value_baseline_noise,
    linefmt="C1-",
    markerfmt="C1o",
    basefmt="k-",
    label="Value baseline",
    )

plt.axhline(true_grad_fixed, linestyle="--", linewidth=2, label="True expected gradient")
plt.axhline(0, color="black", linewidth=1)

plt.title(r"One-Sample Policy Gradient Estimates at $\theta = 0$ With Reward Noise")
plt.xlabel("Sample step")
plt.ylabel("One-sample gradient estimate")
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# Plot 2: one-sample gradients WITHOUT reward noise
# ============================================================

single_grad_no_baseline_no_noise = []
single_grad_value_baseline_no_noise = []

for action in actions:
    reward = reward_means[action]    # deterministic reward, no noise

    score = grad_log_policy(theta_fixed, action)

    single_grad_no_baseline_no_noise.append(score * reward)
    single_grad_value_baseline_no_noise.append(score * (reward - b_value_fixed))

single_grad_no_baseline_no_noise = np.array(single_grad_no_baseline_no_noise)
single_grad_value_baseline_no_noise = np.array(single_grad_value_baseline_no_noise)

plt.figure(figsize=(10, 5), dpi=300)

plt.stem(
    x - 0.1,
    single_grad_no_baseline_no_noise,
    linefmt="C0-",
    markerfmt="C0o",
    basefmt="k-",
    label="No baseline",
    )

plt.stem(
    x + 0.1,
    single_grad_value_baseline_no_noise,
    linefmt="C1-",
    markerfmt="C1o",
    basefmt="k-",
    label="Value baseline",
    )

plt.axhline(true_grad_fixed, linestyle="--", linewidth=2, label="True expected gradient")
plt.axhline(0, color="black", linewidth=1)

plt.title(r"One-Sample Policy Gradient Estimates at $\theta = 0$ Without Reward Noise")
plt.xlabel("Sample step")
plt.ylabel("One-sample gradient estimate")
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()