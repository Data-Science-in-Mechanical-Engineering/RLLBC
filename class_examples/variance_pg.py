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
    prob = policy(theta)[0]
    if action == 0:
        return 1 - prob
    else:
        return -prob

# Rewards for actions
rewards = [1, 0.1]

# Parameters
theta = 0.0  # Initial parameter
num_episodes = 500
num_runs = 200

# Store gradients
grads_no_baseline = []
grads_with_baseline = []

# Perform multiple runs
for run in range(num_runs):
    grad_no_baseline = []
    grad_with_baseline = []

    # Run episodes
    for episode in range(num_episodes):
        action = sample_action(theta)
        reward = rewards[action]

        grad = grad_log_policy(theta, action)
        grad_no_baseline.append(grad * reward)

        # Using mean reward as baseline (here, baseline=0.5 since expected reward is 0.5)
        baseline = 0.5
        grad_with_baseline.append(grad * (reward - baseline))

    grads_no_baseline.append(np.mean(grad_no_baseline))
    grads_with_baseline.append(np.mean(grad_with_baseline))

# Compute variance
var_no_baseline = np.var(grads_no_baseline)
var_with_baseline = np.var(grads_with_baseline)

# Print variances
print(f"Variance without baseline: {var_no_baseline:.6f}")
print(f"Variance with baseline:    {var_with_baseline:.6f}")

# Plot the first n grads with and without baseline as diarcs
n_grads_plot = 10
plt.figure()
plt.stem(range(n_grads_plot), grad_no_baseline[:n_grads_plot], linefmt='C0-', markerfmt='C0o', basefmt='k-', label="no_baseline")
plt.stem(range(n_grads_plot), grad_with_baseline[:n_grads_plot], linefmt='C1-', markerfmt='C1o', basefmt='k-', label="with_baseline")
plt.title('Policy Gradients with and without Baseline')
plt.xlabel('Step')
plt.ylabel('Gradient Estimate')
plt.legend()
plt.show()


# Visualize the results
plt.figure(figsize=(10,5))
plt.hist(grads_no_baseline, alpha=0.6, label='No Baseline', density=True)
plt.hist(grads_with_baseline, alpha=0.6, label='With Baseline', density=True)
plt.title('Distribution of Policy Gradients')
plt.xlabel('Gradient Estimate')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
