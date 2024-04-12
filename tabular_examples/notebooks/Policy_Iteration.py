import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import custom_envs
import random

class MarkovDecisionProcess():
    def __init__(self, env):
        self.num_states = env.observation_space.n
        self.P = env.unwrapped.P
        self.actions_per_state = [] # list containing the actions available per state
        for state in self.P:
            actions = list(self.P[state].keys())
            self.actions_per_state.append(actions)

class Agent():
    def __init__(self, mdp, gamma=0.9, update_threshold=1e-6):
        self.mdp = mdp # contains state transition function, num_states and actions
        self.update_threshold = update_threshold # stopping distance as criteria for stopping policy evaluation
        self.state_value_fn = np.zeros(self.mdp.num_states) # a table leading from state to value expectations
        # Create random policy
        self.policy = []
        for state in range(self.mdp.num_states):
            random_entry = random.randint(0, len(self.mdp.actions_per_state[state])-1)
            self.policy.append([0 for _ in range(len(self.mdp.actions_per_state[state]))])
            self.policy[state][random_entry] = 1
        self.gamma = gamma # discount rate for return

    def get_greedy_action(self, state):
        # Choose action based on the probabilities defined within the policy
        action = np.random.choice(np.flatnonzero(np.isclose(self.policy[state], max(self.policy[state]), rtol=0.01)))
        return action

    def train(self, in_place=False):
        policy_stable = False
        iteration = 0
        total_sweeps = 0
        while not policy_stable:
            iteration += 1
            # Policy Evaluation
            total_sweeps += self.policy_evaluation()
            # Policy Improvement
            policy_stable = self.policy_improvement()
        print('Sweeps required for convergence ', str(total_sweeps))
        print('Iterations required for convergence ', str(iteration))

    def policy_evaluation(self, in_place=True): # in_place version
        sweeps = 0
        stable = False
        while not stable:
            delta = 0
            sweeps += 1
            if in_place:
                for state in range(self.mdp.num_states):
                    old_state_value = self.state_value_fn[state]
                    new_state_value = 0
                    # sum over potential actions
                    for action in range(len(self.mdp.actions_per_state[state])):
                        new_state_value += self.get_state_value(state, action)
                    self.state_value_fn[state] = new_state_value
                    delta = max(delta, np.abs(old_state_value - self.state_value_fn[state]))
                if delta < self.update_threshold:
                    stable = True
            else:
                new_state_value_fn = np.copy(self.state_value_fn)
                for state in range(self.mdp.num_states):
                    old_state_value = self.state_value_fn[state]
                    new_state_value = 0
                    # sum over potential actions
                    for action in range(len(self.mdp.actions_per_state[state])):
                        new_state_value += self.get_state_value(state, action)
                    new_state_value_fn[state] = new_state_value
                    delta = max(delta, np.abs(old_state_value - self.state_value_fn[state]))
                if delta < self.update_threshold:
                    stable = True
                self.state_value_fn = new_state_value_fn
        return sweeps

    def get_state_value(self, state, action):
        # Value expectation considering the policy
        policy_value = 0
        for transition in self.mdp.P[state][action]:
            transition_prob = transition[0]  # prob of next state
            successor_state = transition[1]  # value/name of next state
            reward = transition[2]  # reward of next state
            policy_value += self.policy[state][action] * transition_prob * (reward + self.gamma * self.state_value_fn[successor_state])
        return policy_value
    
    def get_action_value(self, state, action):
        # Value expectation without considering the policy
        action_value = 0
        for transition in self.mdp.P[state][action]:
            transition_prob = transition[0]  # prob of next state
            successor_state = transition[1]  # value/name of next state
            reward = transition[2]  # reward of next state
            action_value += transition_prob * (reward + self.gamma * self.state_value_fn[successor_state])
        return action_value

    def policy_improvement(self):
        policy_stable = True
        current_policy = self.policy # Cache current policy
        best_policy = []
        for state in range(self.mdp.num_states):
            best_policy.append([0 for _ in range(len(self.mdp.actions_per_state[state]))])
            # Calculate best possible policy based on current value function
            action_values = []
            for action in range(len(self.mdp.actions_per_state[state])):
                action_values.append(self.get_action_value(state, action))
            best_actions = np.where(action_values == max(action_values))[0]
            for index in best_actions:
                best_policy[state][index] = 1
            best_policy[state] = [best_policy[state][action] / len(best_actions)
                                  for action in range(len(self.mdp.actions_per_state[state]))]
            # If the current policy is not the best policy, update it
            if not np.array_equal(current_policy[state], best_policy[state]):
                policy_stable = False
                self.policy[state] = best_policy[state]
        self.visualize()
        return policy_stable

    # for frozen_lake:
    def visualize(self):
        print('Value function:\n', np.asmatrix(self.state_value_fn))
        x_axis = 4
        y_axis = 4
        X1 = np.reshape(self.state_value_fn, (x_axis, y_axis))
        fig, ax = plt.subplots(1, 1)
        cmap = cm.get_cmap("Blues_r")
        cmap.set_under("black")
        img = ax.imshow(X1, interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
        ax.axis('off')
        for i in range(x_axis):
            for j in range(y_axis):
                ax.text(j, i, str(X1[i][j])[:4], fontsize=12, color='black', ha='center', va='center')
        plt.show()
        self.render_policy()

    def render_policy(self):
        print('Policy of the agent:')
        print('-----------------------------')
        out = '| '
        for i in range(self.mdp.num_states):
            token = ""
            if self.policy[i][3] > 0:   # left
                token += "\u2190"
            if self.policy[i][0] > 0:   # up
                token += "\u2191"
            if self.policy[i][1] > 0:   # down
                token += "\u2193"
            if self.policy[i][2] > 0:   # right
                token += "\u2192"
            if token == "":     # empty
                token += "  "
            if len(token) == 1:
                token += '  '
                token = ' ' + token
            elif len(token) == 2:
                token += ' '
                token = ' ' + token
            elif len(token) == 3:
                token += ' '
            out += token + ' | '
            if (i + 1) % 4 == 0:
                print(out)
                print('-----------------------------')
                out = '| '
    

    """ # for basketball: 
    def visualize(self):
        print('Value function:\n', np.asmatrix(self.state_value_fn))
        x_axis = 1
        y_axis = self.mdp.num_states - 2
        vmin = min(self.state_value_fn)
        vmax = max(self.state_value_fn)
        X1 = np.reshape(self.state_value_fn[:-2], (x_axis, y_axis))
        fig, ax = plt.subplots(1, 1)
        cmap = plt.colormaps["Blues_r"]
        cmap.set_under("black")
        img = ax.imshow(X1, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis('off')
        ax.set_title("Values of the state value function on the field")
        for i in range(x_axis):
            for j in range(y_axis):
                ax.text(j, i, str(X1[i][j])[:4], fontsize=12, color='black', ha='center', va='center')
        plt.show()
        self.render_policy()

    def render_policy(self):
        print('Policy of the agent:')
        out = ' | '
        render = out
        for i in range(self.mdp.num_states-2):
            token = ""
            if self.policy[i][0] > 0:   # move
                token += "Move"
            if self.policy[i][1] > 0:   # up
                token += "Throw"
            if len(token) > 5:
                token = 'Move or Throw'
            render += token + out
        print(render) """

    """ for recycling_bot: 
    def visualize(self):
        print('Value function:\n', np.asmatrix(self.state_value_fn))
        x_axis = 1
        y_axis = 2
        X1 = np.reshape(self.state_value_fn, (x_axis, y_axis))
        fig, ax = plt.subplots(1, 1)
        cmap = cm.get_cmap("Blues_r")
        cmap.set_under("black")
        img = ax.imshow(X1, interpolation="nearest", vmin=0, vmax=max(self.state_value_fn), cmap=cmap)
        ax.axis('off')
        for i in range(x_axis):
            for j in range(y_axis):
                ax.text(j, i, str(X1[i][j])[:4], fontsize=12, color='black', ha='center', va='center')
        plt.show()
        self.render_policy()

    def render_policy(self):
        print('Policy of the agent:')
        out = ' | '
        render = out
        for i in range(self.mdp.num_states):
            token = ""
            if self.policy[i][0] > 0:   # search
                token += "Search"
            if self.policy[i][1] > 0:   # wait
                if token != "":
                    token += " or Wait"
                else:
                    token += "Wait"
            if len(self.policy[i]) > 2:
                if self.policy[i][2] > 0:   # recharge
                    if token != "":
                        token += " and Recharge"
                    else:
                        token += "Recharge"
            render += token + out
        print(render) """
    
    def evaluate(self, env, num_runs=1):
        tot_reward = [0]
        for _ in range(num_runs):
            done = False
            obs, info = env.reset()
            reward_per_run = 0
            while not done:
                action = self.get_greedy_action(obs)
                obs, reward, done, _ , info = env.step(action)
                reward_per_run += reward
            tot_reward.append(reward_per_run + tot_reward[-1])
        return tot_reward
    
        
########################################################################################################################

if __name__ == "__main__":

    # for frozen_lake: 
    # Evaluation of the random policy.
    env = gym.make('CustomFrozenLake-v1', render_mode='human', is_slippery=False)
    env.reset()
    mdp = MarkovDecisionProcess(env) # in our case contains dynamics function
    agent = Agent(mdp, gamma=0.9, update_threshold=0.0001)

    # Train the agent.
    input("Press Enter to train the agent")
    agent.train()

    # Evaluation of the new policy.
    print('Sampling from final policy...')
    num_runs=4
    agent.evaluate(env, num_runs=num_runs)

    
    """ for basketball: 
    env = gym.make('CustomBasketBall-v1', render_mode=None, field_length=6, line_position = 2)
    env.reset()
    mdp = MarkovDecisionProcess(env) # in our case contains dynamics function
    agent = Agent(mdp, gamma=1.0)

    print(env.P)

    # Train the agent
    input("Press Enter to start")
    num_runs=10000
    random_results = agent.evaluate(env, num_runs=num_runs)
    input("Press Enter to train the agent")
    agent.train()

    # Evaluate the agent
    input("Press Enter to evaluate the agent")
    env = gym.make('CustomBasketBall-v1', render_mode='human', field_length=12, line_position = 6)
    trained_results = agent.evaluate(env, num_runs=num_runs)

    # Comparing both agents
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
    ax.plot(range(num_runs + 1), random_results, label="Random Agent")
    ax.plot(range(num_runs + 1), trained_results, label="Trained Agent")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Cumulative Reward')
    ax.legend()
    plt.show() """
    
    

    """ for recycling_bot:
    env = gym.make('RecyclingRobot-v1', render_mode='text')
    env.reset()
    mdp = MarkovDecisionProcess(env) # in our case contains dynamics function
    agent = Agent(mdp, gamma=0.9, update_threshold=0.05)

    # Train the agent.
    input("Press Enter to train the agent")
    agent.train()

    # Evaluation of the new policy.
    print('Sampling from final policy...')
    num_runs=4
    agent.evaluate(env, num_runs=num_runs)
    """

