import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import custom_envs

class TD_Agent():
    def __init__(self, env, gamma=1.0, learning_rate=0.05, epsilon=0.1, dyn_q_iters=100):
        self.env = env
        self.action_value_fn = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # the q-fn
        self.model = np.zeros((self.env.observation_space.n, self.env.action_space.n, 2))
        class StateActionTracker:
            def __init__(self):
                self.states = {}
                self.actions = {}
            def add_state_action_pair(self, state, action):
                if state not in self.states:
                    self.states[state] = 1
                    self.actions[state] = set([action])
                else:
                    self.states[state] += 1
                    if action not in self.actions[state]:
                        self.actions[state].add(action)
            def get_state(self):
                if self.states:
                    return np.random.choice(list(self.states.keys()))
            def get_action(self, state):
                if state in self.actions and self.actions[state]:
                    return np.random.choice(list(self.actions[state]))
        self.visited_states_and_actions = StateActionTracker()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon # choose 0.0 to make a totally greedy policy, 1.0 for a fully random policy
        self.dyn_q_iters = dyn_q_iters

    def get_random_action(self):
        random_action = np.random.choice(range(self.env.action_space.n))
        return random_action
    def get_best_action(self, state):
        best_action = np.random.choice(np.flatnonzero(np.isclose(self.action_value_fn[state], self.action_value_fn[state].max(),
                                                                 rtol=0.01)))
        return best_action
    def epsilon_greedy_policy(self, state):
        # returns action, choosing a random action with probability epsilon, or the best action
        # regarding Q with probability (1 - epsilon)
        randomly = np.random.random() < self.epsilon
        if randomly:
            action = self.get_random_action()
        else:
            action = self.get_best_action(state)
        return action

    def train(self, num_episodes):
        # Reset environment and pick first action
        for i in range(num_episodes+1):
            state, info = env.reset()  # most values are replaced in first env.step(),
            done = False
            while not done:
                # Choose action and perform step
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, truncated, info = env.step(action)
                # TD Update
                best_next_action = self.get_best_action(next_state)
                self.action_value_fn[state][action] = self.action_value_fn[state][action] + self.learning_rate * (reward + self.gamma * self.action_value_fn[next_state][best_next_action] - self.action_value_fn[state][action])
                self.model[state][action] = np.array([next_state, reward]) # Assuming deterministic environment
                self.visited_states_and_actions.add_state_action_pair(state, action)
                state = next_state
                for j in range(self.dyn_q_iters):
                    prev_state = self.visited_states_and_actions.get_state()
                    prev_action = self.visited_states_and_actions.get_action(prev_state)
                    prev_next_state, prev_reward = int(self.model[prev_state][prev_action][0]), self.model[prev_state][prev_action][1]
                    best_next_action = self.get_best_action(prev_next_state)
                    self.action_value_fn[prev_state][prev_action] = self.action_value_fn[prev_state][prev_action] + self.learning_rate * (prev_reward + self.gamma * self.action_value_fn[prev_next_state][best_next_action] - self.action_value_fn[prev_state][prev_action])
            if i % 20 == 0:
                self.visualize(i)
    
    def evaluate(self, env, num_runs=1):
        tot_reward = [0]
        for _ in range(num_runs):
            done = False
            state, info = env.reset()
            reward_per_run = 0
            while not done:
                action = self.get_best_action(state)
                state, reward, done, truncated, info = env.step(action)
                reward_per_run += reward
            tot_reward.append(reward_per_run + tot_reward[-1])
        return tot_reward
    
    # for cliff_walking:
    def visualize(self, epoch):
        self.plot_action_value(epoch)

    def plot_action_value(self, epoch): 
        q_fn = np.around(self.action_value_fn, decimals=1)
        fig, ax=plt.subplots(figsize=(18,5))
        lines = 4
        rows = 12
        # Define grid positions:
        pos_x_left = 0.2
        pos_x_mid = 0.5
        pos_x_right = 0.8
        pos_y_up = 0.2
        pos_y_mid = 0.5
        pos_y_down = 0.8
        grid_size = {'x': lines, 'y': rows}
        def gridcreator(pos_x, pos_y):
            grid = []
            for i in range(grid_size['x']):
                for j in range(grid_size['y']):
                    x = pos_x + j
                    y = pos_y + i
                    grid.append((x, y))
            return grid
        top = q_fn[:,0].reshape((lines,rows))
        top_value_positions = gridcreator(pos_x_mid, pos_y_up)
        right = q_fn[:,1].reshape((lines,rows))
        right_value_positions = gridcreator(pos_x_right, pos_y_mid)
        bottom = q_fn[:,2].reshape((lines,rows))
        bottom_value_positions = gridcreator(pos_x_mid, pos_y_down)
        left= q_fn[:,3].reshape((lines,rows))
        left_value_positions = gridcreator(pos_x_left, pos_y_mid)
        # Define triangles
        ax.set_ylim(lines, 0)
        anchor_points = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]]) # Corner coordinates
        corner_indizes = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]]) # Corner indices
        xy_coordinates = np.zeros((lines * rows * 5,2))
        triangles = np.zeros((lines * rows * 4, 3))
        for i in range(lines):
            for j in range(rows):
                k = i*rows+j
                xy_coordinates[k*5:(k+1)*5,:] = np.c_[anchor_points[:,0]+j, 
                                                      anchor_points[:,1]+i]
                triangles[k*4:(k+1)*4,:] = corner_indizes + k*5
        colours = np.c_[left.flatten(), top.flatten(), 
                right.flatten(), bottom.flatten()].flatten()
        ax.triplot(xy_coordinates[:,0], xy_coordinates[:,1], triangles, 
                   **{"color":"k", "lw":1})
        tripcolor = ax.tripcolor(xy_coordinates[:,0], xy_coordinates[:,1], triangles, 
                                 facecolors=colours, **{"cmap": "coolwarm"}, vmin=-10, vmax=0.0)
        ax.margins(0)
        ax.set_aspect("equal")
        fig.colorbar(tripcolor)
        # Define text:
        textsize = 7
        for i, (xi,yi) in enumerate(top_value_positions):
            plt.text(xi,yi,round(top.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        for i, (xi,yi) in enumerate(right_value_positions):
            plt.text(xi,yi,round(right.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        for i, (xi,yi) in enumerate(left_value_positions):
            plt.text(xi,yi,round(left.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        for i, (xi,yi) in enumerate(bottom_value_positions):
            plt.text(xi,yi,round(bottom.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        ax.axis('off')
        plt.title("Q-Function, Epoch "+str(epoch))
        for i in range(lines+1):
            x = [0, rows]
            y = [i, i]
            plt.plot(x,y, color='black')
        for i in range(rows+1):
            x = [i, i]
            y = [0, lines]
            plt.plot(x,y, color='black')
        plt.show()

    """ # for frozen_lake
    def visualize(self, epoch):
        self.plot_action_value(epoch)

    def plot_action_value(self, epoch): 
        fig, ax=plt.subplots(figsize=(7,5))
        lines = 4
        rows = 4        
        # Define grid positions:
        pos_x_left = 0.2
        pos_x_mid = 0.5
        pos_x_right = 0.8
        pos_y_up = 0.2
        pos_y_mid = 0.5
        pos_y_down = 0.8
        grid_size = {'x': lines, 'y': rows}
        def gridcreator(pos_x, pos_y):
            grid = []
            for i in range(grid_size['x']):
                for j in range(grid_size['y']):
                    x = pos_x + j
                    y = pos_y + i
                    grid.append((x, y))
            return grid
        top = self.action_value_fn[:,3].reshape((lines, rows))
        top_value_positions = gridcreator(pos_x_mid, pos_y_up)
        right = self.action_value_fn[:,2].reshape((lines, rows))
        right_value_positions = gridcreator(pos_x_right, pos_y_mid)
        bottom = self.action_value_fn[:,1].reshape((lines, rows))
        bottom_value_positions = gridcreator(pos_x_mid, pos_y_down)
        left= self.action_value_fn[:,0].reshape((lines, rows))
        left_value_positions = gridcreator(pos_x_left, pos_y_mid)
        # Define triangles
        ax.set_ylim(lines, 0)
        anchor_points = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]]) # Corner coordinates
        corner_indizes = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]]) # Corner indices
        xy_coordinates = np.zeros((lines * rows * 5,2))
        triangles = np.zeros((lines * rows * 4, 3))
        for i in range(lines):
            for j in range(rows):
                k = i*rows+j
                xy_coordinates[k*5:(k+1)*5,:] = np.c_[anchor_points[:,0]+j, 
                                                      anchor_points[:,1]+i]
                triangles[k*4:(k+1)*4,:] = corner_indizes + k*5
        colours = np.c_[left.flatten(), top.flatten(), 
                right.flatten(), bottom.flatten()].flatten()
        ax.triplot(xy_coordinates[:,0], xy_coordinates[:,1], triangles, 
                   **{"color":"k", "lw":1})
        tripcolor = ax.tripcolor(xy_coordinates[:,0], xy_coordinates[:,1], triangles, 
                                 facecolors=colours, **{"cmap": "coolwarm"})
        ax.margins(0)
        ax.set_aspect("equal")
        fig.colorbar(tripcolor)
        # Define text:
        textsize = 10
        for i, (xi,yi) in enumerate(top_value_positions):
            plt.text(xi,yi,round(top.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        for i, (xi,yi) in enumerate(right_value_positions):
            plt.text(xi,yi,round(right.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        for i, (xi,yi) in enumerate(left_value_positions):
            plt.text(xi,yi,round(left.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        for i, (xi,yi) in enumerate(bottom_value_positions):
            plt.text(xi,yi,round(bottom.flatten()[i],2), size=textsize, color="w", 
                     ha='center', va='center')
        ax.axis('off')
        plt.title("Q-Function, Epoch "+str(epoch))
        for i in range(lines+1):
            x = [0, rows]
            y = [i, i]
            plt.plot(x,y, color='black')
        for i in range(rows+1):
            x = [i, i]
            y = [0, lines]
            plt.plot(x,y, color='black')
        plt.show()
    """

if __name__ == "__main__":

    """ # for frozen_lake:
    map = ["SFFH", "FFFH", "HFFH", "HFFG"]
    env = gym.make('CustomFrozenLake-v1', render_mode=None, desc=map, is_slippery=False) # set render_mode=None for speeding up learning
    env.reset()
    agent = TD_Agent(env, gamma=0.9, learning_rate=0.1, epsilon=0.1)

    input("Press Enter to train the agent")
    agent.train(num_episodes=100)

    input("Press Enter to evaluate the agent")
    env = gym.make('CustomFrozenLake-v1', render_mode='human', desc=map, is_slippery=False) # set render_mode=None for speeding up learning
    num_runs = 5
    agent.evaluate(env, num_runs=num_runs)
    """

    # for cliff_walking:
    env = gym.make('CliffWalking-v1', render_mode=None)
    env.reset()
    agent = TD_Agent(env, gamma=0.9, learning_rate=0.1, epsilon=0.3)

    input("Press Enter to train the agent")
    agent.train(num_episodes=100)

    # Compare the returns
    input("Press Enter to evaluate the agent")
    env = gym.make('CliffWalking-v1', render_mode='human')
    agent.evaluate(env, num_runs=100)