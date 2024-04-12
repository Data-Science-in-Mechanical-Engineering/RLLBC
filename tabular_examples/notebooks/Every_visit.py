import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import custom_envs

class MC_Agent():
    def __init__(self, env, gamma=0.9, epsilon=0.1, q_tolerance=0.001):
        self.env = env
        self.ret_sum = defaultdict(float) # Keeping track of sum of reward in each state
        self.ret_count = defaultdict(float) # Keeping track of count in each state
        self.action_value_fn = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.gamma = gamma
        self.epsilon = epsilon # choose 0.0 to make a totally greedy policy, 1.0 for a fully random policy
        self.q_tolerance = q_tolerance

    def get_random_action(self):
        random_action = np.random.choice(range(self.env.action_space.n))
        return random_action
    def get_best_action(self, obs):
        best_action = np.random.choice(np.flatnonzero(np.isclose(self.action_value_fn[obs], self.action_value_fn[obs].max(),
                                                                 rtol=self.q_tolerance))) 
        return best_action
        
    def epsilon_greedy_policy(self, obs):
        # returns action, choosing a random action with probability epsilon, or the best action
        # regarding to Q with probability (1 - epsilon)
        randomly = np.random.random() < self.epsilon # float between 0.0 and 1.0 is generated,
        # being lower than epsilon leads to a random choice
        if randomly:
            action = self.get_random_action()
        else:
            action = self.get_best_action(obs)
        return action

    def train(self, num_episodes, episode_max_duration=100000):
        # Run through episodes sampled to improve policy incrementally
        for i_episode in range(1, num_episodes + 1):
            # Generate an episode, an array of (state, action, reward) tuples
            episode = []
            obs, info = env.reset()
            for t in range(episode_max_duration):
                action = self.epsilon_greedy_policy(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                episode.append((obs, action, reward))
                if done:
                    break
                obs = next_obs
            episode = np.array(episode)
            episode_duration = len(episode[:,:1])
            # Calculate returns for the whole episode from the back to save resources
            G = np.zeros([episode_duration, ])
            for i in range(episode_duration - 1, -1, -1):
                if i + 1 > episode_duration - 1:
                    G[i] = episode[i][2] # Last step of the episode
                else:
                    G[i] = episode[i][2] + self.gamma * G[i + 1]  # Every other step
            # Update the policy with average over all episodes
            for index in range(episode_duration):
                obs = episode[index][0]
                action = episode[index][1]
                self.ret_sum[(obs, action)] += G[index]
                self.ret_count[(obs, action)] += 1.0
                update = self.ret_sum[(obs, action)] / self.ret_count[(obs, action)]
                self.action_value_fn[int(obs)][int(action)] = update
            if i_episode % 10000 == 0:
                self.visualize(i_episode)

    def evaluate(self, env, num_runs=1):
        tot_reward = [0]
        for _ in range(num_runs):
            done = False
            obs, info = env.reset()
            reward_per_run = 0
            while not done:
                action = self.get_best_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                reward_per_run += reward
            tot_reward.append(reward_per_run + tot_reward[-1])
        return tot_reward
    
    def visualize_evals(self):
        x = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        y = np.array(['1', '2', '3', '4'])
        # Only a part of the value function array is necessary for showing the policy. This is sliced following S&B:
        retcounts = np.zeros((self.env.observation_space.n, 1))
        for i in range(self.action_value_fn.shape[0]):
            for j in range(4):
                retcounts[i] += self.ret_count[(i, j)]
        # Print the value function
        retcounts = retcounts.reshape(12, 4)
        print(retcounts)
        fig, ax = plt.subplots(nrows=1, figsize=(12, 4), subplot_kw={'projection': '3d'})
        X, Y = np.meshgrid(y.astype(int), x.astype(int))
        ax.plot_wireframe(X, Y, retcounts)
        ax.set_zlim(0, 200000)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlabel('retcount')
        plt.show()
    
    # for cliff_walking:
    def visualize(self, episode):
        self.plot_action_value(episode)

    def plot_action_value(self, episode): 
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
                                 facecolors=colours, **{"cmap": "winter"}, vmin=-50, vmax=0.0)
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
        plt.title("Q-Function, Episode "+str(episode))
        # add lines for separation
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
    # for frozen_lake:
    def visualize(self, episode):
        self.plot_action_value(episode)

    def plot_action_value(self, episode): 
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
        plt.title("Q-Function, Episode "+str(episode))
        # add lines for separation
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

########################################################################################################################

if __name__ == "__main__":

    """
    map = ["SFFH", "FFFH", "HFFH", "HFFG"]
    env = gym.make('CustomFrozenLake-v1', render_mode=None, desc=map, is_slippery=False)
    env.reset()
    agent = MC_Agent(env, gamma=0.9, epsilon=0.1)

    input("Press Enter to train the agent")
    agent.train(num_episodes=5000)

    input("Press Enter to evaluate the agent")
    num_runs = 100
    env = gym.make('CustomFrozenLake-v1', render_mode='human', desc=map, is_slippery=False)
    agent.evaluate(env, num_runs=num_runs)
    """

    # for cliff_walking:
    # Warning: There has to be a bug with the environment
    env = gym.make('CliffWalking-v1', render_mode=None)
    env.reset()
    agent = MC_Agent(env, gamma=0.9, epsilon=0.3, q_tolerance=0.1)

    input("Press Enter to train the agent")
    agent.train(num_episodes=10000, episode_max_duration=100)

    # Compare the returns
    input("Press Enter to evaluate the agent")
    env = gym.make('CliffWalking-v1', render_mode='human')
    agent.visualize_evals()
    agent.evaluate(env, num_runs=100)

