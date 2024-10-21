import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_action_value(q_fn, title): 
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
    top = q_fn[:,3].reshape((4,4))
    top_value_positions = gridcreator(pos_x_mid, pos_y_up)
    right = q_fn[:,2].reshape((4,4))
    right_value_positions = gridcreator(pos_x_right, pos_y_mid)
    bottom = q_fn[:,1].reshape((4,4))
    bottom_value_positions = gridcreator(pos_x_mid, pos_y_down)
    left= q_fn[:,0].reshape((4,4))
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
    plt.title(str(title))
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

def evaluate(agent, env, num_runs=1, file=None):
    frames = []
    video_created = False
    tot_reward = [0]
    for _ in range(num_runs):
        done = False
        obs, info = env.reset()
        reward_per_run = 0
        while not done:
            action = agent.get_best_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            reward_per_run += reward
            # save frame
            out = env.render()
            frames.append(out)
        tot_reward.append(reward_per_run + tot_reward[-1])

    # create animation out of saved frames
    if all(frame is not None for frame in frames):
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        img = plt.imshow(frames[0])

        def animate(index):
            img.set_data(frames[index])
            return [img]
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
        plt.close()
        anim.save(file, writer="ffmpeg", fps=5)
        video_created = True

    return tot_reward, video_created