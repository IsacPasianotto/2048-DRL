import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch

cmap = matplotlib.colors.ListedColormap(['#ffffff', '#f5f5f5', '#f5f5dc', '#ffa07a', '#ff7f50', '#ff7f50', '#ff0000', '#ff0066', '#f0e050', '#f0e010', '#f0e010', '#f0c000'])
bounds=[0,2,4,8,16,32,64,128,256,512,1024]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Function to convert action to arrow
def action_to_arrow(action):
    if action == 0: return "⬆"
    if action == 1: return "➡"
    if action == 2: return "⬇"
    if action == 3: return "⬅"
    return ""

def plot_board(env, agent, cmap=cmap, norm=norm, to_save=False, fname="2048.jpg"):
    fig, ax = plt.subplots()
    ax.imshow(env.board)
    ax.axis('off')

    # set the colormap for tiles
    ax.imshow(env.board, cmap=cmap, norm=norm)

    # color the tiles with the appropriate colors, add the score to the figure
    # and draw the gridlines
    for i in range(4):
        for j in range(4):
            if env.board[i, j] != 0:
                #plt.text(j, i, env.board[i, j], ha="center", va="center", fontsize=20, color="black")
                ax.text(j, i, env.board[i, j], ha="center", va="center", fontsize=20, color="black")
            rect = patches.Rectangle((i-0.5,j-0.5),1,1,linewidth=2,edgecolor='black',facecolor='none')
            ax.add_patch(rect)

    # Show the score on the left corner, the last action on the right corner
    ax.text(0, -0.8, "Score: " + str(int(env.score)), ha="center", va="center", color="black", fontsize=20)
    #ax.text(3, -0.8, "Last Action: " + action_to_arrow(agent.act()), ha="center", va="center", color="black", fontsize=20)

    # separate the label 'last action' from the arrow to make that one bigger
    ax.text(2.7, -0.8, "Last Action: ", ha="center", va="center", color="black", fontsize=12)

    # random agent
    # ax.text(3.5, -0.8, action_to_arrow(agent.act()), ha="center", va="center", color="black", fontsize=35)
    ax.text(3.5, -0.8, action_to_arrow(env.last_action), ha="center", va="center", color="black", fontsize=35)

    if to_save:
        # save the figure
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    if not to_save:
        # Return the image data instead of AxesImage object
        fig.canvas.draw()
        #frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame = np.array(fig.canvas.renderer._renderer)
        plt.close(fig)
        return frame
    plt.close(fig)
    #return p


#####################################################
##                                                ##
##                FUNCTION TO PLAY                ##
##                                                ##
#####################################################

def random_agent_gif(env, agent, verbose=False, gif=False, fname_gif="2048.gif", fps=4.5):
    """
    Play a game given an environment and an agent
    :param env:     environment to play in
    :param agent:   agent to play with
    :param verbose: should we print the board at each step?
    :param plot:    should we plot the board at the end of the game?
    :param fname_plt:   name of the file to save the plot
    :param gif:     should we save the game as a gif?
    :param fname_gif:   name of the file to save the gif
    :param fps:     frames per second of the gif
    :return:   env.score, agent.cumreward
    """
    env.reset()
    # agent.reset()
    done = False
    frames = []  # Store frames for gif

    if gif:
        fig = plt.figure(layout='compressed')

    while not done:
        if gif:
            frame = plot_board(env, agent, to_save=False)
            plt.axis('off')
            frames.append([plt.imshow(frame)])



        action = agent.act()
        _, reward, done, won, _ = env.step(action, verbose=verbose)
        agent.cumreward += reward - 2
        if won:
            agent.cumreward += 300000

    if gif:

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True) # blit=True to speed up the animation
        ani.save(fname_gif, writer='ffmpeg', fps=fps) # uaw ffmpeg writer
        # other possible writer:
        #ani.save(fname_gif, writer='imagemagick', fps=5)
        #ani.save(fname_gif, writer='pillow', fps=5)

    return env.score, agent.cumreward



def dqn_gif(env, agent, verbose=False, gif=False, fname_gif="2048.gif", fps=4.5) :
    """
    Play a game given an environment and an agent
    :param env:     environment to play in
    :param agent:   agent to play with
    :param verbose: should we print the board at each step?
    :param plot:    should we plot the board at the end of the game?
    :param fname_plt:   name of the file to save the plot
    :param gif:     should we save the game as a gif?
    :param fname_gif:   name of the file to save the gif
    :param fps:     frames per second of the gif
    :return:   env.score, agent.cumreward
    """
    state, _ = env.reset()
    state = torch.tensor(state.flatten(), dtype=torch.float32, device=agent.device).unsqueeze(0)
    # agent.reset()
    done = False
    frames = []  # Store frames for gif

    if gif:
        fig = plt.figure(layout='compressed')

    while not done:
        if gif:
            frame = plot_board(env, agent, to_save=False)
            plt.axis('off')
            frames.append([plt.imshow(frame)])

        # metto in modalità di evalutazione
        agent.policy_net.eval()
        action = agent.select_action(state)
        new_state, reward, done, won, _ = env.step(action, verbose=verbose)
        #update the state
        state = torch.tensor(new_state.flatten(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        #state = new_state
        # agent.cumreward += reward - 2
        # if won:
        #    agent.cumreward += 300000

    if gif:

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True) # blit=True to speed up the animation
        ani.save(fname_gif, writer='ffmpeg', fps=fps) # uaw ffmpeg writer
        # other possible writer:
        #ani.save(fname_gif, writer='imagemagick', fps=5)
        #ani.save(fname_gif, writer='pillow', fps=5)

    return env.score, agent.score
