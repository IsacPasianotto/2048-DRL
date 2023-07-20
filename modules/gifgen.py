"""
This module is not strictly necessary for the project, but is useful to
visualize graphically how an agent is playing.
It was used to generate the gif in the report and for debugging purposes,
but it is not used in the training of the agents.
Please remember that duo to its nature, this module is not
as well documented, tested and maintained as the other modules.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch
import modules.agents as agents
import modules.envs as envs
import modules.architectures as architectures


# Constants
cmap = matplotlib.colors.ListedColormap(['#ffffff', '#f5f5f5', '#f5f5dc', '#ffa07a', '#ff7f50', '#ff7f50', '#ff0000', '#ff0066', '#f0e050', '#f0e010', '#f0e010', '#f0c000', '#ffff00'])
bounds=[0,2,4,8,16,32,64,128,256,512,1024,2048]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)


def action_to_arrow(action):
    """
    Convert an action to an arrow
    Args:
        action: int, action to convert

    Returns:
        arrow: str, arrow corresponding to the action
    """  
    
    if action == 0: return "⬆"
    if action == 1: return "⬇"
    if action == 2: return "⬅"
    if action == 3: return "➡"
    return ""

def plot_board(env, agent, cmap=cmap, norm=norm, to_save=False, fname="2048.jpg", last_action=""):
    """
    Plot the board of the 2048 game with the score and the last action
    Args:
        env: the 2048Game environment
        agent: the agent playing the game
        cmap: colormap
        norm: norm
        to_save: boolean, should we save the figure? (default: False)
        fname: if to_save, name of the file to save the figure (default: "2048.jpg")
        last_action: str of the last action performed

    Returns:
        frame: the image of the board (useful for the gif)
    """

    fig, ax = plt.subplots()
    ax.imshow(env.board)
    ax.axis('off')

    ax.imshow(env.board, cmap=cmap, norm=norm)

    # color the tiles with the appropriate colors, add the score to the figure
    # and draw the gridlines
    for i in range(4):
        for j in range(4):
            if env.board[i, j] != 0:
                plt.text(j, i, env.board[i, j], ha="center", va="center", fontsize=20, color="black")
                ax.text(j, i, env.board[i, j], ha="center", va="center", fontsize=20, color="black")
            rect = patches.Rectangle((i-0.5,j-0.5),1,1,linewidth=2,edgecolor='black',facecolor='none')
            ax.add_patch(rect)

    # Show the score on the left corner, the last action on the right corner
    ax.text(0, -0.8, "Score: " + str(int(env.score)), ha="center", va="center", color="black", fontsize=20)
    #ax.text(3, -0.8, "Last Action: " + action_to_arrow(last_action), ha="center", va="center", color="black", fontsize=20)

    # separate the label 'last action' from the arrow to make that one bigger
    ax.text(2.7, -0.8, "Last Action: ", ha="center", va="center", color="black", fontsize=12)

    # random agent
    #ax.text(3.5, -0.8, action_to_arrow(agent.act()), ha="center", va="center", color="black", fontsize=35)
    ax.text(3.5, -0.8, action_to_arrow(last_action), ha="center", va="center", color="black", fontsize=35)

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

def play_gameRA(env, agent, verbose=False, plot=False, fname_plt="2048.jpg", gif=False, fname_gif="2048.gif", fps=4.5):
    """
    Play a game given an environment and an agent, implemented for the random agent.
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
    agent.reset()
    done = False
    frames = []  # Store frames for gif

    if gif:
        fig = plt.figure(layout='compressed')
    action = None
    while not done:
        if gif:
            frame = plot_board(env, agent, to_save=False, last_action=action)
            plt.axis('off')
            frames.append([plt.imshow(frame)])
        # sub-optimal, no check for legal moves, but for random agent it's ok
        action = agent.act(np.array([0,1,2,3]))
        _, reward, done, won, _ = env.step(action, verbose=verbose)
        agent.cumreward += reward - 2
        if won:
            agent.cumreward += 300000

    if gif:
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)  # blit=True to speed up the animation
        ani.save(fname_gif, writer='ffmpeg', fps=fps)  # uaw ffmpeg writer
        # other possible writer:
        # ani.save(fname_gif, writer='imagemagick', fps=5)
        # ani.save(fname_gif, writer='pillow', fps=5)

    if plot:
        plot_board(env, agent, to_save=True, last_action=action)

    return env.score, agent.cumreward

def binary_tensor(state):
    """Convert the state of the game to a binary tensor.
    The tensor is of shape (12, 4, 4), where the first dimension
    represents the power of 2 of the tile, and the other two dimensions
    represent the position of the tile on the board.
    This is needed to make in order to use the function _select_action later
    """
    tensor = torch.zeros(12, 4, 4)
    tensor[0, :, :] = torch.tensor(state == 0, dtype=torch.float32)
    for i in range(1, 12):
        tensor[i, :, :] = torch.tensor(state == 2 ** i, dtype=torch.float32)
    return tensor



def play_gameDQN (env, agent, verbose=False, gif=False, fname_gif="2048.gif", fps=4.5, plot=False, fname_plt="2048.png"):

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
    # state = torch.tensor(state. flatten(), dtype=torch.float32, device=agent.device).unsqueeze(0)

    frames = []  # Store frames for gif
    if gif:
        fig = plt.figure(layout='compressed')

    done = False
    action = None

    agent.policy_net.eval()

    while not done:
        if gif:
            frame = plot_board(env, agent, to_save=False, last_action=action)
            plt.axis('off')
            frames.append([plt.imshow(frame)])
        # The agent now must play, not learn again
        state = binary_tensor(state)
        # select and perform an action
        action = agent.select_action(state, env.get_legit_actions(), train=False)
        new_state, reward, done, won, _ = env.step(action, verbose=verbose)
        # update state
        state = new_state
        
    if gif:

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True) # blit=True to speed up the animation
        ani.save(fname_gif, writer='ffmpeg', fps=fps) # uaw ffmpeg writer
        # other possible writer:
        #ani.save(fname_gif, writer='imagemagick', fps=5)
        #ani.save(fname_gif, writer='pillow', fps=5)
    if plot:
        plot_board(env, agent, to_save=True, last_action=action, fname=fname_plt)


# Random Agent

# env = envs.Game2048Env()
# random_agent = agents.RandomAgent()
# play_gameRA(env, random_agent, verbose=False, plot=True, fname_plt="2048.jpg", gif=True)

# DQN-agent with convolutional network

# env = envs.Game2048Env()
# net = architectures.ConvolutionalNetwork()
# agent = agents.ConvDQN_Agent(model=net)
# agent.fit(num_episodes=2, env=env, verbose=True)
# agent.load_model("path_to_model.pt")
# play_gameDQN(env, agent, verbose=False, gif=True, fname_gif="2048.gif", fps=4.5, plot=True, fname_plt="2048.jpg")


# env = envs.Game2048Env()
# random_agent = agents.RandomAgent()
# play_gameRA(env, random_agent, verbose=False, plot=False, gif=True, fname_gif="2048_random_agent.gif")

# env = envs.Game2048Env()
# model = architectures.BigConvolutionalNetwork()
# trained_agent = agents.ConvDQN_Agent(model=model)

# import sys
# sys.path.append('../')

# trained_agent.load("trained_architectures/convdqn_agent_long_train.pt")

# play_gameDQN(env, trained_agent, verbose=False, gif=True, fname_gif="2048_trained_agent.gif", fps=4.5)
