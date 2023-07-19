# PASTE OF THE CODE NEEDED TO DEFINE THE CLASSES

######################
##   Imports        ##
######################

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.animation as animation



#########################################
##                                     ##
##           ENVIROMENT                ##
##                                     ##
#########################################

"""
We have 4 moves: Up=0, Down=1, Left=2, Right=3
"""

class Game2048Env(gym.Env, object):
    
    def __init__(self):
        super(Game2048Env, self).__init__()

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        # self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int16)

        # Initialize the game board
        self.board = np.zeros((4, 4), dtype=np.int16)
        self.score = np.int16(0)

        # Set the initial state
        self.reset()

    def reset(self):
        # Reset the game board and score
        self.board = np.zeros((4, 4), dtype=np.int16)
        self.score = 0

        # Add two random tiles to the board
        self._add_random_tile()
        self._add_random_tile()

        return self.board, {}

    def is_legit_action(self):
        # Check if the action is valid
        return self._can_DOWN() or self._can_UP() or self._can_LEFT() or self._can_RIGHT()

    def _update_score(self, delta):
        self.score += delta

    def _UP(self):
        delta = 0
        for i in range(4):
            col = self.board[:, i]
            col = col[col != 0]
            col = np.concatenate((col, np.zeros(4-len(col))))
            for j in range(3):
                if col[j] == col[j+1]:
                    col[j] = col[j] * 2
                    delta += col[j]
                    col[j+1] = 0
            col = col[col != 0]
            col = np.concatenate((col, np.zeros(4-len(col))))
            self.board[:, i] = col
        self._update_score(delta)
        return delta

    # function to check if I can move up
    def _can_UP(self):
        for i in range(4):
            # need to check if any zeros in between
            col = self.board[:, i]
            for j in range(3):
                if col[j] == 0 and col[j+1] != 0:
                    return True
            # need to check if two consecutive numbers are the same
            for j in range(3):
                if col[j] == col[j+1]:
                    return True
        return False

    def _DOWN(self):
        delta = 0
        for i in range(4):
            col = self.board[:, i]
            col = col[col != 0]
            col = np.concatenate((np.zeros(4-len(col)), col))
            for j in range(3, 0, -1):
                if col[j] == col[j-1]:
                    col[j] = col[j] * 2
                    delta += col[j]
                    col[j-1] = 0
            col = col[col != 0]
            col = np.concatenate((np.zeros(4-len(col)), col))
            self.board[:, i] = col
        self._update_score(delta)
        return delta

    def _can_DOWN(self):
        for i in range(4):
            col = self.board[:, i]
            # need to check if any zeros in between
            for j in range(3, 0, -1):
                if col[j] == 0 and col[j-1] != 0:
                    return True
            # need to check if two consecutive numbers are the same
            for j in range(3, 0, -1):
                if col[j] == col[j-1]:
                    return True
        return False

    def _LEFT(self):
        delta = 0
        for i in range(4):
            row = self.board[i, :]
            row = row[row != 0]
            row = np.concatenate((row, np.zeros(4-len(row))))
            for j in range(3):
                if row[j] == row[j+1]:
                    row[j] = row[j] * 2
                    delta += row[j]
                    row[j+1] = 0
            row = row[row != 0]
            row = np.concatenate((row, np.zeros(4-len(row))))
            self.board[i, :] = row
        self._update_score(delta)
        return delta

    def _can_LEFT(self):
        for i in range(4):
            row = self.board[i, :]
            # need to check if any zeros in between
            for j in range(3):
                if row[j] == 0 and row[j+1] != 0:
                    return True
            # need to check if two consecutive numbers are the same
            for j in range(3):
                if row[j] == row[j+1]:
                    return True
        return False

    def _RIGHT(self):
        delta = 0
        for i in range(4):
            row = self.board[i, :]
            row = row[row != 0]
            row = np.concatenate((np.zeros(4-len(row)), row))
            for j in range(3, 0, -1):
                if row[j] == row[j-1]:
                    row[j] = row[j] * 2
                    delta += row[j]
                    row[j-1] = 0
            row = row[row != 0]
            row = np.concatenate((np.zeros(4-len(row)), row))
            self.board[i, :] = row
        self._update_score(delta)
        return delta

    def _can_RIGHT(self):
        for i in range(4):
            row = self.board[i, :]
            # need to check if any zeros in between
            for j in range(3, 0, -1):
                if row[j] == 0 and row[j-1] != 0:
                    return True
            # need to check if two consecutive numbers are the same
            for j in range(3, 0, -1):
                if row[j] == row[j-1]:
                    return True
        return False

    def step(self, action, verbose=False):
        # Perform the specified action
        # Update the board and score accordingly

        if (action == 0):
            delta = self._UP()

        if action == 1: # Down
            delta = self._DOWN()

        if action == 2: # Left
            delta = self._LEFT()

        if action == 3: # Right
            delta = self._RIGHT()

        # Check if the game is over (no more valid moves or 2048 tile reached)
        done, won = self._is_game_over()

        if verbose:
            print(f"-------------")
            print("Action: ", action)
            print("Score: ", self.score)
            print("Board: \n", self.board)
            print("Done: ", done)
            print(f"-------------")
        # Return the updated board, reward, done flag, and additional info
        return self.board, delta, done, won, {}

    def _render(self, mode='human', close=False):
        # Implement the rendering of the game board
        pass

    def _is_full_board(self):
        return np.all(self.board != 0)

    def _add_random_tile(self, p=0.9):
        # return a couple of random indices representing the board coordinates
        while True:
            idx = np.random.randint(low=0, high=4, size=(2,))
            if self.board[idx[0], idx[1]] == 0:
                self.board[idx[0], idx[1]] = np.random.choice([2, 4], p=[p, 1-p])
                break
            if self._is_full_board():
                break

    def _is_game_over(self):
        # Check if the game is over (no more valid moves or 2048 tile reached)
        if np.any(self.board > 2047):  # 2048 tile reached, we are using float32 so we can't check for == 2048
            return True, True
        # Add a random tile to the board only if the action done was valid
        if not self.is_legit_action():
            return True, False

        self._add_random_tile()
        return False, False


#####################################################
##                                                ##
##                AGENT DEFINITION                ##
##                                                ##
#####################################################

class RandomAgent(object):
    def __init__(self, env):
        self.action_space = env.action_space
        self.cumreward = 0

    def act(self, observation=None, reward=None, done=None):
        return self.action_space.sample()

    def reset(self):
        self.cumreward = 0


####################################################
##                                                ##
##        FUNCTION DEFINED TO PLOT RESULTS        ##
##                                                ##
####################################################



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
    ax.text(3.5, -0.8, action_to_arrow(agent.act()), ha="center", va="center", color="black", fontsize=35)
    

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

def play_game(env, agent, verbose=False, plot=False, fname_plt="2048.jpg", gif=False, fname_gif="2048.gif", fps=4.5):
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
    agent.reset()
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

    if plot:
        plot_board(env, agent, to_save=True)

    return env.score, agent.cumreward


def play_games(env, agent, n_games=10, verbose=False):
    """
    Play n_games with the given agent and return the scores
    :param env:   environment to play in
    :param agent: agent to play with
    :param n_games: number of games to play
    :param verbose: should we print the board at each step? (it will be done for each game)
    :return:    scores, cumrewards
    """
    scores = np.zeros(n_games)
    cumrewards = np.zeros(n_games)
    for i in range(n_games):
        score, cumreward = play_game(env, agent, verbose=verbose)
        scores[i] = score
        cumrewards[i] = cumreward
    return {"Scores":scores,
            "Cumrewards": cumrewards}



#####################################################
##                                                ##
##                PLAY THE GAME                   ##
##                                                ##
#####################################################


env = Game2048Env() # instantiate the game environment
random_agent = RandomAgent(env) # instantiate the random agent
play_game(env, random_agent, verbose=False, plot=True, gif=True, fname_gif="2048.gif") # play a game with the random agent and save it as a gif