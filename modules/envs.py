import gym
from gym import spaces
import numpy as np
import torch 
 
class Game2048Env(gym.Env, object):
    """_Extension of the class gym.Env which represent
    the environment of the game 2048. The board is represented
    with a 2D array of size 4x4. The actions are encoded as
    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT. The reward is the increase
    of the score. The game is over when the board is full and no
    move is possible or when the tile 2048 is reached.

    Attributes:
        board (np.array): the board of the game
        action_space (gym.spaces.Discrete): the action space
        legit_actions (list): the list of legit actions
        score (np.int): the score of the game
        log_reward (bool): if True, the reward returned by the step function is the log2 of the increase of the score
    """

    def __init__(self, log_rewards=False):
        """_Initializes  the environment instancing the class attributes.
        Then calls the reset function to take it to the starting configuration._

        Args:
            log_reward (bool, optional): _The reward should be calculates as log2(reward)?_. Defaults to False.
        """

        super(Game2048Env, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.legit_actions = np.ones(4, dtype=np.int16)
        self.board = np.zeros((4, 4), dtype=np.int16)
        self.score = np.int16(0)
        self.log_rewards = log_rewards

        # Set the initial state
        self.reset()

    def reset(self):
        """_Resets the environment to the starting configuration.
        This is achieved by clearing the board, setting the score to 0 and
        adding two random tiles to the board._

        Returns:
            _self.board_: _the 2D numpy array representing the board_
            _{}_: _empty dictionary (required by the gym.Env class)_
        """

        self.board = np.zeros((4, 4), dtype=np.int16)
        self.score = np.int16(0)

        # Add two random tiles to the board
        self._add_random_tile()
        self._add_random_tile()
        self.legit_actions = self.get_legit_actions()

        return self.board, {}

    def get_legit_actions(self):
        """_Checks among the 4 possibles actions which one is legal_

        Returns:
            _np.array((4))_: _a np.array with values True if the move is legit, False otherwise. The order is up, down, left right._
        """

        return np.array([self._can_UP(), self._can_DOWN(), self._can_LEFT(), self._can_RIGHT()])

    def _update_score(self, delta):
        """_Increments the score by delta_

        Args:
            delta (_int_): _How much the score should be incremented_
        """

        self.score += delta

    def _UP(self):
        """_Performs the action "UP". all the tiles are moved to the most upper position
        they can reach. If two tiles with the same value are in the same column, they are
        summed and the score is updated._

        Returns:
            _int_: _The sum of the values (or their log2 values if self.log_reward is True) of the
            new generates tile performing the action_
        """

        delta = 0
        for i in range(4):
            col = self.board[:, i]
            col = col[col != 0]
            col = np.concatenate((col, np.zeros(4-len(col))))
            for j in range(3):
                if col[j] == col[j+1]:
                    col[j] = col[j] * 2
                    self._update_score(col[j])
                    if self.log_rewards and col[j] != 0:
                        delta += np.log2(col[j])
                    else:
                        delta += col[j]
                    col[j+1] = 0
            col = col[col != 0]
            col = np.concatenate((col, np.zeros(4-len(col))))
            self.board[:, i] = col
        return delta

    def _can_UP(self):
        """_Checks if an action "UP" could change the board_

        Returns:
            _Bool_: _True, if "UP" is a legit action, False otherwise_
        """

        for i in range(4):
            col = self.board[:, i]
            for j in range(1, 4):
                if col[j] != 0:
                    if col[j - 1] == 0 or col[j - 1] == col[j]:
                        return True
        return False

    def _DOWN(self):
        """_Performs the action "DOWN". all the tiles are moved to the most lower position
        they can reach. If two tiles with the same value are in the same column, they are
        summed and the score is updated._

        Returns:
            _int_: _The sum of the values (or their log2 values if self.log_reward is True) of the
            new generates tile performing the action_
        """

        delta = 0
        for i in range(4):
            col = self.board[:, i]
            col = col[col != 0]
            col = np.concatenate((np.zeros(4-len(col)), col))
            for j in range(3, 0, -1):
                if col[j] == col[j-1]:
                    col[j] = col[j] * 2
                    self._update_score(col[j])
                    if self.log_rewards and col[j] != 0:
                        delta += np.log2(col[j])
                    else:
                        delta += col[j]
                    col[j-1] = 0
            col = col[col != 0]
            col = np.concatenate((np.zeros(4-len(col)), col))
            self.board[:, i] = col
        return delta

    def _can_DOWN(self):
        """_Checks if an action "DOWN" could change the board_

        Returns:
            _Bool_: _True, if "DOWN" is a legit action, False otherwise_
        """
        
        for i in range(4):
            col = self.board[:, i]
            for j in range(0, 3):
                if col[j] != 0:
                    if col[j + 1] == 0 or col[j + 1] == col[j]:
                        return True
        return False

    def _LEFT(self):
        """_Performs the action "LEFT". all the tiles are moved to the most left position
        they can reach. If two tiles with the same value are in the same row, they are
        summed and the score is updated._

        Returns:
            _int_: _The sum of the values (or their log2 values if self.log_reward is True) of the
            new generates tile performing the action_
        """

        delta = 0
        for i in range(4):
            row = self.board[i, :]
            row = row[row != 0]
            row = np.concatenate((row, np.zeros(4-len(row))))
            for j in range(3):
                if row[j] == row[j+1]:
                    row[j] = row[j] * 2
                    self._update_score(row[j])
                    if self.log_rewards and row[j] != 0:
                        delta += np.log2(row[j])
                    else:
                        delta += row[j]
                    row[j+1] = 0
            row = row[row != 0]
            row = np.concatenate((row, np.zeros(4-len(row))))
            self.board[i, :] = row
        return delta
    
    def _can_LEFT(self):
        """_Checks if an action "LEFT" could change the board_

        Returns:
            _Bool_: _True, if "LEFT" is a legit action, False otherwise_
        """

        for i in range(4):
            row = self.board[i, :]
            for j in range(1, 4):
                if row[j] != 0:
                    if row[j - 1] == 0 or row[j - 1] == row[j]:
                        return True
        return False

    def _RIGHT(self):
        """_Performs the action "LEFT". all the tiles are moved to the most left position
        they can reach. If two tiles with the same value are in the same row, they are
        summed and the score is updated._

        Returns:
            _int_: _The sum of the values (or their log2 values if self.log_reward is True) of the
            new generates tile performing the action_
        """

        delta = 0
        for i in range(4):
            row = self.board[i, :]
            row = row[row != 0]
            row = np.concatenate((np.zeros(4-len(row)), row))
            for j in range(3, 0, -1):
                if row[j] == row[j-1]:
                    row[j] = row[j] * 2
                    self._update_score(row[j])
                    if self.log_rewards and row[j] != 0:
                        delta += np.log2(row[j])
                    else:
                        delta += row[j]
                    row[j-1] = 0
            row = row[row != 0]
            row = np.concatenate((np.zeros(4-len(row)), row))
            self.board[i, :] = row
        return delta
    
    def _can_RIGHT(self):
        """__Checks if an action "RIGHT" could change the board_
        Returns:
            _Bool_: _True, if "RIGHT" is a legit action, False otherwise_
        """

        for i in range(4):
            row = self.board[i, :]
            for j in range(2, -1, -1):
                if row[j] != 0:
                    if row[j + 1] == 0 or row[j + 1] == row[j]:
                        return True
        return False

    def _is_changed(self, board):
        """_Checks if the board has some differences with respect to
        a board passed as argument_

        Args:
            board (np.Array((4,4))): _The boar you want to check if is equal to self.board_

        Returns:
            _Bool_: _True if the 2 boards are equals, false otherwise_
        """
        return not np.array_equal(board, self.board)
    
    def step(self, action, verbose=False):
        """_Performs the action passed as argument, checks if the game is over
        and if it is not, adds a random tile to the board and updates the
        environment attributes. The reward is the increase of the score._

        Args:
            action (_int_): _The action to perform. 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT_
            verbose (bool, optional): _If True, prints relevant information about the step taken (debugging purposes)_. Defaults to False.

        Returns:
            _self.board (np.array((4,4))): _The new board after the action is performed_
            _delta (int)_: _The increase of the score after the action is performed_
            _done (bool)_: _True if the game is over, False otherwise_
            _won (bool)_: _True if the game is won (2048 reached), False otherwise_
            _{}_: _Empty dictionary (required by the gym.Env class)_
        """

        delta = 0
        board = self.board.copy()
        if action == 0:
            delta = self._UP()

        if action == 1: # Down
            delta = self._DOWN()

        if action == 2: # Left
            delta = self._LEFT()

        if action == 3: # Right
            delta = self._RIGHT()

        if self._is_changed(board):
            self._add_random_tile()
            self.legit_actions = self.get_legit_actions()
        else:
            print("Invalid action taken!")

        # Check if the game is over (no more valid moves or 2048 tile reached)
        done, won = self._is_game_over()

        if won:
            if self.log_rewards:
                delta += 11
            else:
                delta += 2048

        if verbose:
            print(f"-------------")
            print("Action: ", action)
            print("Score: ", self.score)
            print("Board: \n", self.board)
            print("Done: ", done)
            print("Won: ", won)
            print("legit actions: ", self.legit_actions)
            print("Is changed board: ", self._is_changed(board))
            print(f"-------------")
        # Return the updated board, reward, done flag, and additional info
        return self.board, delta, done, won, {}

    def _render(self, mode='human', close=False):
        # Implement the rendering of the game board
        pass

    def _is_full_board(self):
        """_Checks if the board is full_

        Returns:
            _Bool_: _True if the board is full, False otherwise_
        """
        return np.all(self.board != 0)

    def _add_random_tile(self, p=0.9):
        """_Extract a random free position on the board and add a tile with value 2 or 4
        with probability p and 1-p respectively. If the board is full, the function
        will not add any tile._

        Args:
            p (float, optional): _Probability to spawn a 2 tile (otherwise it will be spawned a 4)_. Defaults to 0.9.
        """

        while True:
            idx = np.random.randint(low=0, high=4, size=(2,))
            if self.board[idx[0], idx[1]] == 0:
                self.board[idx[0], idx[1]] = np.random.choice([2, 4], p=[p, 1-p])
                break
            if self._is_full_board():
                break

    def _is_game_over(self):
        """_Checks if the game is over. The game is over if the board is full and no
        move is possible or if the tile 2048 is reached._

        returns:
            _done (bool)_: _True if the game is over, False otherwise_
            _won (bool)_: _True if the game is won (2048 reached), False otherwise_

        """
        
        if np.any(self.board > 2047):  
            return True, True
        if not np.any(self.get_legit_actions()):
            return True, False
        return False, False