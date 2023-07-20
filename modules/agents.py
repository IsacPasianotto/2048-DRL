
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import warnings
from collections import namedtuple, deque
import tqdm
import numpy as np
from itertools import count
import os

class RandomAgent(object):
    """An agent that acts randomly. It is used as a baseline for the other agents.
    Every time the agent is asked to act, it samples a random action from the set of legit actions.
    """

    def __init__(self):
        """Initialize the agent object.
        """

        self.cumreward = 0

    def act(self, legit_actions):
        """Sample a random action from the set of legit actions.
        
        Args:
            legit_actiocumrewardns (_np.array_): A numpy array of 4 boolean values, one for each action (up, down, left, right).
                (True if the action is legit, False otherwise)
                
        Returns:
            _int_: The index of the action to perform: 0 for up, 1 for down, 2 for left, 3 for right.
        """

        acts = np.arange(4)[legit_actions]
        return np.random.choice(acts)
        

    def reset(self):
        """
        Reset the cumulative reward to 0.
        """

        self.cumreward = 0

# object for storing the transitions: s, a -> r, s'
Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer



# create a deque object with a max length of capacity, when new items are added and the length is above capacity, the oldest items are automatically removed
class ReplayMemory(object):
    """Create a deque object with a max length of capacity, when new items are added and the
    length is above capacity, the oldest items are automatically removed. It is used to store
    the transitions of the agent.
    """

    def __init__(self, capacity):
        """Initialize the ReplayMemory object.
        
        Args:
            capacity (_int_): The maximum number of transitions to store.
        """

        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Add a transition to the replay buffer.
        """

        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size (int): The number of transitions to sample.
            
        Returns:
            _list_: A list of transitions of length batch_size.
        """

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class ConvDQN_Agent():
    """
    This class implements a DQN agent with a convolutional neural network as model.
    It's act with an epsilon-greedy policy, which behavior is controlled by the parameters. 
    Since the space of the states is too large (upper bounded by approximately 11^16), a Q-learning
    approach is not feasible. Instead we use a DQN approach, where the Q-function which maps
    the state-action pair to the expected reward is approximated by a neural network
    which takes as input the state of the game and outputs the expected reward for each action.
    In particular, this class implements a DQN agent with a convolutional neural network as model.
    
    Args:
        model (torch.nn.Module): The model to use as policy network.
        device (str): The device to use for the computations (cpu or cuda).
        path_to_weights (str): The path to the weights of the model to load.
        batch_size (int): The number of transitions sampled from the replay buffer.
        gamma (float): The discount factor.
        eps_start (float): The starting value of epsilon.
        eps_end (float): The final value of epsilon.
        eps_decay (int): How slowly/fast epsilon decays. It's influenced by kind_epsg.
        kins_epsg (str): The kind of epsilon greedy policy to use. It can be: "torch_version" (default), "lai_robbins", "entropy".
        tau (float): The update rate of the target network.
        lr (float): The learning rate of the optimizer (AdamW).
        replay_memory_size (int): The maximum number of transitions to store in the replay buffer.
        epochs_checkpoint (int): The number of epochs between each checkpoint.
        penalty_move (int): The penalty for each move, used to reward shaping.
        target_net (torch.nn.Module): The target network.
        policy_net (torch.nn.Module): The policy network.
        loss_history (np.array): The loss history of the training.
        cumulative_reward (np.array): The cumulative reward history of the training.
        score (int): The achieved score of the agent.
        steps_done (int): The number of steps done by the agent.
        steps_per_action (np.array): The number of steps done for each action.        
    """

    def __init__(self, model, 
                 path_to_weigths=None, 
                 device='cpu',
                 batch_size=128,
                 gamma=0.9995,
                 eps_start=0.7,
                 eps_end=0.02,
                 eps_decay=1000,
                 tau=5e-5,
                 lr=1e-3,
                 alpha=1,
                 replay_memory_size=10000,
                 l2_reg=3e-3,
                 epochs_checkpoint=50,
                 kind_action="torch_version",
                 name="My_agent",
                 penalty_move=0):
        
        """Initialize the ConvDQN_Agent object. It creates the policy network and the target network,
        and initializes the optimizer, the replay buffer and the steps done counter.
        If path_to_weights is not None, the model is initialized with the weights stored in the file, otherwise
        the model is initialized with random weights and the training can start from scratch.
    
        Args:
            model (torch.nn.Module): The model to use as policy network. (It must be a subclass of torch.nn.Module)
            device (str, optional): The device to use for the computations (cpu or cuda). Defaults to 'cpu'.
            path_to_weights (str, optional): The path to the weights of the model to load. Defaults to None. If None, the model is initialized with random weights.
            batch_size (int, optional): The number of transitions sampled from the replay buffer. Defaults to 128.
            gamma (float, optional): The discount factor. Defaults to 0.9995.
            eps_start (float, optional): The starting value of epsilon. Defaults to 0.7.
            eps_end (float, optional): The final value of epsilon. Defaults to 0.02, it's used only in 'torch_version' of selected_action().
            eps_decay (int, optional): How slowly/fast epsilon decays, it's influenced by kind_epsg. Defaults to 1000.
            alpha (float, optional): The parameter of the entropy regularization (works only for kind_action="entropy"). Defaults to 1.
            kind_action (str, optional): How the epsilon greedy policy is computed. It can be: "torch_version" (default), "lai_robbins", "entropy".
            tau (float, optional): The update rate of the target network. Defaults to 5e-5.
            lr (float, optional): The learning rate of the optimizer (AdamW). Defaults to 1e-3.
            replay_memory_size (int, optional): _description_. Defaults to 10000.
            l2_reg (float, optional): L2 regularization coefficient. Defaults to 1e-4.
            epochs_checkpoint (int, optional): The number of epochs between each checkpoint. Defaults to 50.
            penalty_move (int, optional): The penalty for each move, used to reward shaping. Defaults to 0.
        
        Attributes:
            name (str): The name of the agent, if specified.
            kind_action (str): How the epsilon greedy policy is computed. It can be: "torch_version" (default), "lai_robbins", "entropy".
            policy_net (torch.nn.Module): The policy network.
            target_net (torch.nn.Module): The target network.
            loss_history (np.array): The loss history of the training.
            cumulative_reward (np.array): The cumulative reward history of the training.
            score (int): The achieved score of the agent.
            steps_done (int): The number of steps done by the agent.
            max_duration (int): The maximum duration of the game.
            action_dist (list): The distribution of the actions taken by the agent.
            max_tile (int): The maximum tile reached by the agent.
            num_actions (int): The number of actions available to the agent.
            optimizer (torch.optim.AdamW): The optimizer used to train the policy network.
            memory (ReplayMemory): The replay buffer.
            steps_per_action (np.array): The number of steps done for each action.
            device (str): The device to use for the computations (cpu or cuda).
            BATCH_SIZE (int): The number of transitions sampled from the replay buffer.
            GAMMA (float): The discount factor.
            EPS_START (float): The starting value of epsilon.
            EPS_END (float): The final value of epsilon.
            EPS_DECAY (int): How slowly/fast epsilon decays. It's influenced by kind_epsg.
            TAU (float): The update rate of the target network.
            LR (float): The learning rate of the optimizer (AdamW).
            REPLAY_MEMORY_SIZE (int): The maximum number of transitions to store in the replay buffer.
            LAMBDA (float): The coefficient of the L2 regularization.
            EPOCHS_CHECKPOINT (int): The number of epochs between each checkpoint.
            PENALTY_MOVE (int): The penalty for each move, used to reward shaping.
            replay_memory_size (int): The maximum number of transitions to store in the replay buffer.
            mean_board (np.array): The mean of the board of the game.
        """
        assert model is not None

        self.name = None
        self.kind_action = kind_action
        self.policy_net = model.to(device)
        if path_to_weigths is not None:
            self.policy_net.load_state_dict(torch.load(path_to_weigths, map_location=torch.device(device)))
        # init the target network with the same weights of the policy network
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.loss_history = None
        self.alpha=alpha
        self.cumulative_reward = None
        self.score = None
        self.max_duration = None
        self.action_dist = []
        self.max_tile = None
        self.num_actions = 4
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_size)
        self.replay_memory_size = replay_memory_size
        self.steps_done = 0
        self.device = device
        self.BATCH_SIZE = batch_size # number of transitions sampled from the replay buffer
        self.GAMMA = gamma # discount factor
        self.EPS_START = eps_start # epsilon greedy start value
        self.EPS_END = eps_end # epsilon greedy final value
        self.EPS_DECAY = eps_decay # number of steps to decay epsilon
        self.TAU = tau # coefficient for the soft update of the target network
        self.LR = lr # coefficient of the AdamW optimizer
        self.LAMBDA = l2_reg # coefficient of the L2 regularization
        self.EPOCHS_CHECKPOINT = epochs_checkpoint # number of epochs between each checkpoint
        self.penalty_move = penalty_move
        self.mean_board = np.zeros((4, 4))

        self.steps_per_action = np.zeros(4)


    def reset(self):
        """Reset the specs of the agent.
        """

        self.action_dist = []
        self.score = None
        self.loss_history = None
        self.max_duration = None
        self.cumulative_reward = None
        self.max_tile = None
        self.mean_board = np.zeros((4, 4))
        self.steps_per_action = np.zeros(4)
        self.memory = ReplayMemory(self.replay_memory_size)
        self.steps_done = 0
        
    def select_action(self, state, legit_actions, kind="torch_version", train=False):
        """Select an action to perform given the current state and the legit actions.
        The action is selected with an epsilon-greedy policy, where epsilon is computed
        according to the kind of epsilon greedy policy selected.
        In particular:
            - "torch_version": epsilon decays exponentially from eps_start to eps_end with eps_decay as rate. (see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
            - "lai_robbins": epsilon decays as 1/(1+0.01*steps_done) (see Lai & Robbins, 1985).
            - "entropy": epsilon is computed as the softmax of the policy multiplied by a beta coefficient.
        Args:
            state (torch.tensor(12, 4, 4)): The current state of the game, represented as a tensor of shape (12, 4, 4) encoding the board.
            legit_actions (np.array): A numpy array of 4 boolean values, one for each action (up, down, left, right).
            kind (str, optional): The kind of epsilon greedy policy to use. It can be: "torch_version" (default), "lai_robbins", "entropy"
            train (bool, optional): Whether the agent is training or not. Defaults to False.
            
        Returns:
            _torch.tensor_: The action to perform, encoded as a tensor of shape (1, 1).
        """
        
        # We want to adopt a policy that is greedy with respect to the policy network,
        # but we want to select only legit actions (given as input)
        
        match kind:
            case "lai_robbins":
                sample = 1
                if train:
                    sample = random.random()
                # Set the threshold for the epsilon-greedy policy.
                legit_actions = torch.tensor(legit_actions, device=self.device, dtype=torch.bool)
                state = state.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value_policy = self.policy_net(state)
                    filtered_policy = torch.where(legit_actions, value_policy, torch.tensor(-torch.inf, device=self.device, dtype=torch.float32))
                    chosen_action = filtered_policy.max(1)[1].item()

                self.steps_per_action[chosen_action] += 1
                self.action_dist.append(self.steps_per_action/np.sum(self.steps_per_action))
                eps_threshold = 0.8/(1.0+0.01*self.steps_per_action[chosen_action])
                self.steps_done += 1
                if sample > eps_threshold:
                    return torch.tensor(chosen_action, device=self.device, dtype=torch.long).view(1, 1)

                legit_actions = legit_actions.cpu().numpy()
                valid_indices = np.arange(4)[legit_actions]
                chosen_action = torch.tensor(np.random.choice(valid_indices), device=self.device, dtype=torch.long).view(1, 1)
                return chosen_action

            case "torch_version":
                sample = 1

                # Set the threshold for the epsilon-greedy policy.
                if train:
                    sample = random.random()
                eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
                self.steps_done += 1

                if sample > eps_threshold:
                    legit_actions = torch.tensor(legit_actions, device=self.device, dtype=torch.bool)
                    state = state.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        value_policy = self.policy_net(state)
                    # use torch utils to set to -10 the value of the actions that are not legit
                    filtered_policy = torch.where(legit_actions, value_policy, torch.tensor(-torch.inf, device=self.device, dtype=torch.float32))
                
                    chosen_action = filtered_policy.max(1)[1].view(1, 1)
                    self.steps_per_action[chosen_action.item()] += 1
                    self.action_dist.append(self.steps_per_action/np.sum(self.steps_per_action))

                    return chosen_action

                else:
                    # Random policy
                    legit_actions = np.array(legit_actions)
                    valid_indices = np.arange(4)[legit_actions]
                    chosen_action = torch.tensor(np.random.choice(valid_indices), device=self.device, dtype=torch.long).view(1, 1)
                    self.steps_per_action[chosen_action.item()] += 1
                    self.action_dist.append(self.steps_per_action/np.sum(self.steps_per_action))
                    return chosen_action
            
            case "entropy":
                # Set the threshold for the epsilon-greedy policy.

                self.steps_done += 1

                legit_actions = torch.tensor(legit_actions, device=self.device, dtype=torch.bool)
                state = state.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value_policy = self.policy_net(state)

                # compute the softmax of the policy
                beta = torch.log(torch.tensor(self.steps_per_action) + 1)*self.alpha
                # multiply the policy by the beta
                arg = value_policy * beta.to(self.device)
                # compute the softmax
                filtered_policy = torch.where(legit_actions, arg, torch.tensor(-torch.inf, device=self.device, dtype=torch.float32))
                softmax = torch.nn.functional.softmax(filtered_policy, dim=1)
                # if softmax is nan, set it to 0
                softmax[torch.isnan(softmax)] = 0
                # if softmax is inf, set it to 0
                softmax[torch.isinf(softmax)] = 0
                # if softmax is -inf, set it to 0
                softmax[torch.isneginf(softmax)] = 0
                # if softmax is negative, set it to 0
                softmax[softmax < 0] = 0
                # chose an action according to the softmax
                chosen_action = torch.multinomial(softmax, 1).item()
    
                self.steps_per_action[chosen_action] += 1
                self.action_dist.append(self.steps_per_action/np.sum(self.steps_per_action))
                return torch.tensor(chosen_action, device=self.device, dtype=torch.long).view(1, 1)


    def optimize_model(self):
        """Perform a step of the optimization of the policy network.
        The optimization is performed using a batch of transitions sampled from the replay buffer.
        The optimization is performed using the Huber loss, due to its robustness to outliers.
        For more details see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        
        Returns:
            _float_: The loss of the optimization.
        """

        if len(self.memory) < self.BATCH_SIZE:
            return 0

        # set the network in training mode
        self.policy_net.train()
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions)) # len(batch) = 4, batch[0] = state, batch[1] = action, batch[2] = next_state, batch[3] = reward, each of len = BATCH_SIZE
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).unsqueeze(1).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # loss = self.policy_net.calculate_loss(state_action_values, expected_state_action_values.unsqueeze(1), criterion)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        """Save the model to the specified path. It saves several parameters 
        of the agent, so that it can be loaded and the training can be resumed.
        The file is saved in the .pt format.
        
        Args:
            path (_str_): The file where to save the model.
        """
        
        l, _ = path.split("trained_architectures")
        
        path_net = path.replace(".pt", "_net.pt")
        nets = {"policy_net": self.policy_net,
                "target_net": self.target_net,
                "optimizer": self.optimizer.state_dict()}
        
        torch.save(nets, path_net)

        path_specs = path.replace("trained_architectures", "trained_architectures/specs")
        path_specs = path.replace(".pt", "_specs.pt")       
        
        specs = {"steps_done": self.steps_done,
                 "TAU": self.TAU,
                 "alpha": self.alpha,
                 "EPS_START": self.EPS_START,
                 "EPS_END": self.EPS_END,
                 "EPS_DECAY": self.EPS_DECAY,
                 "BATCH_SIZE": self.BATCH_SIZE,
                 "GAMMA": self.GAMMA,
                 "LR": self.LR,
                 "mean_board": self.mean_board,
                 "steps_per_action": self.steps_per_action}
        
        torch.save(specs, path_specs)

        attribute_to_folder = {
            "losses": "losses",
            "cumulative_reward": "cumulative_reward",
            "score": "score",
            "max_duration": "max_duration",
            "max_tile": "max_tile",
            "action_dist": "action_dist"
        }

        for attr, folder in attribute_to_folder.items():
            save_path = os.path.join("trained_architectures", folder, os.path.basename(path).replace(".pt", f"_{attr}.pt"))
            if l != '':
                save_path = os.path.join(l, save_path)
            if os.path.exists(save_path):
                torch.save(getattr(self, attr), save_path)
            else:
                warnings.warn(f"File not saved: {save_path}. The attribute {attr} will remain unsaved.")

    def load(self, path):
        """Load the model from the specified path. It loads the parameters of the agent
        to resume the training.
        
        Args:
            path (_str_): The file where to load the model from.
        """

        l, _ = path.split("trained_architectures")

        path_net = path.replace(".pt", "_net.pt")
        nets = torch.load(path_net, map_location=self.device)
        policy_net = nets["policy_net"]
        target_net = nets["target_net"]
        optimizer = nets["optimizer"]
        self.policy_net.load_state_dict(policy_net.state_dict())
        self.target_net.load_state_dict(target_net.state_dict())
        self.optimizer.load_state_dict(optimizer)

        path_specs = path.replace("trained_architectures", "trained_architectures/specs")
        path_specs = path_specs.replace(".pt", "_specs.pt")

        specs = torch.load(path_specs, map_location=self.device)
        self.steps_done = specs["steps_done"]
        self.TAU = specs["TAU"]
        self.EPS_START = specs["EPS_START"]
        self.EPS_END = specs["EPS_END"]
        self.EPS_DECAY = specs["EPS_DECAY"]
        self.BATCH_SIZE = specs["BATCH_SIZE"]
        self.GAMMA = specs["GAMMA"]
        self.LR = specs["LR"]
        self.steps_per_action = specs["steps_per_action"]
        self.mean_board = specs["mean_board"]
        self.alpha = specs["alpha"]

        attribute_to_folder = {
            "loss_history": "losses",
            "cumulative_reward": "cumulative_reward",
            "score": "score",
            "max_duration": "max_duration",
            "max_tile": "max_tile",
            "action_dist": "action_dist"
        }

        for attr, folder in attribute_to_folder.items():
            save_path = os.path.join("trained_architectures", folder, os.path.basename(path).replace(".pt", f"_{attr}.pt"))
            # if l is not empty, add it to the path
            if l != "":
                save_path = os.path.join(l, save_path)
        
            if os.path.exists(save_path):
                setattr(self, attr, torch.load(save_path, map_location=self.device))
            else:
                warnings.warn(f"File not found: {save_path}. The attribute {attr} will remain unchanged.")

        # set the network in evaluation mode
        self.policy_net.eval()
        self.target_net.eval()
 
    def _binary_tensor(self, state):
        """Convert the state of the game to a binary tensor.
        The tensor is of shape (12, 4, 4), where the first dimension
        represents the power of 2 of the tile, and the other two dimensions
        represent the position of the tile on the board.
        
        Args:
            state (_torch.tensor_): The state of the game, represented as a tensor of shape (4, 4).
            
        Returns:
            _torch.tensor_: The state of the game, represented as a binary tensor of shape (12, 4, 4).
        """

        tensor = torch.zeros(12, 4, 4)
        tensor[0, :, :] = torch.tensor(state == 0, dtype=torch.float32)
        for i in range(1, 12):
            tensor[i, :, :] = torch.tensor(state == 2**i, dtype=torch.float32)
        return tensor

    def fit(self, env, num_episodes=50, verbose=False):
        """Fit the agent to the environment. It performs num_episodes episodes of the game.
        At each step, it plays a game selecting an action with an epsilon-greedy policy,
        performing it and storing the transition in the replay buffer. Then it performs
        a step of the optimization of the policy network. 
        The function implements the reward shaping in order to (try to) speed up the training.
        
        Args:
            env (_Environment_): The environment to fit the agent to.
            num_episodes (int, optional): The number of episodes to play. Defaults to 50.
            verbose (bool, optional): (Debugging purposes) Whether to print the loss and the max tile at each episode. Defaults to False.
        """

        self.cumulative_reward = np.zeros(num_episodes)
        self.max_duration = np.zeros(num_episodes)
        self.loss_history = np.zeros(num_episodes)
        self.score = np.zeros(num_episodes)
        self.max_tile = np.zeros(num_episodes)
        
        for i_episode in tqdm.tqdm(range(num_episodes)):
    
            state, info = env.reset()
            state = self._binary_tensor(state)
            for t in count():
                # forecast with the model, it has batch norm and dropout layers, so we need to set it to eval mode
                self.policy_net.eval()
                action = self.select_action(state, env.get_legit_actions(), train=True, kind=self.kind_action).clone().detach()
                
                observation, reward, done, won, _ = env.step(action.item())

               # reward shaping
                num_512 = np.sum(observation == 512)
                num_1024 = np.sum(observation == 1024)
                num_2048 = np.sum(observation == 2048)
                if num_512 > 0:
                    reward += num_512
                if num_1024 > 0:
                    reward += 2.5 * num_1024
                if num_2048 > 0:
                    reward += 6 * num_2048
                # end of reward shaping

                self.cumulative_reward[i_episode] += reward - self.penalty_move  # penalize for each step
                reward = torch.tensor([reward - self.penalty_move], device=self.device)
            
                if done and not won:
                    next_state = None
                else:
                    next_state = self._binary_tensor(observation)

                self.memory.push(state, action, next_state, reward)

                state = next_state  # state = next_state does not work due to the way pytorch handles tensors
                # state = copy.deepcopy(next_state)
                self.loss_history[i_episode] += self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (1 - self.TAU) * target_net_state_dict[key] + self.TAU * policy_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.max_duration[i_episode] = t
                    self.loss_history[i_episode] /= t
                    break
            
            # get the max tile on the board
            self.max_tile[i_episode] = np.max(observation)
            self.score[i_episode] = env.score

            if verbose:
                print(f'Episode {i_episode} Loss {self.loss_history[i_episode]} Max tile: {self.max_tile[i_episode]} Max duration: {self.max_duration[i_episode]}')

            if i_episode % self.EPOCHS_CHECKPOINT == 0:
                # self.save(f"dqn_model_{i_episode}.pt") # used for testing only 
                print(f'Episode {i_episode} finished after {t+1} steps. Max tile: {self.max_tile[i_episode]}')


    def update_mean_board(self, board):
        """Update the mean board of the agent. It is used as metric in the test function.
        
        Args:
            board (_torch.tensor_): The board to update the mean board with.
        """
        # set the board to 1 if it is 0
        board[board == 0] = 1
        board = np.log2(board)

        self.mean_board = (self.mean_board * (self.steps_done-1) + board) / (self.steps_done)

    def test(self, env, num_episodes=100):
        """Test the agent on the environment. It performs num_episodes episodes of the game
        and returns the achieved score for each episode.

        Args:
            env (_Environment_): The environment to test the agent on.
            num_episodes (int, optional): How many episodes to play. Defaults to 100.
            
        Returns:
            _np.array_: The achieved score for each episode.
            _np.array_: The achieved duration for each episode.
        """

        self.policy_net.eval()
        # make a list to store each reward per episode
        scores = np.zeros(num_episodes)
        durations = np.zeros(num_episodes)

        for i in tqdm.tqdm(range(num_episodes)):
            state, _ = env.reset()
            state = self._binary_tensor(state)
            for t in count():
                done, _ = env._is_game_over()
                if done:
                    break

                action = self.select_action(state, env.get_legit_actions(), kind=self.kind_action).clone().detach() # gather expected int64
                observation, _, done, won, _ = env.step(action.item())

                self.update_mean_board(copy.deepcopy(observation))
                if done and not won:
                    next_state = None
                else:
                    next_state = self._binary_tensor(observation)
                state = next_state
            
            durations[i] = t
            scores[i] = env.score

        return scores, durations
    
    def __repr__(self):
        """Return a string representation of the agent. (Debugging purposes)

        Returns:
            _str_: A string representation of the agent.
        """

        return f"Model: {self.__class__.__name__}\n" + \
                f"Steps done {self.steps_done}\n" + \
                f"Memory size {len(self.memory)}\n" \
                f"Name {self.name}" 
