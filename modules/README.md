In this folder, you can find all the main modules we have written from scratch for this project!

`agents.py`: in this file, you can find the implementation of the agents we used for the project. We have implemented the following agents:
- RandomAgent: this agent is used to test the environment and the game. It takes random actions at each step.
- ConvDQN: this agent is the one we used for the final submission. It supports both fully connected, convolutional neural networks with a DQN as a policy. It is trained with a replay buffer and a target network.

`architectures.py`: in this file, you can find the implementation for the neural networks we used for the project (and also possible nets you may want to use). We have implemented the following neural networks:
- fc_QNET (fully connected Q network): this is a simple fully connected neural network with N hidden layers. It takes as input the state and outputs the Q values for each action; you can also specify the size of the net and the L2 regularization parameter
- BigConvolutionalNetwork is a convolutional neural network with BatchNorm, 1 convolutional layer, and 4 fully connected layers. It takes as input the state and outputs the Q values for each action; you can also specify the number of layers and the L2 regularization parameter

`envs.py`: in this file, you can find the implementation of the environment we used for the project. We have implemented the following environment:
- Game2048Env: in this environment, you can simulate the dynamic of the game with its rules and constraints

`gifgen.py`: in this file, you can find the implementation of the gif generator we used for the project.
To generate a gif, you can import the module in a Jupyter Notebook, or you can write in the terminal

```bash
python gifgen.py --version <version> 
```
where version is the version can be "entropy", "random", "lai_robbins", or "torch_version", default is "entropy".

`stats.py`: a utility file to compute some statistics and plot of the results.
