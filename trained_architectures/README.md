### Loading and Saving the trained architectures

In this folder, you can find some of the architectures we trained in order to allow you to use them in your DQN algorithms.

The files .pt can be loaded using the torch.load() function.

The 4 models you can load are in this level of the folder: 3 architectures contain the number 300, which means that they have the same architecture, and they have been trained with 300 episodes. The difference between them is the technique used for the epsilon-greedy policy; the other architecture has kind="entropy", and we trained it with 10000 episodes since its performances were better in the early stages of the training.

Each of the subfolders contains different metrics you can use to compare and better understand the performances of the models, during the loading, the metrics will be assigned to the attributes of the agent.


To save or load the models, and the metrics, you can use agent.load() and agent.save() functions, which are implemented in the DQN class, the paths available for different models are:

**trained_architectures/convdqn_agent_long_train.pt** -> for the model trained with 10000 episodes

**trained_architectures/convdqn_entropy_300.pt** -> for the model trained with 300 episodes and kind="entropy"

**trained_architectures/convdqn_torch_300.pt** -> for the model trained with 300 episodes and kind="torch_version"

**trained_architectures/convdqn_torch_lairobbins.pt** -> for the model trained with 300 episodes and kind="lai_robbins"

You can check the different metrics available by looking in the respective folders, they can be useful because in some cases to obtain the metrics is necessary to run the model for a long time, and it can be expensive to do it.
