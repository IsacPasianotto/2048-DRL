import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    """_This architecture was the mainly used for training the agent, it has 1 conv 3D layer, 
    and 3 fully connected layers, we used batch normalization for the first conv layer and the first fully connected;
    the activation function is the ReLU, and we applied L2 regularization_

    Args:
        _out_channels (int, optional): [description]. Defaults to 3.
        n_fc (int, optional): [description]. Defaults to 128.
        l2_regularization (float, optional): [description]. Defaults to 0.001.
    """

    def __init__(self, out_channels=3, n_fc=128, l2_regularization=0.001):
        super(ConvolutionalNetwork, self).__init__()
        self.out_channels = out_channels
        self.l2_regularization = l2_regularization
        self.conv = nn.Conv3d(in_channels=1, out_channels=self.out_channels, kernel_size=(2, 4, 4), stride=1, padding=0)
        self.bnc = nn.BatchNorm3d(self.out_channels)
        self.fc = nn.Linear(11*self.out_channels, out_features=n_fc)
        self.bn1 = nn.BatchNorm1d(n_fc)
        self.fc2 = nn.Linear(n_fc, n_fc)
        self.fc3 = nn.Linear(n_fc, out_features=4)
        # if we want to use the l2 regularization

        if self.l2_regularization > 1e-8:
            self.fc.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc.weight)
            self.fc2.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc2.weight)
            self.fc3.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.conv(x))
        # set the right shape for batch norm 3D
        x = x.view(-1, self.out_channels, 11, 1, 1)
        x = self.bnc(x)
        x = x.view(-1, 11*self.out_channels)
        x = F.relu(self.fc(x))
        x = F.relu(self.bn1(self.fc2(x)))
        return self.fc3(x)
        

class BigConvolutionalNetwork(nn.Module):
    """_This architecture was the mainly used for training the agent, it has 1 conv 3D layer, 
    and 3 fully connected layers, we used batch normalization for the first conv layer and the first fully connected;
    the activation function is the ReLU, and we applied L2 regularization_

    Args:
        _out_channels (int, optional): [description]. Defaults to 3.
        n_fc (int, optional): [description]. Defaults to 128.
        l2_regularization (float, optional): [description]. Defaults to 0.001.
    """

    def __init__(self, out_channels=4, n_fc=128, l2_regularization=0.001):
        """ _The constructor of the class ConvolutionalNetwork_

        Args:
            out_channels (int, optional): Number of output channels of the convolutional layer. Defaults to 3.
            n_fc (int, optional): Number of neurons in the fully connected layer. Defaults to 128.
            l2_regularization (float, optional): L2 regularization parameter. Defaults to 0.001.
        """

        super(BigConvolutionalNetwork, self).__init__()
        self.out_channels = out_channels
        self.l2_regularization = l2_regularization
        self.conv = nn.Conv3d(in_channels=1, out_channels=self.out_channels, kernel_size=(2, 4, 4), stride=1, padding=0)
        self.bnc = nn.BatchNorm3d(self.out_channels)
        self.fc = nn.Linear(11*self.out_channels, out_features=2*n_fc)
        self.bn1 = nn.BatchNorm1d(2*n_fc)
        self.fc2 = nn.Linear(2*n_fc, 4*n_fc)
        self.bn2 = nn.BatchNorm1d(4*n_fc)
        self.fc3 = nn.Linear(4*n_fc, n_fc)
        self.bn3 = nn.BatchNorm1d(n_fc)
        self.fc4 = nn.Linear(n_fc, out_features=4)
        # if we want to use the l2 regularization

        if self.l2_regularization > 1e-8:
            self.fc.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc.weight)
            self.fc2.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc2.weight)
            self.fc3.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc3.weight)
            self.fc4.weight.register_hook(lambda grad: grad + self.l2_regularization * self.fc4.weight)

    def forward(self, x):
        """Performs a forward pass through the network and returns the output of the last layer.

        Args:
            x (_torch.Tensor_): Input tensor of shape (batch_size, 1, 12, 4, 4)

        Returns:
            _torch.Tensor_: Output tensor of shape (batch_size, 4).
        """

        x = F.relu(self.conv(x))
        # set the right shape for batch norm 3D
        x = x.view(-1, self.out_channels, 11, 1, 1)
        x = self.bnc(x)
        x = x.view(-1, 11*self.out_channels)
        x = F.relu(self.bn1(self.fc(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)
        
class fc_QNet(nn.Module):
    """Network with fully connected layers.

    Args:
        _
        n_fc (int): Number of neurons in the fully connected layer
        l2_regularization (float, optional): L2 regularization parameter. Defaults to 0.001.
    """

    def __init__(self, n_observations=12*4*4, n_actions=4, n_layers=3, n_neurons=128, l2_regularization=0.001):
        """_The constructor of the class Small_QNet_

        Args:
            n_observations (_int_): _Number of observations of the environment_
            n_actions (_int_): _Number of actions of the environment_
            n_neurons (_int_): _Number of neurons in the fully connected layer_
            l2_regularization (float, optional): L2 regularization parameter. Defaults to 0.001.
        """

        super(fc_QNet, self).__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.layer1 = nn.Linear(n_observations, n_neurons)
        for i in range(1,n_layers-1):
            setattr(self, f"layer{i+1}", nn.Linear(n_neurons, n_neurons))
        setattr(self, f"layer{n_layers}", nn.Linear(n_neurons, n_actions))
        self.l2_regularization = l2_regularization

        if self.l2_regularization > 1e-8:
            for i in range(n_layers):
                getattr(self, f"layer{i+1}").weight.register_hook(lambda grad: grad + self.l2_regularization * getattr(self, f"layer{i+1}").weight)
    
    def forward(self, x):
        """Performs a forward pass through the network and returns the output of the last layer.
    
        Args:
            x (_torch.Tensor_): Input tensor of shape (batch_size, n_observations).

        Returns:
            _torch.Tensor_: Output tensor of shape (batch_size, n_actions).
        """

        x = x.flatten().view(x.size(0), -1)
        for i in range(self.n_layers):
            x = F.relu(getattr(self, f"layer{i+1}")(x))
        return x

