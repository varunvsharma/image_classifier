# Import modules
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models
from torch.utils.data import DataLoader

class classifier(nn.Module):
    """Fully connected neural network classifier."""
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        """
        Initialize classifier attributes.

        Keyword arguments:
        input_size -- number of input nodes
        output_size -- number of output nodes
        hidden_layers -- list of nodes related to each hidden layer
        drop_p -- dropout probability
        """
        # Initialize parent class
        super().__init__()

        # Add Hidden Layers
        for ii, layer in enumerate(hidden_layers):
            if ii == 0:
                self.hidden = nn.ModuleList([nn.Linear(input_size, layer)])
            else:
                self.hidden.extend([nn.Linear(hidden_layers[ii-1], layer)])
        
        # Add Output Layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # Add Dropout Layer
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """
        Forward pass definition.

        Keyword arguments:
        x -- input tensor
        """
        # Pass input tensor through hidden layers and ReLU activations
        # Apply dropout     
        for layer in self.hidden:
            x = self.dropout(F.relu(layer(x)))
        
        # Pass final hidden layer to output layer and log_softmax activation
        x = F.log_softmax(self.output(x), dim=1)
        return x
