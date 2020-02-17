# Import modules
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models

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

def build_model(output_size, hidden_units=[1024, 512, 256], arch='resnet50'):
    """Build full model from pretrained network.
    
    Keyword arguments:
    arch -- pretrained network architecture
    """
    # Download pretrained model
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set up a dictionary that contains the names of pretrained network groups
    # and the names of their classifiers
    clf_dict = {
        'resnet': 'fc',
        'vgg': 'classifier',
        'alexnet': 'classifier',
        'squeeznet': 'classifier',
        'densenet': 'classifier',
        'inception': 'fc',
        'googlenet': 'fc',
        'shufflenet': 'fc',
        'mobilenet': 'fc',
        'resnext': 'fc',
        'wide_resnet': 'fc',
        'mnasnet': 'classifier'
        }

    # Get the fully connected classifier's input features
    for network, clf in clf_dict.items():
        if network in arch:
            clf_name = clf
            clf_net = getattr(model, clf)
            if type(clf_net).__name__ == 'Sequential':
                input_size = clf_net[0].in_features
            elif type(clf_net).__name__ == 'Linear':
                input_size = clf_net.in_features

    # Build new classifier
    fc_clf = classifier(
        input_size=input_size, \
        output_size=output_size, \
        hidden_layers=hidden_units)

    # Replace the pretrained network's classifier with new classifier
    setattr(model, clf_name, fc_clf)
    
    # Return the modified model
    return model

def load_checkpoint(checkpoint):
    """Rebuild model from checkpoint.

    Keyword arguments:
    checkpoint -- path to the checkpoint file
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint)
    
    # Rebuild model
    model = custom_model.build_model(
        output_size = checkpoint['output_size']
        hidden_units = checkpoint['hidden_layers']
        arch = checkpoint['arch']
    )
    
    # Load model's state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Attach class to index mapping in checkpoint to model
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model