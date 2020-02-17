import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models
import custom_model
import process_data
import argparse

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
     
def main():

    parser = argparse.ArgumentParser(
        description="Predict the class of an image" 
    )
    parser.add_argument(
        'path_to_image',
        action='store',
        help='enter the path to the image file'
    )

    parser.add_argument(
        'checkpoint',
        action='store',
        help='enter the path to the model checkpoint'
    )

    parser.add_argument(
        '--top_k',
        action='store', 
        type=int,
        default=5,
        help='set value for the number of most likely classes to be returned (default 5)'
    )
    parser.add_argument(
        '--category_names',
        action='store',
        help='enter the path to the category to name map'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='use flag to enable gpu for inference'
    )

if __name__ == '__main__':
    main()