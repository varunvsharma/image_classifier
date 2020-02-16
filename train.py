# Import modules
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models
from torch.utils.data import DataLoader
import custom_model
import process_data
import argparse


def main():

    parser = argparse.ArgumentParser(
        description="Train a neural network to classify images" 
    )
    parser.add_argument(
        'data_directory',
        action='store',
        help='directory that contains the images for training, testing and validation'
    )
    parser.add_argument(
        '--save_dir',
        action='store', 
        help='set directory for saving model checkpoints'
    )
    parser.add_argument(
        '--arch',
        action='store',
        default='resnet50',
        help='choose pretrained pytorch neural network architecture, default (resnet50)'
    )
    parser.add_argument(
        '--learning_rate',
        action='store',
        default='0.003',
        type=float,
        help='set learning rate for training, default (0.003)'
    )
    parser.add_argument(
        '--hidden_units',
        action='store',
        nargs='+',
        type=int,
        default=[1024, 512, 256],
        help='sequentially list the number of nodes for each hidden layer, default (1024 512 256)'
    )
    parser.add_argument(
        '--epochs',
        action='store',
        type=int,
        default=10,
        help='set the number of training epochs, default (10)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=False,
        help='use flag to train on GPU, default (False)'
    )

    args = parser.parse_args()

    return None

if __name__ == '__main__':
    main()