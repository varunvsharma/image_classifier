# Import modules
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from PIL import Image

def load_data(data_dir):
    """Load the training data and return datasets for training, validation and testing.

    Keyword arguments:
    data_dir -- Directory containing training data.
        Training data directory must be split into train, valid and test directories.
        For each class the directories should be organised as follows.
        data_dir/train/{class}/{image_files}
        data_dir/valid/{class}/{image_files}
        data_dir/test/{class}/{image_files}
    """
    # Get training, validation and testing directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define image transformations for training
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    # Define image transformations for validation
    valid_transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    # Define image transformations for testing
    test_transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    return train_dataset, valid_dataset, test_dataset
