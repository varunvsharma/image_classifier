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

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model and returns a Numpy array.
        
        Keyword arguments:
        image -- PIL image
    '''
    # Get image attributes and compute aspect ratio
    width, height = image.size
    aspect_ratio = (width / height)
    
    # Resize images
    # Make the shortest size 256 and maintain the aspect ratio
    if width < height:
        new_width = 256
        new_height = int(np.round(new_width/aspect_ratio))
        image = image.resize((new_width, new_height))
    elif width > height:
        new_height = 256
        new_width = int(np.round(new_height*aspect_ratio))
        image = image.resize((new_width, new_height))
        
    # Centre Crop image
    centre_crop_size = 224
    w_margin = (new_width - centre_crop_size)/2
    h_margin = (new_height - centre_crop_size)/2
    left = w_margin
    right = w_margin + centre_crop_size
    upper = h_margin
    lower = h_margin + centre_crop_size
    image = image.crop(box=(left, upper, right, lower))
    
    
    # Convert image into a numpy array
    np_image = np.array(image)
    
    # Transpose array and make colour channels the first dimension
    np_image = np_image.transpose(2, 0, 1)
    
    # Scale the values in each colour channel to a value between 0 and 1
    np_image = np_image / 255
    
    # Normalize the image
    means = np.array([0.485, 0.456, 0.406])[:, None, None]
    stds = np.array([0.229, 0.224, 0.225])[:, None, None]
    np_image = (np_image - means) / stds
    
    return np_image    
