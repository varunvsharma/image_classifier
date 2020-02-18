# Image Classifier

## Purpose

Build an AI application that can classify images using Deep Learning.

## Scope

This repo contains an ipython notebook with python code for an image classifier and a command line application that builds/trains a new neural network (using a pretrained network as a feature extractor). The application also includes a module for inference. Built with PyTorch.

## Install

Minimum requirements:

* Python 3.x
* Software to run and execute an iPython notebook
* NumPy
* PyTorch (including torchvision)
* matplotlib

## Run

### Notebook

To the run the notebook, navigate to the top level directory of this repository and run one of the following commands:

```bash
ipython notebook Image\ Classifier\ Project.ipynb
```

or 

```bash
jupter notebook Image\ Classifier\ Project.ipynb
```

### Building and Training

To use the application for building and training a neural network, navigate to the top level directory of this repository and run the following command:

```bash
python train.py data_directory [-h] [--save_dir] [--arch] [--learning_rate] [--hidden_units] [--epochs] [--gpu]
```

**Positional Arguments:**

* *data_directory* - Directory that contains the images for training, validation and testing (required)

**Optional Arguments:**

* *-h, --help* - Show help message and exit
* *--save_dir* - Set directory for saving model checkpoints
* *--arch* - Choose pretrained pytorch neural network architecture, default (resnet50)
* *--learning_rate* - Set learning rate for training, default (0.003)
* *--hidden_units* - Sequentially list the number of nodes for each hidden layer, default (1024 512 256)
* *--epochs* - Set the number of training epochs, default (10)
* *--gpu* - Use flag to train on GPU, default (False)

### Inference

To use the application for inference, navigate to the top level of this repository and run the following command:

```bash
python predict.py image_path model_checkpoint [-h] [--top_k] [--category_names] [--gpu]
```

**Positional Arguments:**

* *image_path* - Enter the path to the image file (required)
* *model_checkpoint* - Enter the path to the model checkpoint (required)

**Optional Arguments:**

* *-h, --help* - Show this help message and exit
* *--top_k* - Set value for the number of most likely classes to be returned
* *--category_names* - Enter the path to the category to name map (JSON)
* *--gpu* - Use flag to enable gpu for inference