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

def train(model, train_dataset, valid_dataset, epochs, learning_rate, gpu, save_dir):
    # Set default device to CPU
    device = 'cpu'

    # If the gpu flag has been used, check if Cuda is available
    # If Cuda isn't available train on CPU
    if gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        if device == 'cuda':
            print('Cuda is available! Training on GPU')
        else:
            print('Cuda is not avaiable, Training on CPU')

    # Move model to appropriate device
    model = model.to(device)

    # Set loss function
    criterion = nn.NLLLoss()

    # Set optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Initialize minimum validation loss
    valid_loss_min = np.Inf

    # Set print steps value
    print_every = 5

    # Use training and validation datasets to create data loaders
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    for e in range(epochs):

        # Initialize running loss and steps
        running_loss = 0
        steps = 0

        # Training Pass
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            
            if steps % print_every == 0:
                
                # Turn off gradients
                with torch.no_grad():
                    
                    #Set model to evaluation mode
                    model.eval()
                    
                    # Initialize validation loss and accuracy score
                    valid_loss = 0
                    accuracy = 0
                    
                    #Validation pass
                    for images, labels in validloader:
                        
                        #Move validation tensors to appropriate device
                        images, labels = images.to(device), labels.to(device)
                        
                        log_ps = model(images)
                        ps = torch.exp(log_ps)
                        valid_loss += criterion(log_ps, labels)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                        
                    print(
                        f'Epoch: {e+1}/{epochs} |',
                        f'Training Loss: {running_loss/print_every:.3f} |',
                        f'Validation Loss: {valid_loss/len(validloader):.3f} |',
                        f'Accuracy: {accuracy/len(validloader)*100:.3f}%'
                    )
                    
                    running_loss = 0
                    model.train()

                    # Check if validation loss is less than the minimum validation loss
                    if valid_loss < valid_loss_min:
                        valid_loss_min = valid_loss

                        # Display change in minimum validation loss
                        # Save checkpoint
                        print(f'New minimum validation loss {valid_loss_min} --> {valid_loss} | Saving Checkpoint')
                        checkpoint = {
                            'input_size': model.fc.hidden[0].in_features,
                            'output_size': model.fc.output.out_features,
                            'hidden_layers': [layer.out_features for layer in model.fc.hidden],
                            'drop_p': 0.2,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': e,
                            'learning_rate': learning_rate,
                            'class_to_idx': train_dataset.class_to_idx
                        }
                        torch.save(checkpoint, save_dir)


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
    train_dataset, valid_dataset, test_dataset = process_data.load_data(args.data_directory)
    output_size = len(train_dataset.classes)
    model = custom_model.build_model(
        output_size=output_size,
        hidden_units=args.hidden_units,
        arch=args.arch
    )
    train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gpu=args.gpu,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()