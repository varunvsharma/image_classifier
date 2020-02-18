import numpy as np
import torch
import custom_model
import process_data
import argparse
from PIL import Image
import json

def predict(image_path, model_checkpoint, top_k, category_name, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Keyword arguments:
    image_path -- file path of the image
    model_checkpoint -- pytorch model checkpoint
    top_k -- number of most likely classes
    category_name -- class to name map
    gpu -- flag to use gpu for inference
    ''' 
    # Open image using PIL
    image = Image.open(image_path)
    
    # Preprocess image
    image = process_data.process_image(image)
    image = torch.Tensor(image)
    image = image.view(1, 3, 224, 224)
    image = image
    
    # Rebuild model from checkpoint
    model = custom_model.load_checkpoint(model_checkpoint)
    model = model

    # If the gpu flag has been used, check if Cuda is available
    # If Cuda isn't available run inference on CPU
    if gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        if device == 'cuda':
            print('Cuda is available! Running inference on GPU')
        else:
            print('Cuda is not avaiable, running inference on CPU')

        image = image.to(device)
        model = model.to(device)
    
    # Forward pass
    log_ps = model(image)
    ps = torch.exp(log_ps)
    top_ps, top_classes = ps.topk(top_k, dim=1)
    
    # Convert top_ps to list
    top_ps = list(top_ps.view(-1).detach().numpy())
    
    # Get index to class mapping
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[c.item()] for c in top_classes[0]]

    if category_name:
        with open(category_name, 'r') as f:
            cat_to_name = json.load(f)

        top_names = [cat_to_name[c] for c in top_classes]
        y = top_names

        return top_ps, top_names

    return top_ps, top_classes

def main():

    parser = argparse.ArgumentParser(
        description="Predict the class of an image" 
    )
    parser.add_argument(
        'image_path',
        action='store',
        help='enter the path to the image file'
    )

    parser.add_argument(
        'model_checkpoint',
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
        default=False,
        help='use flag to enable gpu for inference'
    )

    args = parser.parse_args()
    top_ps, top_classes = predict(
        image_path=args.image_path,
        model_checkpoint=args.model_checkpoint,
        top_k=args.top_k,
        category_name=args.category_names,
        gpu=args.gpu  
    )
    print('Top predictions:')
    for i in range(len(top_ps)):
        print(f'Class: {top_classes[i]} | Probability: {top_ps[i]:.3f}')

if __name__ == '__main__':
    main()