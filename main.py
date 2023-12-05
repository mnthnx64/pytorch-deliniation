"""Inference script for the UNetResNet50_X model."""
from models.unet_models import UNetResNet50_9, UNetResNet50_3
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description='Inference script for the UNetResNet50_X model.')
parser.add_argument('--model', type=str, default='3', help='Number of season\'s images to use (1 or 3)')
parser.add_argument('--model_path', type=str, default='checkpoints/UNetResNet50_9.pt', help='Path to model weights')
parser.add_argument('--input_path', type=str, default='test_data/', help='Path to input images')

args = parser.parse_args()

# Load model
if args.model == '3':
    model = UNetResNet50_9()
elif args.model == '1':
    model = UNetResNet50_3()
else:
    raise ValueError('Model must be 1 or 3')

try:
    model.load_state_dict(torch.load(args.model_path))
except FileNotFoundError:
    print('Model not found. Please check the path to the model weights.')
    # exit(1)

# Load input images
images = np.array([])
for filename in os.listdir(args.input_path):
    if filename.endswith('.jpeg'):
        img = Image.open(os.path.join(args.input_path, filename))
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        if images.size == 0:
            images = img
        else:
            images = np.concatenate((images, img), axis=0)

# Convert to tensor
input_tensor = torch.from_numpy(images).float().unsqueeze(0)

# Perform inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)
    output = output.squeeze(0).squeeze(0).numpy()

# Plot results
# plt.imshow(output)
# plt.show()



