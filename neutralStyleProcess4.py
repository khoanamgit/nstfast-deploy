import imutils
import cv2
import os
import numpy as np
import torch
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from PIL import Image
import re

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

mean = np.array([0.4764, 0.4504, 0.4100])
std = np.array([0.2707, 0.2657, 0.2808])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors


def deprocess(image_tensor):
    """ Denormalizes and rescales image tensor """
    img = denormalize(image_tensor)
    img *= 255
    image_np = torch.clamp(img, 0, 255).numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np

def save_image(filename, data):
    img = deprocess(data)
    img = Image.fromarray(img)
    img.save(filename)


def neuralStyleTransfer(directoryName, filename, selected_style):
    # Neural style transfer codes adapted from pyimageSearch

    # Load image
    content_image = utils.load_image(directoryName + filename)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    # Load model
    with torch.no_grad():
        target = os.path.join(APP_ROOT, 'saved_models/')
        style_model = TransformerNet()
        state_dict = torch.load(target + selected_style)

        style_model.load_state_dict(state_dict['state_dict'])
        style_model.eval()
        output = style_model(content_image)

    #save Image
    filename, file_extension = os.path.splitext(filename)
    stylename, _ = os.path.splitext(selected_style)
    print(filename)
    newFileName = stylename + '_' + filename + file_extension
    # cv2.imwrite(directoryName + newFileName, img)
    save_image(directoryName + newFileName, output[0])
    print(newFileName)
    print(directoryName)

    return newFileName


if __name__ == '__main__':
    # swapT with output/=255 doesnt work
    # swapT without output/=255 doesnt look nice
    # swapF with output/=255 doesnt work
    # swapF without output/=255 looks best
    neuralStyleTransfer('images/', 'amber.jpg', 'candy.pth')


