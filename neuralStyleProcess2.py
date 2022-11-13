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


def neuralStyleTransfer(directoryName, filename, selected_style):
    # Neural style transfer codes adapted from pyimageSearch

    # Load image
    content_image = utils.load_image(directoryName + filename)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    # Load model
    with torch.no_grad():
        target = os.path.join(APP_ROOT, 'saved_models/')
        style_model = TransformerNet()
        state_dict = torch.load(target + selected_style)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]

        style_model.load_state_dict(state_dict)
        style_model.eval()
        output = style_model(content_image)

    #save Image
    filename, file_extension = os.path.splitext(filename)
    stylename, _ = os.path.splitext(selected_style)
    print(filename)
    newFileName = stylename + '_' + filename + file_extension
    # cv2.imwrite(directoryName + newFileName, img)
    utils.save_image(directoryName + newFileName, output[0])
    print(newFileName)
    print(directoryName)

    return newFileName


if __name__ == '__main__':
    # swapT with output/=255 doesnt work
    # swapT without output/=255 doesnt look nice
    # swapF with output/=255 doesnt work
    # swapF without output/=255 looks best
    neuralStyleTransfer('images/', 'amber.jpg', 'candy.pth')


