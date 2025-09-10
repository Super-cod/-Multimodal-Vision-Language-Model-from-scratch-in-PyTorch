import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from PIL import Image
from modeling_slip import simplevisionconfig

class ImageProcessing:
    def __init__(self, config:simplevisionconfig):
        self.transform = T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ])
    
    def process(self, image):
        return self.transform(image).unsqueeze(0)