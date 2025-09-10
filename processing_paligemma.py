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
    
    def process(self, images):
        """Process a single image or list of images"""
        if isinstance(images, list):
            # Process multiple images
            processed_images = []
            for image in images:
                processed_images.append(self.transform(image))
            return torch.stack(processed_images)
        else:
            # Process single image
            return self.transform(images).unsqueeze(0)