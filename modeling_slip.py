import torch
import torch.nn as nn

class simplevisionconfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_img_tokens: int = None,
        IMAGE_MEAN = [0.5, 0.5, 0.5],
        IMAGE_STD = [0.5, 0.5, 0.5],
        text_dim=2048,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_img_tokens = num_img_tokens
        self.IMAGE_MEAN = IMAGE_MEAN
        self.IMAGE_STD = IMAGE_STD
        self.text_dim=2048

