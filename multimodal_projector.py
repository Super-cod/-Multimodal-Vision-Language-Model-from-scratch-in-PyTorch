# multimodal_projector.py
import torch.nn as nn
from modeling_slip import simplevisionconfig

class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        # Simple linear projection from vision to text dimension
        self.linear = nn.Linear(vision_dim, text_dim)
        self.activation = nn.GELU()  # Optional activation
    
    def forward(self, image_embeddings):

        projected = self.linear(image_embeddings)
        projected = self.activation(projected)
        return projected
    