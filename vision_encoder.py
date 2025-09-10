import torch 
import torch.nn as nn

from modeling_slip import simplevisionconfig

class Vision_encoder(nn.Module):
    def __init__(self, config: simplevisionconfig):
        super().__init__()
        self.config = config

        # Turn the input image into smaller patches using a convolution
        self.patch_embed = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        #cls token 
        self.cls_token=nn.Parameter(torch.zeros(1,1,config.hidden_size))

        # psotion encodeing and the cls tocken
        self.num_patches=(config.image_size // config.patch_size)**2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches,config.hidden_size))
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)



    def forward(self,pixel_values):
        patches = self.patch_embed(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)

        batch_size = patches.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_size]
        hidden_states = torch.cat([cls_tokens, patches], dim=1)

        hidden_states = hidden_states + self.pos_embed

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        # Pre-norm: apply layer norm before attention
        normed_x = self.norm1(x)
        attn_output, _ = self.attention(normed_x, normed_x, normed_x)
        x = x + attn_output
        
        # Pre-norm: apply layer norm before MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output
        
        return x
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x