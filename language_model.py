# language_model.py
import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=2048, num_layers=6, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, image_embeddings=None, image_positions=None):
        text_embeds = self.token_embedding(input_ids)
        if image_embeddings is not None and image_positions is not None:
            combined_embeds = self._insert_image_embeddings(
                text_embeds, image_embeddings, image_positions
            )
        else:
            combined_embeds = text_embeds
        x = combined_embeds
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.output_layer(x)
        return logits
    
    def _insert_image_embeddings(self, text_embeds, image_embeddings, image_positions):
        """
        Insert image embeddings at specified positions in the text sequence.
        """
        batch_size, text_seq_len, hidden_size = text_embeds.shape
        num_image_tokens = image_embeddings.shape[1]  # Number of image patches
        
        # Create the combined sequence
        # For each image position, we insert all image tokens
        total_image_tokens = num_image_tokens * len(image_positions)
        combined_seq_len = text_seq_len + total_image_tokens
        
        combined_embeds = torch.zeros(batch_size, combined_seq_len, hidden_size, 
                                     device=text_embeds.device, dtype=text_embeds.dtype)
        
        combined_idx = 0
        text_idx = 0
        
        # For each position in the original text sequence
        for pos in range(text_seq_len + len(image_positions)):
            if pos in image_positions:
                # Insert all image tokens at this position
                combined_embeds[:, combined_idx:combined_idx + num_image_tokens, :] = image_embeddings
                combined_idx += num_image_tokens
            else:
                # Insert text token (but make sure we don't exceed text sequence length)
                if text_idx < text_seq_len:
                    combined_embeds[:, combined_idx, :] = text_embeds[:, text_idx, :]
                    combined_idx += 1
                    text_idx += 1
        
        return combined_embeds

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x
