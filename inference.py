# inference.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class TextGenerator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
    def generate(self, image, prompt: str, max_length: int = 50, 
                temperature: float = 1.0, top_p: float = 0.9):
        """
        Generate text from image and prompt
        """
        # Process image
        pixel_values = self.model.image_processor.process(image).to(self.device)
        image_embeddings = self.model.vision_encoder(pixel_values)
        projected_embeddings = self.model.projector(image_embeddings)
        
        # Tokenize prompt
        text_tokens, image_positions, _ = self.tokenizer.prepare_input(
            prompt, projected_embeddings
        )
        
        # Convert to tensor
        input_ids = torch.tensor([text_tokens], device=self.device)
        
        # Generate text autoregressively
        generated_ids = self._generate_autoregressive(
            input_ids, projected_embeddings, image_positions, 
            max_length, temperature, top_p
        )
        
        # Decode to text
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text
    
    def _generate_autoregressive(self, input_ids, image_embeddings, image_positions,
                               max_length, temperature, top_p):
        """
        Autoregressive generation using greedy sampling
        """
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                logits = self.model.language_model(
                    generated_ids, image_embeddings, image_positions
                )
            
            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Get next token (greedy for now)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.stoi.get("<EOS>", 2):
                break
        
        return generated_ids
    
    def _top_p_filtering(self, logits, top_p: float = 0.9):
        """
        Top-p (nucleus) sampling filter
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('Inf')
        
        return logits
    
    def greedy_search(self, image, prompt, max_length=50):
        """Greedy decoding (always choose most likely token)"""
        return self.generate(image, prompt, max_length, temperature=1.0, top_p=1.0)
    
    def beam_search(self, image, prompt, beam_width=3, max_length=50):
        """Beam search decoding (not implemented yet)"""
        # Placeholder for beam search
        return self.generate(image, prompt, max_length)