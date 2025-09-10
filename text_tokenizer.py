# text_tokenizer.py
import json
import torch

class TextTokenizer:
    def __init__(self, vocab_path):
        # Load your existing vocabulary - READ ONLY
        with open(vocab_path, 'r') as f:
            self.stoi = json.load(f)  # Use your existing mapping
        
        # Don't add any new tokens - use only what exists
        print(f"Loaded vocabulary with {len(self.stoi)} tokens")
    
    def __len__(self):
        """Return the size of the vocabulary"""
        return len(self.stoi)
    
    def prepare_input(self, text, image_embeddings):
        """
        Prepare input using ONLY existing vocabulary
        Returns: text_tokens, image_positions, image_embeddings
        """
        words = text.split()
        
        image_positions = []
        text_words = []
        current_position = 0
        
        for i, word in enumerate(words):
            if word in self.stoi:  # Only use words that exist in vocabulary
                if word == "<image>":
                    image_positions.append(current_position)  # Position where image should be inserted
                    current_position += 1  # Image takes up one position
                else:
                    text_words.append(word)
                    current_position += 1
        
        # Convert to token IDs using ONLY existing vocabulary
        text_token_ids = [self.stoi[word] for word in text_words]
        
        return text_token_ids, image_positions, image_embeddings
    
    def decode(self, token_ids):
        """Convert token IDs back to text using existing vocabulary"""
        itos = {v: k for k, v in self.stoi.items()}  # Create reverse mapping
        return " ".join([itos.get(token_id, "<?>") for token_id in token_ids])