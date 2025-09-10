# multimodal_model.py
import torch
import torch.nn as nn
from processing_paligemma import ImageProcessing
from vision_encoder import Vision_encoder
from multimodal_projector import MultimodalProjector
from text_tokenizer import TextTokenizer
from language_model import LanguageModel

class MultimodalModel(nn.Module):
    def __init__(self, config, vocab_path):
        super().__init__()
        self.config = config
        
        # Initialize all components
        self.image_processor = ImageProcessing(config)
        self.vision_encoder = Vision_encoder(config)
        self.projector = MultimodalProjector(config.hidden_size, config.text_dim)
        self.tokenizer = TextTokenizer(vocab_path)
        self.language_model = LanguageModel(
            vocab_size=len(self.tokenizer),
            hidden_size=config.text_dim,
            num_layers=6,  # Adjust as needed
            num_heads=8    # Adjust as needed
        )
    
    def forward(self, images, texts):
        """
        Forward pass for training
        images: List of PIL Images or tensors
        texts: List of text prompts with <image> placeholders
        """
        # Process images
        pixel_values = self.image_processor.process(images)
        image_embeddings = self.vision_encoder(pixel_values)
        projected_embeddings = self.projector(image_embeddings)
        
        # Process texts and prepare multimodal input
        all_text_tokens = []
        all_image_positions = []
        
        for i, text in enumerate(texts):
            text_tokens, image_positions, _ = self.tokenizer.prepare_input(
                text, projected_embeddings[i:i+1]
            )
            all_text_tokens.append(torch.tensor(text_tokens))
            all_image_positions.extend(image_positions)
        
        # Pad text tokens to same length
        text_tokens_padded = torch.nn.utils.rnn.pad_sequence(
            all_text_tokens, batch_first=True, padding_value=0
        )
        
        # Forward through language model
        logits = self.language_model(
            text_tokens_padded, 
            projected_embeddings, 
            all_image_positions
        )
        
        return logits
    
    def generate(self, image, prompt, max_length=50):
        """
        Inference method for text generation
        """
        # Process image
        pixel_values = self.image_processor.process(image)
        image_embeddings = self.vision_encoder(pixel_values)
        projected_embeddings = self.projector(image_embeddings)
        
        # Prepare input
        text_tokens, image_positions, _ = self.tokenizer.prepare_input(
            prompt, projected_embeddings
        )
        
        # Convert to tensor
        input_ids = torch.tensor([text_tokens])
        
        # Generate text (we'll implement this properly next)
        # For now, just return the forward pass
        with torch.no_grad():
            logits = self.language_model(
                input_ids, projected_embeddings, image_positions
            )
        
        return logits