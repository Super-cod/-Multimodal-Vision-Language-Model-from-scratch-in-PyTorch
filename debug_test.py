#!/usr/bin/env python3

import torch
from multimodal_model import MultimodalModel
from modeling_slip import simplevisionconfig
from PIL import Image

print("Creating config...")
config = simplevisionconfig(
    image_size=224,
    patch_size=16,
    hidden_size=768,
    text_dim=2048,
    num_hidden_layers=4
)

print("Creating model...")
model = MultimodalModel(config, vocab_path='vocab.json')

print("Creating test data...")
test_image = Image.new('RGB', (500, 300), color='red')
test_prompt = "describe this <image>"

print("Processing image...")
pixel_values = model.image_processor.process([test_image])
print(f"Pixel values shape: {pixel_values.shape}")

print("Running vision encoder...")
image_embeddings = model.vision_encoder(pixel_values)
print(f"Image embeddings shape: {image_embeddings.shape}")

print("Running projector...")
projected_embeddings = model.projector(image_embeddings)
print(f"Projected embeddings shape: {projected_embeddings.shape}")

print("Tokenizing text...")
text_tokens, image_positions, _ = model.tokenizer.prepare_input(
    test_prompt, projected_embeddings[0:1]
)
print(f"Text tokens: {text_tokens}")
print(f"Image positions: {image_positions}")

print("Converting to tensor...")
text_tokens_tensor = torch.tensor([text_tokens])
print(f"Text tokens tensor shape: {text_tokens_tensor.shape}")

print("Running language model...")
try:
    logits = model.language_model(
        text_tokens_tensor, 
        projected_embeddings, 
        image_positions
    )
    print(f"✅ Success! Logits shape: {logits.shape}")
except Exception as e:
    print(f"❌ Error in language model: {e}")
    import traceback
    traceback.print_exc()
