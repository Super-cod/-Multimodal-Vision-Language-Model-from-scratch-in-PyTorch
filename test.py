# test_complete_pipeline.py
from multimodal_model import MultimodalModel
from inference import TextGenerator
from modeling_slip import simplevisionconfig
from PIL import Image
import torch

def test_complete_pipeline():
    # Create config
    config = simplevisionconfig(
        image_size=224,
        patch_size=16,
        hidden_size=768,
        text_dim=2048,
        num_hidden_layers=4
    )
    
    # Create model
    model = MultimodalModel(config, 'vocab.json')
    
    # Create text generator
    generator = TextGenerator(model, model.tokenizer)
    
    # Test data
    test_image = Image.new('RGB', (500, 300), color='red')
    test_prompt = "describe this image"
    
    print("Testing complete pipeline...")
    
    # Test forward pass (training)
    try:
        logits = model.forward([test_image], [test_prompt])
        print("✓ Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # Test generation (inference)
    try:
        # For now, just test that it runs (since model is untrained)
        generated_text = generator.generate(test_image, test_prompt, max_length=10)
        print("✓ Generation successful!")
        print(f"Generated: {generated_text}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
    
    print("\n🎉 Pipeline test completed!")

if __name__ == "__main__":
    test_complete_pipeline()