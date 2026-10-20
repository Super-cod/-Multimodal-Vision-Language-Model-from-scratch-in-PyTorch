"""
Comprehensive Architectural & Gradient Flow Test Suite for Multimodal VLM
Verifies patch embedding dimensions, multimodal projection alignment,
cross-modal sequence fusion, and end-to-end backpropagation gradients.
"""

import sys
import os
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_slip import simplevisionconfig
from multimodal_model import MultimodalModel
from vision_encoder import Vision_encoder
from multimodal_projector import MultimodalProjector

def test_vision_encoder_patch_dimensions():
    config = simplevisionconfig(
        image_size=224,
        patch_size=16,
        hidden_size=768,
        text_dim=2048,
        num_hidden_layers=2
    )
    encoder = Vision_encoder(config)
    encoder.eval()

    # Input: (batch=2, channels=3, height=224, width=224)
    dummy_pixels = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = encoder(dummy_pixels)

    # Expected tokens: 1 CLS token + (224 / 16) * (224 / 16) = 1 + 196 = 197 tokens
    assert features.shape == (2, 197, 768), f"Expected shape (2, 197, 768), got {features.shape}"
    print(f"[PASS] Test 1: Vision encoder token dimensions (1 CLS + 196 patches) verified: {tuple(features.shape)}")

def test_multimodal_projector_alignment():
    vision_dim = 768
    text_dim = 2048
    projector = MultimodalProjector(vision_dim, text_dim)

    dummy_vision_feats = torch.randn(2, 197, vision_dim)
    projected = projector(dummy_vision_feats)

    assert projected.shape == (2, 197, text_dim), f"Expected shape (2, 197, {text_dim}), got {projected.shape}"
    print(f"[PASS] Test 2: Multimodal projector visual-textual dimension alignment verified: {tuple(projected.shape)}")

def test_end_to_end_gradient_flow():
    config = simplevisionconfig(
        image_size=224,
        patch_size=16,
        hidden_size=256,
        num_attention_heads=8,
        text_dim=512,
        num_hidden_layers=2
    )
    model = MultimodalModel(config, 'vocab.json')
    model.train()

    test_image = Image.new('RGB', (224, 224), color='blue')
    test_prompt = "<image> describe image"

    logits = model([test_image], [test_prompt])
    # Compute dummy loss to verify backward pass
    targets = torch.randint(0, len(model.tokenizer), (logits.size(0), logits.size(1)))
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()

    # Verify gradients exist in all 3 core submodules
    vision_grad_norm = sum(p.grad.norm().item() for p in model.vision_encoder.parameters() if p.grad is not None)
    projector_grad_norm = sum(p.grad.norm().item() for p in model.projector.parameters() if p.grad is not None)
    lm_grad_norm = sum(p.grad.norm().item() for p in model.language_model.parameters() if p.grad is not None)

    assert vision_grad_norm > 0, "Vision encoder received zero gradient!"
    assert projector_grad_norm > 0, "Multimodal projector received zero gradient!"
    assert lm_grad_norm > 0, "Language model received zero gradient!"

    print(f"[PASS] Test 3: End-to-end backpropagation gradient flow verified across all 3 sub-networks:")
    print(f"       - Vision Encoder Grad Norm : {vision_grad_norm:.4f}")
    print(f"       - Projector Grad Norm      : {projector_grad_norm:.4f}")
    print(f"       - Language Model Grad Norm : {lm_grad_norm:.4f}")

if __name__ == "__main__":
    print(">>> Running Multimodal Architecture Verification Suite <<<")
    test_vision_encoder_patch_dimensions()
    test_multimodal_projector_alignment()
    test_end_to_end_gradient_flow()
    print("\n>>> ALL MULTIMODAL ARCHITECTURE TESTS PASSED <<<")
