# Multimodal Vision-Language Model (VLM) from Scratch

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Design-PaliGemma%20%2F%20LLaVA%20Style-blue.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Shape%20%26%20Gradient%20Flow%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, end-to-end autoregressive Vision-Language Model (VLM) implemented in PyTorch from first principles with zero external high-level model wrappers. The architecture follows modern multimodal designs (PaliGemma / LLaVA), fusing a **Vision Transformer (ViT)** visual patch encoder with a **Causal Transformer Decoder** via a cross-modal linear projection bottleneck.

---

## High-Level System Architecture

```
                    ┌──────────────────────────────┐
                    │      RGB Input Image         │
                    │      (B, 3, 224, 224)        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Patch Embedding (Conv2d)   │
                    │   Kernel: 16x16, Stride: 16   │
                    └──────────────┬───────────────┘
                                   │  (B, 196, d_vis)
                                   ▼
                    ┌──────────────────────────────┐
                    │ Prepend Learnable [CLS]      │
                    │ + 1D Positional Embeddings   │
                    └──────────────┬───────────────┘
                                   │  (B, 197, d_vis)
                                   ▼
                    ┌──────────────────────────────┐
                    │ ViT Transformer Encoder      │
                    │ Pre-LN MHA + MLP (GELU)      │
                    └──────────────┬───────────────┘
                                   │  (B, 197, d_vis)
                                   ▼
                    ┌──────────────────────────────┐
                    │ Multimodal Projector (Linear)│
                    │ W_proj: d_vis ──► d_text     │
                    └──────────────┬───────────────┘
                                   │  (B, 197, d_text)
                                   ▼
┌──────────────────┐    ┌──────────────────────────┐
│ Text Prompt with │───►│ Sequence Binding Engine  │
│ <image> Anchor   │    │ Inserts visual tokens at │
└──────────────────┘    │ <image> placeholder pos  │
                        └──────────┬───────────────┘
                                   │  (B, 197 + L_text - 1, d_text)
                                   ▼
                        ┌──────────────────────────┐
                        │ Causal Transformer LM    │
                        │ Multi-Head Self-Attention│
                        │ + Autoregressive Mask    │
                        └──────────┬───────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────┐
                        │ Vocabulary Head Logits   │
                        │ (B, S, vocab_size)       │
                        └──────────────────────────┘
```

---

## Architectural & Mathematical Foundations

### 1. Visual Tokenization & Patch Projection
Given an input image $I \in \mathbb{R}^{C \times H \times W}$ with $C=3, H=224, W=224$ and patch size $P=16$:
$$N = \frac{H \cdot W}{P^2} = \left(\frac{224}{16}\right)^2 = 196 \text{ visual patches}$$

Each patch is mapped to a $d_{\text{vis}}$-dimensional latent embedding ($d_{\text{vis}} = 768$) via a 2D convolution with stride $P$:
$$X_{\text{patch}} = \text{Conv2D}(I, \text{kernel}=16, \text{stride}=16) \in \mathbb{R}^{B \times N \times d_{\text{vis}}}$$

A learnable class token $x_{\text{cls}} \in \mathbb{R}^{1 \times 1 \times d_{\text{vis}}}$ is prepended, yielding $N + 1 = 197$ visual tokens:
$$Z_0 = [x_{\text{cls}} \,\|\, X_{\text{patch}}] + E_{\text{pos}}, \quad E_{\text{pos}} \in \mathbb{R}^{1 \times (N+1) \times d_{\text{vis}}}$$

### 2. Cross-Modal Linear Projector
The visual encoder outputs tokens in vision latent space $\mathbb{R}^{d_{\text{vis}}}$ ($768$). The language model operates in textual hidden space $\mathbb{R}^{d_{\text{text}}}$ ($2048$). The multimodal projector performs affine subspace alignment:
$$H_{\text{vis}} = Z_L W_{\text{proj}} + b_{\text{proj}}, \quad W_{\text{proj}} \in \mathbb{R}^{d_{\text{vis}} \times d_{\text{text}}}$$

### 3. Dynamic Multimodal Sequence Interleaving
Text prompts contain an explicit semantic anchor token `<image>`. The tokenizer scans for the index $k$ of `<image>` and splices the projected visual token sequence $[h_1, \dots, h_{197}]$ directly into the textual embedding stream:
$$S_{\text{multimodal}} = [e(w_1), \dots, e(w_{k-1}), h_1, h_2, \dots, h_{197}, e(w_{k+1}), \dots, e(w_M)]$$
This enables the causal decoder to attend directly to visual features through standard autoregressive self-attention without specialized cross-attention cross-talk modules.

### 4. Autoregressive Language Generation & Sampling
During inference, next-token prediction uses Nucleus (Top-$p$) filtering:
$$P(w_{t} \mid w_{<t}, I) = \text{softmax}\left(\frac{\text{logits}_t}{\tau}\right)$$
Tokens with cumulative probability exceeding threshold $p$ are masked to $-\infty$, preventing low-probability degenerative hallucination.

---

## Setup & Execution Instructions

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* Pillow (`pip install torch torchvision pillow`)

### Running the Complete Suite

Clone and run in PowerShell:

```powershell
Set-Location "C:\Users\swaya\Desktop\TIMEPASS\claude\git\Multimodal-Vision-Language-Model-from-scratch"
.\run_tests.ps1
```

Or execute independently via Python:

```bash
# 1. Run architectural shape contract and gradient flow verification tests
python tests/test_multimodal_architecture.py

# 2. Run end-to-end forward pass and autoregressive generation
python test.py
```

---

## Concrete Inference Example

Below is a minimal end-to-end inference script using the unified model:

```python
import torch
from PIL import Image
from modeling_slip import simplevisionconfig, simpletextconfig
from multimodal_model import MultiModalModel
from text_tokenizer import TextTokenizer
from processing_paligemma import PaliGemmaProcessor

# 1. Initialize tokenizer and processor
tokenizer = TextTokenizer(vocab_file="vocab.json")
processor = PaliGemmaProcessor()

# 2. Load configurations and model
vision_cfg = simplevisionconfig(image_size=224, patch_size=16, num_hidden_layers=6, hidden_size=768)
text_cfg = simpletextconfig(vocab_size=len(tokenizer.vocab), max_position_embeddings=512, hidden_size=2048)
model = MultiModalModel(vision_cfg, text_cfg)
model.eval()

# 3. Prepare image and multimodal prompt with <image> semantic anchor
image = Image.new("RGB", (224, 224), color=(73, 109, 137))
pixel_values = processor.process_image(image)  # Shape: (1, 3, 224, 224)
prompt = "<image> What objects are visible in this scene?"
input_ids, image_insert_idx = tokenizer.prepare_input(prompt)

# 4. Generate response with Nucleus (Top-p) sampling
with torch.no_grad():
    output_tokens = model.generate(
        input_ids=input_ids.unsqueeze(0),
        pixel_values=pixel_values,
        image_insert_idx=image_insert_idx,
        max_new_tokens=20,
        temperature=0.8,
        top_p=0.9
    )

response = tokenizer.decode(output_tokens[0])
print(f"Generated Response: {response}")
```

---

## Automated Verification & Test Outputs

```
>>> Running Multimodal Architecture Verification Suite <<<
[PASS] Test 1: Vision encoder token dimensions (1 CLS + 196 patches) verified: (2, 197, 768)
[PASS] Test 2: Multimodal projector visual-textual dimension alignment verified: (2, 197, 2048)
Loaded vocabulary with 6748 tokens
[PASS] Test 3: End-to-end backpropagation gradient flow verified across all 3 sub-networks:
       - Vision Encoder Grad Norm : 8.1193
       - Projector Grad Norm      : 1.3865
       - Language Model Grad Norm : 16.3071

>>> ALL MULTIMODAL ARCHITECTURE TESTS PASSED <<<
```

---

## Honest Limitations & Engineering Scope

| Dimension | Implemented Scope | Out of Current Scope / Future Enhancements |
| :--- | :--- | :--- |
| **Model Weights** | Clean architectural implementation from first principles | Multi-billion parameter pre-trained checkpoints (weights randomly initialized or loaded via adapters) |
| **Tokenizer** | Custom word-level tokenizer with `<image>` anchor | Production Byte-Pair Encoding (BPE) / SentencePiece tokenizer |
| **Attention Mechanism** | Standard PyTorch Scaled Dot-Product Attention | FlashAttention-2 custom CUDA kernels or PagedAttention |
| **Multimodal Inputs** | Single RGB image ($224 \times 224$) per prompt | Multi-image interleaved dialogues or video sequence processing |
| **Sequence Length** | Capped at 512 tokens ($197 \text{ vision} + 315 \text{ text}$) | Long-context RoPE scaling for multi-thousand token contexts |

---

## File Structure

```
Multimodal-Vision-Language-Model-from-scratch/
├── modeling_slip.py              # Vision & model hyperparameter configurations
├── vision_encoder.py             # ViT patch projection & transformer encoder layers
├── multimodal_projector.py       # Cross-modal linear projection bottleneck
├── processing_paligemma.py       # Image resizing, normalization, and tensor conversions
├── text_tokenizer.py             # Vocabulary management and <image> anchor extraction
├── language_model.py             # Causal autoregressive transformer decoder
├── multimodal_model.py           # Unified end-to-end VLM orchestrator
├── inference.py                  # Autoregressive generation with Top-p nucleus sampling
├── vocab.json                    # Base token vocabulary (6,748 tokens)
├── test.py                       # Full forward pass and autoregressive generation test
├── tests/
│   └── test_multimodal_architecture.py # Shape invariants & gradient flow unit tests
├── run_tests.ps1                 # Master verification script
├── .gitignore                    # Python & PyTorch ignore rules
├── LICENSE                       # MIT License
└── README.md                     # Architectural specification & technical documentation
```

---

## License

MIT License. Copyright (c) 2026 Swayam.
