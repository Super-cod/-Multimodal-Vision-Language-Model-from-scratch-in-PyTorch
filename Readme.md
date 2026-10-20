# Multimodal Vision-Language Model (VLM) from Scratch

A complete, end-to-end autoregressive Vision-Language Model (VLM) implemented in PyTorch from first principles. The architecture follows modern multimodal designs (PaliGemma / LLaVA), fusing a Vision Transformer (ViT) visual patch encoder with a causal Transformer Decoder via a cross-modal linear projection bottleneck.

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

## Architectural & Mathematical Specification

### 1. Visual Tokenization (Patch Projection)
Given an input image $I \in \mathbb{R}^{C \times H \times W}$ with $C=3, H=224, W=224$ and patch size $P=16$:
$$N = \frac{H \cdot W}{P^2} = \left(\frac{224}{16}\right)^2 = 196 \text{ visual patches}$$

Each patch is mapped to a $d_{\text{vis}}$-dimensional latent embedding via a 2D convolution with stride $P$:
$$X_{\text{patch}} = \text{Conv2D}(I, \text{kernel}=16, \text{stride}=16) \in \mathbb{R}^{B \times N \times d_{\text{vis}}}$$

A learnable class token $x_{\text{cls}} \in \mathbb{R}^{1 \times 1 \times d_{\text{vis}}}$ is prepended, yielding $N + 1 = 197$ visual tokens:
$$Z_0 = [x_{\text{cls}} \,\|\, X_{\text{patch}}] + E_{\text{pos}}, \quad E_{\text{pos}} \in \mathbb{R}^{1 \times (N+1) \times d_{\text{vis}}}$$

### 2. Cross-Modal Projector
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

## Verification & Test Suite

The repository contains an automated verification suite validating tensor shape contracts, cross-modal dimension alignment, and end-to-end backpropagation gradients across all sub-networks.

### Running All Tests

Execute the automated test runner in PowerShell:

```powershell
Set-Location "C:\Users\swaya\Desktop\TIMEPASS\claude\git\Multimodal-Vision-Language-Model-from-scratch"
.\run_tests.ps1
```

Or run the modules independently:

```powershell
# 1. Architectural Shape Contracts & Gradient Flow
python tests/test_multimodal_architecture.py

# 2. End-to-End Pipeline & Autoregressive Inference
python test.py
```

### Verified Test Outputs

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
