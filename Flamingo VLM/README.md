# Flamingo VLM: Visual Language Model

**Few-Shot Multimodal Learning with Interleaved Image-Text Understanding**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Vision Encoder](#vision-encoder)
   - [Perceiver Resampler](#perceiver-resampler)
   - [Gated Cross-Attention](#gated-cross-attention)
   - [Language Model](#language-model)
3. [Inference Notebook](#inference-notebook)
4. [Resources](#resources)
5. [Implementation Status](#implementation-status)
6. [References](#references)

---

## Overview

Flamingo is a Visual Language Model (VLM) developed by DeepMind that bridges powerful pretrained vision and language models to handle interleaved image-text inputs. It achieves state-of-the-art few-shot performance on a wide range of multimodal tasks — image captioning, visual question answering, and image classification — using only a handful of examples as context.

The key insight: **freeze** the pretrained vision encoder and language model, and only train a small set of new cross-attention layers that connect them. This allows Flamingo to leverage billions of parameters of pretrained knowledge while learning vision-language alignment with minimal new parameters.

### Key Innovations

- **Few-Shot Multimodal Learning**: Performs tasks like captioning and VQA with just 4–32 example image-text pairs as context, no task-specific fine-tuning required
- **Interleaved Image-Text**: Processes arbitrarily interleaved sequences of images and text (e.g., webpages, documents, conversations)
- **Perceiver Resampler**: Converts variable-size visual features into a fixed set of visual tokens, enabling efficient cross-attention
- **Gated Cross-Attention**: New layers inserted into the frozen LM that attend to visual features, with a learned gating mechanism initialized at zero for stable training
- **Frozen Backbones**: Vision encoder and language model remain frozen; only ~2% of parameters are trained

### Model Variants

| Variant | Vision Encoder | Language Model | Total Params | Trainable |
|---------|---------------|----------------|--------------|-----------|
| Flamingo-3B | NFNet-F6 | Chinchilla 1.4B | ~3B | ~200M |
| Flamingo-9B | NFNet-F6 | Chinchilla 7B | ~9B | ~1.6B |
| Flamingo-80B | NFNet-F6 | Chinchilla 70B | ~80B | ~10B |

> **OpenFlamingo** is the open-source reproduction used in our inference notebook, pairing ViT-L/14 (CLIP) with MPT-7B.

---

## Architecture

![Flamingo VLM Architecture Breakdown](Flamingo%20VLM%20%20breakdown.png)

Flamingo connects a frozen vision encoder to a frozen language model through two new trained components:

```
Input Images (interleaved with text)
     |
     v
┌──────────────────┐
│  Vision Encoder  │  ← Frozen (NFNet-F6 or CLIP ViT-L/14)
│  (per image)     │
└────────┬─────────┘
         |  variable-length spatial features
         v
┌──────────────────┐
│   Perceiver      │  ← TRAINED (new)
│   Resampler      │  Compress to fixed 64 visual tokens per image
└────────┬─────────┘
         |  fixed-size visual tokens
         v
┌──────────────────────────────────────────┐
│         Frozen Language Model            │
│  ┌─────────────────────────────────┐     │
│  │  Self-Attention (frozen)        │     │
│  │  ↓                              │     │
│  │  Gated Cross-Attention (TRAINED)│ ← Visual tokens attend here
│  │  ↓                              │     │ 
│  │  Feed-Forward (frozen)          │     │
│  │  ... repeated every N layers ...│     │
│  └─────────────────────────────────┘     │
└──────────────────────────────────────────┘
         |
         v
    Text Output (caption, answer, classification)
```

---

### Vision Encoder

Processes each input image independently into spatial feature maps.

| Aspect | Detail |
|--------|--------|
| Original Paper | NFNet-F6 (pretrained on ImageNet, frozen) |
| OpenFlamingo | CLIP ViT-L/14 (pretrained, frozen) |
| Output | Spatial feature grid per image |
| Key Property | Frozen — no gradients flow through during training |

---

### Perceiver Resampler

A key innovation that compresses variable-length visual features into a **fixed set of 64 visual tokens** per image, regardless of the input resolution.

| Aspect | Detail |
|--------|--------|
| Input | Variable-length spatial features from vision encoder |
| Output | 64 fixed-length visual tokens per image |
| Mechanism | Learned latent queries attend to visual features via cross-attention |
| Benefit | Decouples visual token count from image resolution, enables efficient cross-attention in LM |

This is inspired by the Perceiver architecture — learned queries "summarize" the visual information into a compact representation that the language model can efficiently attend to.

---

### Gated Cross-Attention

New layers inserted between the frozen LM layers (every N layers). These are the primary mechanism for injecting visual information into the language model.

| Aspect | Detail |
|--------|--------|
| Design | Cross-attention where text tokens attend to visual tokens |
| Insertion | Every 4th layer in the frozen LM (configurable) |
| Gating | `output = x + tanh(α) * cross_attn(x, visual_tokens)` where α is initialized to 0 |
| Why Gating? | At initialization, `tanh(0) = 0`, so the model starts as the original LM and gradually learns to use vision |
| Trainable | Yes — this is the main trained component |

The `tanh(α)` gating is crucial: it ensures the model starts as the pretrained LM (no visual influence) and *smoothly* learns to incorporate visual information, preventing catastrophic forgetting.

---

### Language Model

The autoregressive language model that generates text conditioned on both text and visual inputs.

| Aspect | Detail |
|--------|--------|
| Original | Chinchilla (1.4B / 7B / 70B), frozen |
| OpenFlamingo | MPT-7B, frozen |
| Role | Generates text autoregressively, attending to visual tokens via gated cross-attention |
| Key Property | Frozen — preserves all pretrained language understanding |

---

## Inference Notebook

**Notebook**: [Flamingo_inference.ipynb](Flamingo_inference.ipynb)

This notebook demonstrates **few-shot image captioning** using OpenFlamingo-9B (ViT-L/14 + MPT-7B).

**What the notebook covers:**
- Installing OpenFlamingo and downloading the 9B checkpoint (~5.5 GB)
- Loading the model with ViT-L/14 vision encoder and MPT-7B language model
- **Few-shot captioning**: Providing 2 demo image-caption pairs as context, then generating a caption for a new query image
- Handling the `<image>` and `<|endofchunk|>` special tokens for interleaved image-text input

**Usage pattern:**
```python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4,
)

# Few-shot prompt: 2 examples + 1 query
lang_x = tokenizer([
    "<image>An image of two cats.<|endofchunk|>"
    "<image>An image of a bathroom sink.<|endofchunk|>"
    "<image>An image of"
], return_tensors="pt")

generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=8,
    num_beams=2,
)
```

**Requirements:**
- Google Colab with GPU runtime (T4 minimum, ~15 GB VRAM)
- `open_flamingo`, `open-clip-torch`, HuggingFace Transformers
- ~5.5 GB disk for the OpenFlamingo-9B checkpoint

---

## Resources

### Research Paper

| Resource | File | Description |
|----------|------|-------------|
| Original Paper | [Flamingo VLM paper.pdf](Flamingo%20VLM%20paper.pdf) | "Flamingo: a Visual Language Model for Few-Shot Learning" by Alayrac et al. (NeurIPS 2022) |

### Architecture Diagram

| Resource | File | Description |
|----------|------|-------------|
| Architecture Overview | [Flamingo VLM breakdown.png](Flamingo%20VLM%20%20breakdown.png) | Complete architecture breakdown diagram |
| Editable Source | [Flamingo VLM architecture.excalidraw](Flamingo%20VLM%20architecture.excalidraw) | Excalidraw source file for the architecture diagram |

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Architecture Diagram | Complete (Excalidraw + PNG) |
| Research Paper | Included |
| Inference (OpenFlamingo-9B) | Complete (few-shot captioning) |
| From-Scratch Implementation | Possible future addition |

---

## References

### Original Paper

```bibtex
@inproceedings{alayrac2022flamingo,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Alayrac, Jean-Baptiste and Donahue, Jeff and Luc, Pauline and Miech, Antoine and Barr, Iain and Hasson, Yana and Lenc, Karel and Mensch, Arthur and Millican, Katherine and Reynolds, Malcolm and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={23716--23736},
  year={2022}
}
```

### Key Resources

| Resource | Link |
|----------|------|
| Original Paper (arXiv) | [arxiv.org/abs/2205.07065](https://arxiv.org/abs/2205.07065) |
| OpenFlamingo (open-source) | [mlfoundations/open_flamingo](https://github.com/mlfoundations/open_flamingo) |
| OpenFlamingo Model Hub | [HuggingFace: OpenFlamingo](https://huggingface.co/OpenFlamingo) |
| MPT-7B Language Model | [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) |

### Related Work

- **BLIP-2**: Bootstrapped vision-language pretraining with Q-Former
- **LLaVA**: Visual instruction tuning for large language models
- **GPT-4V**: Multimodal GPT-4 with vision capabilities
- **Kosmos-1**: Language is not all you need — multimodal foundation model
- **PaLM-E**: Embodied multimodal language model by Google

---


