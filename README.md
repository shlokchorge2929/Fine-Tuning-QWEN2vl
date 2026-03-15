# Fine-Tuning Qwen2-VL for LaTeX OCR


<p align="center">
  A parameter-efficient fine-tuning pipeline for <strong>Qwen2-VL</strong> (Vision-Language Model) that converts images of mathematical expressions and handwritten equations into structured <strong>LaTeX code</strong> — optimized for resource-constrained hardware using QLoRA and Unsloth.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [How to Clone](#how-to-clone)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Results](#results)
- [Version & Compatibility](#version--compatibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Mathematical OCR (Optical Character Recognition) is a notoriously difficult task — handwritten or printed equations contain complex symbols, nested structures, and multi-line alignments that general-purpose models struggle with.

This project fine-tunes **Qwen2-VL**, a state-of-the-art multimodal vision-language model, on the specific task of **LaTeX transcription from images**. By leveraging:

- **QLoRA** (Quantized Low-Rank Adaptation) for memory-efficient training
- **Unsloth** for accelerated backward passes
- **4-bit NormalFloat (NF4)** quantization for VRAM reduction

...the model can be fine-tuned even on mid-range GPUs (like NVIDIA T4) while achieving significantly better performance than the base Qwen2-VL on complex mathematical structures.

---

## Features

- Fine-tuning of Qwen2-VL on image to LaTeX pairs
- 4-bit quantization using BitsAndBytes (NF4 format)
- LoRA adapter training via Hugging Face PEFT
- Unsloth-optimized kernels for faster training and reduced memory
- Multimodal chat template formatting for image-text inputs
- Supports complex LaTeX output: matrices, Greek symbols, multi-line alignments
- Inference pipeline included for immediate testing
- Compatible with NVIDIA T4 (16GB) and A100 GPUs

---

## Project Structure

```
Fine-Tuning-QWEN2vl/
|
|-- finetuningqwen_vl.ipynb       # Main Jupyter Notebook (training + inference)
|-- README.md                     # Project documentation
|
-- (outputs/)                     # Auto-generated after training
    |-- lora_model/               # Saved LoRA adapter weights
    -- logs/                      # Training logs
```

> **Note:** The dataset and model weights are loaded directly from Hugging Face Hub at runtime. No large files need to be committed to this repository.

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16GB VRAM) | NVIDIA A100 (40/80GB) |
| RAM | 16 GB | 32 GB+ |
| Storage | 20 GB free | 50 GB free |

### Software & Versions

| Package | Version |
|---------|---------|
| Python | `3.10+` |
| PyTorch | `2.1.0+` |
| CUDA | `11.8` or `12.1` |
| Transformers | `4.45.0+` |
| PEFT | `0.12.0+` |
| TRL | `0.11.0+` |
| Unsloth | `2024.11+` |
| BitsAndBytes | `0.43.0+` |
| xformers | `0.0.27+` |

---

## Installation & Setup

### Step 1 — Set up a Python Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### Step 2 — Install Core Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Install Hugging Face Stack

```bash
pip install transformers==4.45.0 \
            peft==0.12.0 \
            trl==0.11.0 \
            bitsandbytes==0.43.3 \
            accelerate \
            datasets
```

### Step 4 — Install Unsloth (for optimized kernels)

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install xformers
```

### Step 5 — Install Jupyter (if running locally)

```bash
pip install jupyter notebook ipywidgets
```

### Step 6 — (Optional) Hugging Face Login

If you plan to push your fine-tuned model to Hugging Face Hub:

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## How to Clone

### Clone the Repository

```bash
git clone https://github.com/shlokchorge2929/Fine-Tuning-QWEN2vl.git
```

### Navigate into the Project Directory

```bash
cd Fine-Tuning-QWEN2vl
```

### (Optional) Check Out a Specific Version / Branch

```bash
# List all available branches
git branch -a

# Switch to a specific branch
git checkout <branch-name>

# Or clone a specific branch directly
git clone -b <branch-name> https://github.com/shlokchorge2929/Fine-Tuning-QWEN2vl.git
```

### Verify the Clone

```bash
ls -la
# You should see: finetuningqwen_vl.ipynb  README.md
```

---

## Usage

### Option A — Run on Google Colab (Recommended for beginners)

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **File > Upload Notebook**
3. Upload `finetuningqwen_vl.ipynb`
4. Set runtime to **GPU** (T4 or A100 — Colab Pro recommended)
5. Run all cells top to bottom

### Option B — Run Locally with Jupyter

```bash
# After completing the installation steps above
jupyter notebook finetuningqwen_vl.ipynb
```

### Option C — Run as a Script (Advanced)

If you want to convert the notebook to a standalone Python script:

```bash
pip install nbconvert
jupyter nbconvert --to script finetuningqwen_vl.ipynb
python finetuningqwen_vl.py
```

---

### Key Configuration Parameters

Inside the notebook, you can modify these parameters to customize training:

```python
# Model configuration
model_name = "Qwen/Qwen2-VL-7B-Instruct"   # Base model from HF Hub
max_seq_length = 2048                         # Max token length

# QLoRA configuration
load_in_4bit = True                           # Enable 4-bit quantization
lora_rank = 16                                # LoRA rank (higher = more params)
lora_alpha = 16                               # LoRA scaling factor
lora_dropout = 0.05                           # Dropout for regularization

# Training configuration
num_train_epochs = 3                          # Number of training epochs
per_device_train_batch_size = 2               # Batch size per GPU
gradient_accumulation_steps = 4              # Effective batch = 2 x 4 = 8
learning_rate = 2e-4                          # Learning rate
```

---

### Running Inference

After training is complete, run inference on a new image:

```python
from PIL import Image

# Load your image
image = Image.open("your_equation_image.png")

# Run inference
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Convert this mathematical expression to LaTeX."}
        ]
    }
]

# The notebook's inference cell handles tokenization and generation
output = model.generate(...)
print(tokenizer.decode(output[0]))
```

---

## Technical Details

### Quantization Strategy

The model uses **4-bit NormalFloat (NF4)** quantization via BitsAndBytes, which:
- Reduces VRAM consumption by ~75% compared to full precision (fp32)
- Preserves model quality through information-theoretically optimal quantization
- Enables fine-tuning of a 7B parameter model on a 16GB GPU

```
Qwen2-VL-7B (full fp32)  ~28 GB VRAM
Qwen2-VL-7B (4-bit NF4)  ~6-8 GB VRAM (base load)
```

### LoRA Adapters

Instead of updating all 7 billion parameters, **LoRA** injects small trainable adapter matrices into the attention layers:

```
Total trainable params (LoRA): ~10-40M (< 1% of 7B)
Training speed gain: ~3-5x faster than full fine-tuning
```

### Unsloth Optimizations

Unsloth provides custom CUDA kernels that:
- Reduce memory usage during the backward pass by 60%
- Speed up training throughput by 2-5x
- Enable larger batch sizes without OOM errors

### Memory Management

| Technique | Memory Saved |
|-----------|-------------|
| 4-bit NF4 Quantization | ~75% vs fp32 |
| xformers Flash Attention | ~30-40% attention memory |
| Gradient Checkpointing | ~30% activation memory |
| Unsloth Kernels | ~60% backward pass memory |

---

## Results

The fine-tuned model demonstrates measurable improvements over the base Qwen2-VL on mathematical transcription tasks:

| Structure Type | Base Model | Fine-tuned Model |
|----------------|-----------|-----------------|
| Simple fractions | Good | Excellent |
| Greek symbols | Moderate | Excellent |
| Matrices | Poor | Good |
| Multi-line alignments | Poor | Good |
| Nested expressions | Moderate | Good |

> The model successfully transcribes complex structures including matrices, Greek symbols, and multi-line `align` environments, maintaining LaTeX syntax integrity significantly better than the base model.

---

## Version & Compatibility

| Component | Version Used |
|-----------|-------------|
| Qwen2-VL Base Model | `Qwen2-VL-7B-Instruct` |
| Python | `3.10` |
| PyTorch | `2.1.x` |
| Transformers | `4.45.x` |
| PEFT | `0.12.x` |
| TRL | `0.11.x` |
| BitsAndBytes | `0.43.x` |
| Unsloth | `2024.11+` |
| CUDA | `12.1` |

### Git Version Info

```bash
# Check your current version / commit
git log --oneline -5

# Pull the latest changes
git pull origin main
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

```python
# Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 8   # Keep effective batch size the same

# Or reduce max sequence length
max_seq_length = 1024
```

### BitsAndBytes Not Detecting GPU

```bash
# Reinstall with CUDA support explicitly
pip install bitsandbytes --upgrade

# Verify CUDA is visible
python -c "import torch; print(torch.cuda.is_available())"
```

### Unsloth Installation Errors

```bash
# Try pip install with no build isolation
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
```

### Notebook Kernel Crashes

- Ensure you are using a GPU runtime (not CPU)
- Restart the kernel and re-run from the top
- Check available VRAM: `nvidia-smi`

---

## Contributing

Contributions are welcome. Here is how to get started:

1. Fork this repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/Fine-Tuning-QWEN2vl.git
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request on GitHub

---

## License



The base **Qwen2-VL** model is subject to its own license. Please review [Qwen2-VL's license on Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) before commercial use.

---

## Acknowledgements

- [Qwen Team (Alibaba Cloud)](https://github.com/QwenLM/Qwen2-VL) for the base Qwen2-VL model
- [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient fine-tuning kernels
- [Hugging Face](https://huggingface.co/) for PEFT, TRL, and Transformers libraries
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for quantization support
