Qwen2-VL Fine-Tuning for LaTeX OCR
​This repository contains the implementation for fine-tuning the Qwen2-VL (Vision-Language Model) for automated LaTeX OCR. The project focuses on converting mathematical images and handwritten equations into structured LaTeX code.
​Project Overview
​Model: Qwen2-VL
​Objective: Specialized transcription of mathematical notation from visual inputs.
​Optimization: Leveraged Unsloth and QLoRA (4-bit quantization) to optimize VRAM efficiency and training throughput.
​Stack: PyTorch, Hugging Face (TRL, PEFT), and Bitsandbytes.
​Technical Details
​Quantization: Applied 4-bit NormalFloat (NF4) to enable fine-tuning on resource-constrained hardware (NVIDIA T4/A100).
​Memory Management: Integrated xformers and Unsloth kernels to reduce memory overhead during the backward pass.
​Architecture: Fine-tuned vision-language adapters to map visual spatial features to precise LaTeX syntax.
​Implementation Steps
​Setup: Environment configuration for CUDA-accelerated vision-language tasks.
​Formatting: Implementation of multimodal chat templates for image-text pair processing.
​Training: Parameter-efficient fine-tuning using LoRA adapters.
​Inference: Verification of model output against complex multi-line mathematical environments.
​Results
​The model successfully transcribes complex structures, including matrices, Greek symbols, and multi-line alignments, maintaining syntax integrity better than the base pre-trained model.
