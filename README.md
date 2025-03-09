# GRPO-VLLM: Efficient LLM Fine-Tuning with Optimized Resource Utilization 🚀

[![GitHub Stars](https://img.shields.io/github/stars/loxs123/grpo-vllm?style=social)](https://github.com/loxs123/grpo-vllm)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A cutting-edge framework for efficient GRPO algorithm implementation with VLLM acceleration, enabling large language model fine-tuning(7B full finetune+GRPO) on single 80GB GPU.

## 🌟 Key Features

**⚡ Ultra-Efficient Resource Usage**
- Lower GPU memory consumption than other methods
- Serialized sampling & training pipeline for optimal GPU utilization
- Supports training with context lengths up to 8K tokens
- Batch-agnostic answer processing
- Supports Lora fine-tuning

**🚀 Accelerated Performance**
- VLLM-powered sampling acceleration

**🧩 Production-Ready Design**

- Simple directory structure
- DeepSpeed Zero-3 integration
- Seamless HuggingFace ecosystem compatibility

## 🎯 Why GRPO-VLLM?

| Challenge                  | Conventional Solutions | Our Approach               |
|----------------------------|------------------------|----------------------------|
| High VRAM Requirements         | Dual-model loading(vllm && train) | Single GPU support        |
| Slow Sampling Speed        | Transformers processing   | VLLM GPU acceleration      |
| High Min Batch Size Per Device  | group size   | 1   |
| Memory Inefficiency        | Dual-model loading(vllm && train) | Single-model architecture  |

## 🛠️ Getting Started

### Prerequisites
- NVIDIA GPU
- CUDA 11.8+
- Python 3.9+

### Installation
```bash
git clone https://github.com/loxs123/grpo-vllm.git
cd grpo-vllm
pip install -e .
```

### Project Structure
```bash
├── data
│   ├── train.csv        # Training dataset
│   ├── test.csv         # Test dataset
│   └── buffer.json      # Auto-generated training buffer
├── model                # Model directory
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer...
└── grpo_vllm            # Core framework
    ├── grpo_config.py   # Training configuration
    ├── grpo_dataset.py  # Data processing
    └── ...              # Implementation modules
```

### Dataset Format (`train.csv`)
| question                                   | answer |
|-------------------------------------------|--------|
| Mathematical problem in LaTeX...          | 60     |
| Calculus optimization problem...          | 15     |
| Algebraic equation challenge...           | 20     |

### Launch Training
```bash
python scripts/train.py
```

### Tips

1. The larger the Lora rank, the better(≥128); a small Lora rank may lead to poor convergence.
2. The larger the batch size, the better.

## 📊 Experimental Results (Pending)



*Testing on [AIME 2024 Dataset](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)*

## 🧠 Technical Foundation

### Core Components
1. **GRPO Algorithm** - Group Relative Policy Optimization
2. **VLLM Acceleration** - Paged Attention implementation
3. **Dynamic Batching** - Flexible sequence processing
4. **Memory Optimization** - Gradient checkpointing + DeepSpeed


## 📚 References
1. [VLLM Official Implementation](https://github.com/vllm-project/vllm)
2. [DeepSeek-R1 Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
3. [TRL Library](https://github.com/huggingface/trl)
4. [AIME Dataset](https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024)

## 🤝 Contribution Roadmap
    - [ ] Complete initial benchmark results
    - [ ] load checkpoint? or only model?
---

*Empowering efficient LLM fine-tuning for everyone* 🤖
