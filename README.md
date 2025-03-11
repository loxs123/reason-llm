# ReasonLLM: Efficient LLM RL Fine-Tuning with Optimized Resource Utilization 🚀

[![GitHub Stars](https://img.shields.io/github/stars/loxs123/grpo-vllm?style=social)](https://github.com/loxs123/reason-llm)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A cutting-edge framework for efficient GRPO algorithm implementation with VLLM acceleration, enabling large language model fine-tuning with lower GPU memory usage.

## 🌟 Key Features

**⚡ Ultra-Efficient Resource Usage**
- Lower GPU memory consumption than other methods
- Serialized sampling & training pipeline for optimal GPU utilization
- Dynamic-Batch processing
- Supports Lora fine-tuning

**🚀 Accelerated Performance**
- VLLM-powered sampling acceleration

**🧩 Production-Ready Design**
- Simple directory structure
- DeepSpeed Zero-2/3 integration
- Seamless HuggingFace ecosystem compatibility

## 🎯 Why ReasonLLM?

| Challenge                  | Conventional Solutions | Our Approach               |
|----------------------------|------------------------|----------------------------|
| Slow Sampling Speed        | Transformers processing   | VLLM GPU acceleration      |
| High Min Batch Size Per Device  | group size   | 1   |
| Memory Inefficiency/High VRAM Requirements | Dual-model loading(vllm/train) | Single-model architecture  |

## 🛠️ Getting Started

### Prerequisites
- NVIDIA GPU
- CUDA 12+
- Python 3.9+

### Installation
```bash
git clone https://github.com/loxs123/reason-llm.git
cd reason-llm
pip install -e . # If it fails, please install the required dependencies one by one.
```

### Project Structure
```bash
├── data
│   ├── test.csv         # Test dataset
│   └── buffer.json      # Auto-generated training buffer
├── model                # Model directory
│   ├── config.json      # put your model here
│   ├── model.safetensors
│   └── tokenizer...
└── reason_llm            # Core framework
    ├── config.py         # Training configuration
    ├── reward_fn.py      # Reward Functions
    └── ...              # Implementation modules
```

### Launch Training
```bash
nohup python -u scripts/train.py &
```

### Some experiences and tips.

1. The larger the Lora rank, the better(≥128);

2. The larger the batch size, the better.

3. Removing samples with Advantage < 0 can lead to a better result.

4. Removing samples where reward.std() is too small (<0.1).

## 📊 Experimental Results

| Item            | detail                                         |
|---------------|--------------------------------------------|
| **Train Base Model** | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| **Train Type**  | full finetune                           |
| **Train Hardware** | 1×A100(80G) (May 3×3090 also be OK?) |
| **Train Time**  | 12h                                     |
| **Train Dataset**  | [xiaodongguaAIGC/X-R1-7500](https://huggingface.co/datasets/xiaodongguaAIGC/X-R1-7500) |
| **Test Dataset**  | [AIME 2024 Dataset](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) |

![实验结果](images/metrics_analysis.png)
[训练日志](log/log.out)

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

---

*Empowering efficient LLM fine-tuning for everyone* 🤖
