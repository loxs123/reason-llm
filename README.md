# GRPO-VLLM

### 动机

当前使用GRPO算法微调大语言模型一般需要比较大的显存，显存较低的使用场景将会受限。

GRPO算法采样过程是比较慢的，需要使用VLLM加速，但是现在支持GRPO算法+VLLM加速的开源算法库比较少。

当前支持 VLLM+GRPO算法 的开源算法库，例如[trl](https://github.com/huggingface/trl)，同一问题的不同回答只能放在同一批次中，如果需要设置句长比较大的时候（微调 r1 模型），容易出现OOM。

当前支持强化学习微调大语言模型的框架，大多采用异步，需要同时加载推理模型和训练模型，对显存消耗比较大。

本仓库将针对这几个问题进行改进，旨在做到三点：

为充分利用GPU资源，采用串行采样和训练。

支持VLLM加速采样。

支持同一问题的多个回答可以在不同批次训练，以微调更大的句长。

### 使用场景

在一张80G的显卡上运行GRPO算法。

### 运行步骤

#### 数据格式及目录结构

-- data

---- train.csv

-- model

---- config.json

---- model.safetensors

---- tokenizer.json

---- tokenizer_config.json

---- generation_config.json

-- grpo_vllm

---- grpo_config.py

---- grpo_dataset.py

---- [other_code.py]



train.csv 格式

| question                                                     | answer |
| ------------------------------------------------------------ | ------ |
| Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\log_xw=24$ , $\log_y w = 40$ and $\log_{xyz}w=12$ . Find $\log_zw$ . | 60     |
| Let $f(x)=\|x-p\|+\|x-15\|+\|x-p-15\|$ , where $0 < p < 15$ . Determine the minimum value taken by $f(x)$ for $x$ in the interval $p \leq x\leq15$ . | 15     |
| What is the product of the real roots of the equation $x^2 + 18x + 30 = 2 \sqrt{x^2 + 18x + 45}$ ? | 20     |

model 目录：最开始的时候放BaseModel，在训练过程中会动态更新模型。

data目录：要求目录可写，会在这个目录生成buffer.json

#### 环境准备

```bash
git clone https://github.com/loxs123/grpo-vllm.git
pip install -e .
```

#### 运行命令

```bash
python scripts/train.py
```

### 计划验证方案

Train Set：https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024 （去掉24年的部分）

Test Set：https://huggingface.co/datasets/Maxwell-Jia/AIME_2024

BaseModel：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B



### 实验结果

| 模型         | 准确率 | 训练轮次 | 花费 | 硬件配置 |
| ------------ | ------ | -------- | ---- | -------- |
| BaseModel    |        | -        | -    | -        |
| r1蒸馏 + SFT | ？     | 2        | ？   | ？       |
| GRPO         | ？     | 2        | ？   | ？       |

实验结果待补充，r1蒸馏的方案是调用deepseek-r1模型得到训练集的思维链回复，然后再用这些数据全量微调BaseModel；GRPO是使用本仓库的方法微调。

### 参考

简单实现rlhf+vllm：https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf.py

PPO+vllm：https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo_ray.py

https://github.com/casper-hansen/AutoAWQ
