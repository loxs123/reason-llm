# GRPO-VLLM

### 动机

1. 当前VLLM+GRPO算法的开源代码比较匮乏。

2. 当前支持 VLLM+GRPO算法 的开源训练库，例如trl [commit](https://example.comhttps://github.com/huggingface/trl/commit/ed14ed90438860fc59b8b7694d4e103a2a146a57#diff-3dccaf6ed3f406ca989a3fe919c767e614cfc90ba81a8a761567ff5ca2cb97dd)，同一问题的不同回答只能放在同一批次中，如果需要设置句长比较大的时候（微调 r1 模型），容易出现OOM。

本仓库将针对这两个问题进行改进，旨在做到两点：

1. 支持VLLM加速采样，并且实现并行训练和采样。

2. 同一问题的不同回答可以出现在不同批次中，以支持微调更大的句长。

### 使用场景

单机多卡

### 改进思路

使用VLLM加速推理，从而节省训练时间。

使用一张卡用vllm，另外多张卡同步训练。

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

model 目录：最开始的时候放BaseModel（记得在其余地方备份），在训练过程中会动态更新模型。

data目录：要求目录可写，会在这个目录生成buffer.json

#### 环境准备

pip install -r requirements.txt

#### 运行命令

1. 运行vllm收集回复

   ```python
   CUDA_VISIBLE_DEVICES=0 python grpo_vllm/vllm_engine.py
   ```

2. 开启另外一个窗口训练模型

   ```python
   CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file grpo_vllm/deepspeed_zero3.yaml grpo_vllm/main.py
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



