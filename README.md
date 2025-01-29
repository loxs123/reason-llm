# GRPO-VLLM

### 动机

Huggingface 的 TRL 库中，GRPO 的采样过程通过调用 `model.generate` 实现，这种方式在训练过程中耗时较长。为了解决这一问题，本仓库将对此进行优化和改进。

```python

# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
 def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
     if return_outputs:
         raise ValueError("The GRPOTrainer does not support returning outputs")
     
     ...
        
     # Generate completions
     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
         prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config) # speed much time
         # self.num_generations has been set in self.generation_config
     
     ...

     return loss


```

### 改进思路



使用VLLM加速推理，从而节省训练时间。



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



