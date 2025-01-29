# GRPO-VLLM

动机：Huggingface的trl库中GRPO的采样是直接通过调用model.generate来实现的，非常消耗训练时长，本仓库对这个地方进行改进。

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



改进思路：使用VLLM加速推理，从而节省训练时间。









