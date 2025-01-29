from transformers import AutoModel, Trainer
import torch

model = AutoModel.from_pretrained('bert-base-chinese', torch_dtype=torch.bfloat16)
trainer = Trainer(model)

trainer.save_model(r'C:\Users\asndj\Desktop\grpo-vllm\grpo_vllm', ) # bf16 type




