from copy import deepcopy
import torch
from accelerate.utils import is_deepspeed_available
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import LogitsProcessor
import time

if is_deepspeed_available():
    import deepspeed

def prepare_deepspeed(model, accelerator):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model

def create_prefix_mask(input_ids, assistant_id):
    mask = torch.zeros_like(input_ids)  # 初始化全零矩阵
    
    for i, row in enumerate(input_ids):
        # 找到最后一个 assistant_id 在当前行的索引
        assistant_idx = (row == assistant_id).nonzero(as_tuple=True)[0]
        if len(assistant_idx) > 0:
            mask[i, assistant_idx[-1]:] = 1.0  # 从最后一个 assistant_id 位置开始置 1
    
    return mask


# model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
def create_suffix_mask(input_ids, eos_id, ):
    # 使用与单轮
    mask = torch.zeros_like(input_ids)  # 初始化全零矩阵
    
    for i, row in enumerate(input_ids):
        # 找到最后一个 assistant_id 在当前行的索引
        eos_idx = (row == eos_id).nonzero(as_tuple=True)[0]

        if len(eos_idx) > 0:
            mask[i, eos_idx[0]: ] = 1  # 从最后一个 assistant_id 位置开始置 1
    
    mask = 1.0 - mask
    return mask


def apply_lora(model_dir):

    model_name_or_path = model_dir
    output_path = os.path.join(model_dir, 'merge')
    merge_model_path = os.path.join(model_dir, 'merge')
    lora_path = os.path.join(model_dir, 'lora')
    if not os.path.exists(lora_path): return False

    model_name_or_path = model_dir
    
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # base.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    ).to('cuda:0')

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(merge_model_path)

    return True

def get_per_token_logps(model, input_ids):
    logits = model(input_ids).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)