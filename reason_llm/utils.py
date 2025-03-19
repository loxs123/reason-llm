from copy import deepcopy
import torch
from accelerate.utils import is_deepspeed_available
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import LogitsProcessor
import time
import re

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
            mask[i, assistant_idx[-1] + 1:] = 1.0  # 从最后一个 assistant_id 位置开始置 1
    
    return mask


def create_suffix_mask(input_ids, eos_id, ):
    # 使用与单轮
    mask = torch.zeros_like(input_ids)  # 初始化全零矩阵

    for i, row in enumerate(input_ids):
        eos_idx = (row == eos_id).nonzero(as_tuple=True)[0]

        if len(eos_idx) > 0:
            mask[i, eos_idx[0] + 1: ] = 1
    
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

def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    copy from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/utils.py

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio

def cumulative_sum(tensor: torch.Tensor, dim: int = 0, reverse: bool = False):
    """
    Compute the cumulative sum of a tensor.
    
    Parameters:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the cumulative sum, default is 0.
        reverse (bool): Whether to compute the cumulative sum in reverse order, default is False.
    
    Returns:
        torch.Tensor: The tensor with the computed cumulative sum.
    """

    if reverse:
        return torch.flip(torch.cumsum(torch.flip(tensor, [dim]), dim=dim), [dim])
    return torch.cumsum(tensor, dim=dim)


def z_score_normalization(x, dim = 1):
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    z_scores = (x - mean) / (std + 1e-4)
    return z_scores



def masked_z_score_normalization(data, mask, fill_value=0):
    """
    Compute the z-score along dim=1 for data after applying the mask, and keep the output shape the same as data.
    Positions where mask == 0 are filled with fill_value.

    Args:
        data (torch.Tensor): Input 2D tensor.
        mask (torch.Tensor): 2D mask tensor with the same shape as data, where 1 indicates to keep and 0 indicates to discard.
        fill_value (float): Value to fill positions where mask == 0. Default is 0.

    Returns:
        torch.Tensor: Tensor with the same shape as data, where positions with mask == 1 contain z-scores and positions with mask == 0 contain fill_value.
    """
    # Ensure data and mask have the same shape
    assert data.shape == mask.shape, "data and mask must have the same shape"
    
    # Convert mask to boolean type
    mask = mask.bool()
    
    # Initialize the output tensor with fill_value
    output = torch.full_like(data, fill_value)
    
    # Process each row individually
    for i in range(data.size(0)):
        # Get the current row's data and mask
        row_data = data[i]
        row_mask = mask[i]
        
        # If the current row has at least two mask == 1 value
        if row_mask.sum() >= 2:
            # Extract data where mask == 1
            masked_data = row_data[row_mask]
            
            # Compute mean and standard deviation
            mean = masked_data.mean()
            std = masked_data.std()
            
            # Compute z-scores, adding a small epsilon to avoid division by zero
            z_scores = (masked_data - mean) / (std + 1e-4)
            
            # Place the z-scores back into the corresponding positions in the output
            output[i][row_mask] = z_scores
    
    return output

def masked_mean(data, mask, dim = 1):
    return (data * mask).sum(dim = dim) / (mask.sum(dim = dim) + 1e-4)

def remove_stutter(text, min_repeat=3, max_len=5):
    """
    Removes repetitive "stuttering" at the end of a text.

    :param text: The input text to be processed.
    :param min_repeat: Minimum number of times a repeated pattern is considered stuttering.
    :param max_len: Maximum length of the repeated phrase to check.
    :return: The processed text with stuttering removed.
    """
    for length in range(1, max_len + 1):  # Iterate over possible repetition pattern lengths
        pattern = r"(" + re.escape(text[-length:]) + r"){%d,}$" % min_repeat
        if re.search(pattern, text):
            return re.sub(pattern, "", text)  # Remove excessive repetitions
    return text
