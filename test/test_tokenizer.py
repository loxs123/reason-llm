from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

# model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
def create_prefix_mask(input_ids, assistant_id):
    mask = torch.zeros_like(input_ids)  # 初始化全零矩阵
    
    for i, row in enumerate(input_ids):
        # 找到最后一个 assistant_id 在当前行的索引
        assistant_idx = (row == assistant_id).nonzero(as_tuple=True)[0]
        if len(assistant_idx) > 0:
            mask[i, assistant_idx[-1]:] = 1  # 从最后一个 assistant_id 位置开始置 1
    
    return mask

# example()
example1 = {'messages': [{'role': 'user', 'content': '今天是几号？'},{'role': 'assistant', 'content': '今天是1号'}]}
example2 = {'messages': [{'role': 'user', 'content': '北京在哪里，你在做什么'},{'role': 'assistant', 'content': '北京在中国的北部。你在做什么'}]}
prompt_text1 = maybe_apply_chat_template(example1, tokenizer)['text']
prompt_text2 = maybe_apply_chat_template(example2, tokenizer)['text']
prompt_texts = [prompt_text1, prompt_text2]
assistant_id = tokenizer('<｜Assistant｜>', add_special_tokens=False)['input_ids'][0]
input_ids = tokenizer(prompt_texts, return_tensors="pt", padding=True,padding_side="left",  add_special_tokens=False)['input_ids']
prefix_mask = create_prefix_mask(input_ids, assistant_id)
print(assistant_id)
print(input_ids)
print(prefix_mask)

# <｜Assistant｜>

# inputs = [
#     [ {'messages': []}, {'messages': []}, ...  ],
#     ...

# ]