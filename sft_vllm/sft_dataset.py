import torch.utils.data as data
import json
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import time
import random
import torch
from dataclasses import dataclass
from trl.data_utils import maybe_apply_chat_template

def create_prefix_mask(input_ids, assistant_id):
    mask = torch.zeros_like(input_ids)  # 初始化全零矩阵
    
    for i, row in enumerate(input_ids):
        # 找到最后一个 assistant_id 在当前行的索引
        assistant_idx = (row == assistant_id).nonzero(as_tuple=True)[0]
        if len(assistant_idx) > 0:
            mask[i, assistant_idx[-1]:] = 1  # 从最后一个 assistant_id 位置开始置 1
    
    return mask

# model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
def create_suffix_mask(input_ids, eos_id, ):
    # 使用与单轮
    mask = torch.zeros_like(input_ids)  # 初始化全零矩阵
    
    for i, row in enumerate(input_ids):
        # 找到最后一个 assistant_id 在当前行的索引
        eos_idx = (row == eos_id).nonzero(as_tuple=True)[0]

        if len(eos_idx) > 0:
            mask[i, eos_idx[0] + 1:] = 1  # 从最后一个 assistant_id 位置开始置 1
    
    mask = 1 - mask
    return mask


class SFTDataset(data.Dataset):
    def __init__(self, filename, tokenizer: AutoTokenizer, max_retries=10, retry_interval=0.5, sample_num = 8, data_size = 128):
        super(SFTDataset, self).__init__()

        self.filename = filename
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.sample_num = sample_num

        with open(self.filename) as f:
            self.data = json.load(f)[:data_size]
            
        self.index = [i for i in range(self.len_data)]

        self.tokenizer = tokenizer
        # random.shuffle(self.index)

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):

        return self.data[self.index[index]]
        
@dataclass
class DataCollatorForDialog:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features, return_tensors=None):
        """
            Args: features[batch_data]:({'input_ids':[],'labels':[]},{},...)
            return {'input_ids':[],'attention_mask':[],'labels':[]}
        """
        
        prompts_text = [maybe_apply_chat_template({'messages': example['completion'] }, self.tokenizer)["text"] for example in features]

        prompt_inputs  = self.tokenizer(
            prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        ) # 只训练前面的
        prompt_inputs, prompt_inputs_mask = prompt_inputs['input_ids'], prompt_inputs['attention_mask']

        # only for single turn
        assistant_id = self.tokenizer('<｜Assistant｜>', add_special_tokens=False)['input_ids'][0]
        prefix_mask = create_prefix_mask(prompt_inputs, assistant_id)
        suffix_mask = create_suffix_mask(prompt_inputs, self.tokenizer.eos_token_id)
        
        mask = prefix_mask * suffix_mask
        labels = torch.where(mask == 0, torch.tensor(-100), prompt_inputs)

        return {'input_ids': prompt_inputs, 'attention_mask': prompt_inputs_mask, 'labels': labels}
    

# if __name__ == '__main__':
#     a = torch.LongTensor([[1,2,3],[1,2,3]])
#     m = torch.LongTensor([[0, 0, 1], [0, 1, 1]])
#     l = torch.where(m == 0, torch.tensor(-100), a)
#     print(l)








