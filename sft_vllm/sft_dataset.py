import torch.utils.data as data
import json
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import time
import random
import torch
from dataclasses import dataclass
from trl.data_utils import maybe_apply_chat_template

from .utils import create_prefix_mask, create_suffix_mask

@dataclass
class DataCollatorForDialog:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features, return_tensors=None):
        """
            Args: features[batch_data]:({'input_ids':[],'labels':[]},{},...)
            return {'input_ids':[],'attention_mask','labels':[]}
        """
        input_ids = [item['input_ids'].tolist() for item in features]
        attention_mask = []
        labels = [item['labels'].tolist() for item in features]

        input_max_length = max([len(item) for item in input_ids])
        label_max_length = max([len(item) for item in labels])
        
        for _input_ids, _labels in zip(input_ids, labels):
            attention_mask.append([1] * len(_input_ids) + [0] * (input_max_length - len(_input_ids)))
            _input_ids.extend([self.tokenizer.eos_token_id] * (input_max_length - len(_input_ids)))
            _labels.extend([-100] * (label_max_length - len(_labels)))

        ans = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(labels)
        }
        return ans

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
        
        prompts_text = [maybe_apply_chat_template({'messages': self.data[index]['completion'] }, self.tokenizer)["text"]]

        prompt_inputs = self.tokenizer(
            prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )['input_ids'] # 只训练前面的
        # only for single turn
        assistant_id = self.tokenizer('<｜Assistant｜>', add_special_tokens=False)['input_ids'][0]
        prefix_mask = create_prefix_mask(prompt_inputs, assistant_id)
        suffix_mask = create_suffix_mask(prompt_inputs, self.tokenizer.eos_token_id)
        
        mask = prefix_mask * suffix_mask

        labels = torch.where(mask == 0, torch.tensor(-100), prompt_inputs)

        return {'input_ids': prompt_inputs[0], 'labels': labels[0]}


# if __name__ == '__main__':
#     a = torch.LongTensor([[1,2,3],[1,2,3]])
#     m = torch.LongTensor([[0, 0, 1], [0, 1, 1]])
#     l = torch.where(m == 0, torch.tensor(-100), a)
#     print(l)








