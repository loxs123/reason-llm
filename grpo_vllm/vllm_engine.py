import time
import os
import csv
import copy
import json

import torch
from vllm import LLM, SamplingParams

from transformers import LogitsProcessor

current_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

last_time = time.time()
model_dir = os.path.join(current_dir, 'model')
buffer_file = os.path.join(current_dir, 'data', 'buffer.json')
data_file = os.path.join(current_dir, 'data', 'train.csv')

GPU_NUM = 1
MAX_MODEL_LEN = 8192

llm = LLM(
    model_dir,
    max_model_len=MAX_MODEL_LEN, # 4096*10,
    trust_remote_code=True,
    tensor_parallel_size=GPU_NUM,
    max_num_seqs = 32,
    gpu_memory_utilization=0.96, 
)

# 自定义 LogitsProcessor 类
class ThinkCountLogitsProcessor(LogitsProcessor):
    def __init__(self, keyword, tokenizer):

        self.kids = tokenizer.encode(keyword, add_special_tokens=False)
        self.ks = len(self.kids)
        self.eos_id = tokenizer.eos_token_id
        
        self.max_occurrences = 5 # 最大出现5次
        self.keyword_count = 0  # 记录关键词出现次数
    
    def __call__(self, input_ids, scores):

        _input_ids = list(input_ids)
        cnt = 0
        for j in range(len(_input_ids) - len(self.kids)):
            if _input_ids[j:j + self.ks] == self.kids:
                cnt += 1
                if cnt >= self.max_occurrences:
                    break

        if cnt == self.max_occurrences:
            scores[self.eos_id] += 1e5

        return scores


tokenizer = copy.deepcopy(llm.get_tokenizer())
keyword = '<think>'
# 初始化自定义 LogitsProcessor
think_processor = ThinkCountLogitsProcessor(keyword, tokenizer)

def batch_message_generate(list_of_messages, llm_model) -> list[list[dict]]:
        
    sampling_params = SamplingParams(
        temperature=1.0,
        min_p = 0.01,
        max_tokens=MAX_MODEL_LEN,
        skip_special_tokens=True,
        logits_processors=[think_processor]  # 添加自定义 LogitsProcessor
    )
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    
    request_output = llm_model.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )

    for i, single_request_output in enumerate(request_output):
        list_of_messages[i].append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

    return list_of_messages

cur_msgs = []

while True:
    with open(data_file, ) as f:
        lines = csv.DictReader(f)
    
    for line in lines:
        msgs = [[{'role': 'user', 'content': line['question']}]] * 32
        processed_msgs = batch_message_generate(msgs, llm)
        cur_msgs.append({'completion' : processed_msgs, 'label': line['answer']})

        # TODO: 可以从中挑选几个回答好的保存，保证奖励方差

        if os.path.getmtime(os.path.join(model_dir, 'config.json')) > last_time:
            with open(buffer_file,'w') as f:
                json.dump(cur_msgs, f, ensure_ascii=False, indent=2)

            cur_msgs.clear()
            del llm

            last_time = os.path.getmtime(os.path.join(model_dir, 'config.json'))
            llm = LLM(
                model_dir,
                max_model_len=MAX_MODEL_LEN, # 4096*10,
                trust_remote_code=True,
                tensor_parallel_size=GPU_NUM,
                max_num_seqs = 32,
                gpu_memory_utilization=0.96, 
            )



