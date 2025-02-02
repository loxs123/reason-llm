

import json
from grpo_vllm import group_reward_fn

data_file = 'buffer_14b.json'
out_file = 'new_buffer_14.jsons'
sample_num = 8

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for i in range(0, len(data), sample_num):
    msgs = [m['completion'] for m in data[i:i + sample_num]]
    label = data[i]['label']
    reward = group_reward_fn(None, msgs, label)
    for m,r in zip(msgs, reward):
        new_data.append({'completion': m, 'advantage': r, 'label': label})

with (out_file, 'w') as f:
    json.dump(new_data, f, ensure_ascii=False,indent=2)


