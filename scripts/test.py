
import json
from transformers import AutoTokenizer

from trl.data_utils import apply_chat_template
from reason_llm.utils import *
from reason_llm.config import *

with open('/home/user/reason-llm/data/buffer.json', )as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('/home/user/reason-llm/model')

# print(len(data[0]['completion'][-1]['content']))

prompts_text = [tokenizer.apply_chat_template(example["completion"], tokenize=False) for example in data[:1]]
print(len(prompts_text[0]))
#   // "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"

prompt_inputs = tokenizer(
    prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
)['input_ids']
print(prompts_text[0])
print('#' * 20)
print(data[0]['completion'][-1]['content'])
print(len(tokenizer.tokenize(prompts_text[0])))
print(len(tokenizer.tokenize(data[0]['completion'][-1]['content'])))

print(prompt_inputs.shape)

assistant_id = tokenizer(ASSISTANT_TOKEN, add_special_tokens=False)['input_ids'][0] # for deepseek
tokenizer.pad_token_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
prefix_mask = create_prefix_mask(prompt_inputs, assistant_id)
suffix_mask = create_suffix_mask(prompt_inputs, pad_id)

mask = prefix_mask * suffix_mask

print(suffix_mask.sum(dim = 1))
print(prefix_mask.sum(dim = 1))
print(mask.sum(dim = 1))