
import re
import copy
import torch
from tqdm import tqdm

from trl.data_utils import maybe_apply_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from reason_llm.utils import *
from reason_llm.config import *
from reason_llm.reward_fn import *

SEP = '[SEP]'

VISUALIZATION_DICT = ['success','primary','info']

COLORS = [(255,0,255),(0,255,255),(255,255,0),(255,0,255),(0,255,255),(255,255,0)]

IGNORE_CHAR_SET = {'[SEP]'}

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

import torch

def masked_min_max_normalization(tensor, mask):
    """
    对 tensor 进行 masked min-max 归一化
    :param tensor: 输入的 tensor (batch, seq_len)
    :param mask: mask tensor (batch, seq_len)，1 表示有效，0 表示无效
    :return: 归一化后的 tensor
    """
    # 只在 mask 为 True 的位置计算 min 和 max
    masked_tensor = tensor.clone()  # 复制 tensor 避免修改原数据
    masked_tensor[mask == 0] = float('inf')  # 无效位置设为极大值
    min_vals = torch.min(masked_tensor, dim=1, keepdim=True)[0]  # (batch, 1)
    
    masked_tensor[mask == 0] = float('-inf')  # 无效位置设为极小值
    max_vals = torch.max(masked_tensor, dim=1, keepdim=True)[0]  # (batch, 1)

    # 避免除零错误
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 避免除以 0

    # 归一化
    normalized_tensor = (tensor - min_vals) / range_vals
    normalized_tensor[mask == 0] = 0  # 无效位置保持 0 或 NaN

    return normalized_tensor



class Visiuation:
    def __init__(self, model_id):
        self.hightlight_html = """
        <!DOCTYPE html> 
        <html lang="en"> 
        <head> 
            <meta charset="utf-8"> 
            <title>文本注意力可视化</title> 
            <meta name="description" content="Bootstrap Basic Tab Based Navigation Example">
            <!-- 新 Bootstrap 核心 CSS 文件 -->
            <link href="https://cdn.staticfile.org/twitter-bootstrap/3.3.7/css/bootstrap.min.css" rel="stylesheet">
            
        </head>
        <body>
            
        <div class="container">
            <div class="row">
                <div class="span6">
                    <ul class="nav nav-tabs" id="myTab" role="tablist">
                        
                    </ul>
                </div>
            </div>
            <div class="tab-content" id="myTabContent" style="padding: 10px;font-size: 25px;font-weight:bold;">
                
            </div>
        </div>
        </body>

        <!-- jQuery文件。务必在bootstrap.min.js 之前引入 -->
        <script src="https://cdn.staticfile.org/jquery/2.1.1/jquery.min.js"></script>
            
        <!-- 最新的 Bootstrap 核心 JavaScript 文件 -->
        <script src="https://cdn.staticfile.org/twitter-bootstrap/3.3.7/js/bootstrap.min.js"></script>
        </html>
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    def visual_text(self, text, attn, output_path):
        nav_head = ''
        nav_content = ''
        color_id = 0

        nav_head += """
            <li class="nav-item" role="presentation">
                <a class="nav-link" id="d1-tab" data-toggle="tab" href="#d1" role="tab" aria-controls="d1" aria-selected="false"><span class="label label-success">疾病</span></a>
            </li>
        """
        nav_content += """
            <div class="tab-pane fade" id="d1" role="tabpanel" aria-labelledby="d1-tab">
                <font face="楷体">
                    <p>
                        %s
                    </p>
                </font>
            </div>
        """%(self.field_visual(text, attn, COLORS[color_id]),)

        html = copy.deepcopy(self.hightlight_html)

        html = re.sub(r'<ul class=\"nav nav-tabs\" id=\"myTab\" role=\"tablist\">.*?</ul>','<ul class="nav nav-tabs" id="myTab" role="tablist">'+nav_head+'</ul>',html,flags =re.S)
        html = re.sub(r'<div class=\"tab-content\" id=\"myTabContent\" style=\"padding: 10px;font-size: 25px;font-weight:bold;\">.*?</div>','<div class="tab-content" id="myTabContent" style="padding: 10px;font-size: 25px;font-weight:bold;">'+nav_content+'</div>',html,flags =re.S)
        with open(output_path,'w',encoding='utf-8') as f:
            f.write(html)

    @staticmethod
    def field_visual(sentence,attn,color):
        span_str = ''
        for c,a in zip(sentence,attn):
            if c in IGNORE_CHAR_SET:
                continue
            span_str += """<span style="background-color:rgb(%d,%d,%d)">%s </span>"""%(255-(255-color[0])*a,\
                                                                255-(255-color[1])*a,255-(255-color[2])*a,c)
        return span_str

    def process(self, sentences):
        batch_size = 4

        for i in tqdm(range(0, len(sentences), batch_size)):
            prompts_text = [maybe_apply_chat_template({'messages': example['completion']}, self.tokenizer)["text"] \
                                for example in sentences[i:i+batch_size]]
            prompt_inputs = self.tokenizer(
                prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
            )['input_ids'].to('cuda')
            
            assistant_id = self.tokenizer(ASSISTANT_TOKEN, add_special_tokens=False)['input_ids'][0]

            pad_id = self.tokenizer.pad_token_id
            prefix_mask = create_prefix_mask(prompt_inputs, assistant_id)
            suffix_mask = create_suffix_mask(prompt_inputs, pad_id)

            mask = prefix_mask * suffix_mask
            with torch.inference_mode():
                logps = get_per_token_logps(self.model, prompt_inputs)
            
            possiblities = ((1.0 - torch.exp(logps)) * mask[:, 1:]).cpu().tolist()

            for j in range(len(prompts_text)):
                tokens = self.tokenizer.tokenize(prompts_text[j], add_special_tokens=False)
    
                # 逐个解码每个 token，确保 decoded_text 是 list
                decoded_text = [self.tokenizer.convert_tokens_to_string([token]) for token in tokens]

                # 去掉可能的前导空格
                decoded_text = [t.lstrip() for t in decoded_text]
                self.visual_text(decoded_text[1:], possiblities[j][:len(tokens) - 1], f"{i + j}.html")

if __name__ == '__main__':
    v = Visiuation('model/merge')
    with open('data/buffer.json','r') as f:
        data = json.load(f)
    for item in data:
        v.process([item])
        break